//! Waiting for an accelerator, in the two ways this crate has to.
//!
//! A native program may simply stop until the graphics card has answered. A web page may not: the browser hands
//! results back only when the code that asked has returned to it. So everything that may have to wait for a card
//! is written once, as an `async` function (`fit::fit_raw_async`, `predict::predict_raw_async`, the accelerator
//! hooks), and comes in two forms:
//!
//! * the `_async` form, for a browser, where wasm-bindgen-futures drives it;
//! * the plain form, which is `block_on` of the other. Natively the card's own code waits for its answer before
//!   it returns (see `gpu`), so such a call is finished the first time it is asked; without an accelerator there
//!   is nothing to wait for at all. Either way the plain forms cost nothing extra.
//!
//! The futures here are not `Send`: a call runs on the thread that started it.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

/// A boxed future, for the accelerator traits (a trait object's methods cannot be `async fn`).
pub type Later<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// A value that is ready now, as a `Later`.
pub fn ready<'a, T: 'a>(v: T) -> Later<'a, T> {
    Box::pin(std::future::ready(v))
}

struct Idle;

impl Wake for Idle {
    fn wake(self: Arc<Self>) {}
}

/// Run `f` to its end on this thread.
///
/// In a browser this panics if `f` really has to wait: there, call the `_async` form from JavaScript instead.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let waker = Waker::from(Arc::new(Idle));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        #[cfg(target_arch = "wasm32")]
        panic!("a blocking call had to wait for an accelerator; in a browser use the _async form");
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::yield_now();
    }
}

struct Shared<T> {
    value: Option<T>,
    waker: Option<Waker>,
    /// The sending side went away without sending.
    gone: bool,
}

/// The receiving end of a value that arrives once, later, from a callback. Awaiting it gives `Some(value)`, or
/// `None` if the callback was dropped without being called.
pub struct Once<T>(Arc<Mutex<Shared<T>>>);

/// The sending end of a `Once`.
pub struct Send1<T>(Arc<Mutex<Shared<T>>>);

pub fn once<T>() -> (Send1<T>, Once<T>) {
    let s = Arc::new(Mutex::new(Shared { value: None, waker: None, gone: false }));
    (Send1(s.clone()), Once(s))
}

impl<T> Send1<T> {
    pub fn send(self, v: T) {
        let mut s = self.0.lock().unwrap();
        s.value = Some(v);
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    }
}

impl<T> Drop for Send1<T> {
    fn drop(&mut self) {
        let mut s = self.0.lock().unwrap();
        s.gone = true;
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    }
}

impl<T> Future for Once<T> {
    type Output = Option<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let mut s = self.0.lock().unwrap();
        if let Some(v) = s.value.take() {
            return Poll::Ready(Some(v));
        }
        if s.gone {
            return Poll::Ready(None);
        }
        s.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
