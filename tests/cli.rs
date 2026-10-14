//! The command-line program, run as a user would run it, on the tiny committed model: convert a PyTorch
//! checkpoint, fit a CSV table, predict, seal with a key, split and join, and the benchmark commands. Checks the
//! files and the CSV it prints, and that mistakes end with an error status rather than a crash or silence.

mod common;
use common::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tabicl-anywhere")).args(args).output().unwrap()
}

fn ok(args: &[&str]) -> String {
    let o = run(args);
    assert!(o.status.success(), "{args:?} failed:\n{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

fn fails(args: &[&str]) -> String {
    let o = run(args);
    assert!(!o.status.success(), "{args:?} should have failed");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

struct Dir(PathBuf);
impl Dir {
    fn new() -> Dir {
        let d = scratch("cli");
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn p(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write a fixture table as CSV: features, a text column, and the target last. Returns the target labels.
fn write_csv(fx: &tabicl_anywhere::store::Bundle, path: &Path, test_path: &Path, numeric: bool) -> Vec<String> {
    let t = table(fx);
    let label = |i: usize| if numeric { format!("{}", t.yf[i]) } else { ["no", "yes", "maybe"][t.y[i] % 3].to_string() };
    let cell = |v: f64| if v.is_nan() { String::new() } else { format!("{v}") };
    let head: Vec<String> = (0..t.h).map(|j| format!("f{j}")).chain(["colour".into(), "y".into()]).collect();
    let mut s = head.join(",") + "\n";
    let mut labels = Vec::new();
    for r in 0..t.n {
        labels.push(label(r));
        s += &format!("{},{},{}\n", (0..t.h).map(|j| cell(t.x[r * t.h + j])).collect::<Vec<_>>().join(","), ["red", "green"][r % 2], label(r));
    }
    std::fs::write(path, s).unwrap();
    let mut s = head[..t.h + 1].join(",") + "\n";
    for r in 0..t.nt {
        s += &format!("{},\"blue, new\"\n", (0..t.h).map(|j| cell(t.xt[r * t.h + j])).collect::<Vec<_>>().join(","));
    }
    std::fs::write(test_path, s).unwrap();
    labels
}

#[test]
fn convert_fit_predict_seal() {
    let d = Dir::new();
    let ck = root().join("tests/data/tiny-clf.ckpt");
    ok(&["convert", ck.to_str().unwrap(), &d.p("w.tirs")]);
    ok(&["convert", &d.p("w.tirs"), &d.p("w-half.tirs"), "--half"]);
    assert!(tabicl_anywhere::store::Bundle::open(d.p("w-half.tirs")).is_ok());
    let fx = data("tiny-table");
    let labels = write_csv(&fx, d.0.join("train.csv").as_path(), d.0.join("new.csv").as_path(), false);
    ok(&["fit", &d.p("w.tirs"), &d.p("train.csv"), "--target", "y", &d.p("m.tirs"), "--estimators", "2", "--kv", "i16", "--chunk", "50"]);
    let streamed = ok(&["predict", &d.p("w.tirs"), &d.p("m.tirs"), &d.p("new.csv")]);
    let resident = ok(&["predict", &d.p("w.tirs"), &d.p("m.tirs"), &d.p("new.csv"), "--resident", "--key-threads", "2", "--stats"]);
    let lines: Vec<&str> = streamed.lines().collect();
    let mut distinct = labels.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(lines[0], format!("prediction,{}", distinct.iter().map(|l| format!("p({l})")).collect::<Vec<_>>().join(",")));
    assert_eq!(lines.len(), 1 + table(&fx).nt, "a header and one line per new row");
    for l in &lines[1..] {
        let f: Vec<&str> = l.split(',').collect();
        assert!(distinct.iter().any(|x| x == f[0]), "the predicted label is one of the training labels");
        let p: Vec<f64> = f[1..].iter().map(|v| v.parse().unwrap()).collect();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-5);
    }
    // weights streamed from disk or held in memory, with or without extra reader threads: the same CSV to 1e-6
    for (a, b) in streamed.lines().zip(resident.lines()).skip(1) {
        let (a, b): (Vec<&str>, Vec<&str>) = (a.split(',').collect(), b.split(',').collect());
        assert_eq!(a[0], b[0]);
        assert!(a[1..].iter().zip(&b[1..]).all(|(x, y)| (x.parse::<f64>().unwrap() - y.parse::<f64>().unwrap()).abs() <= 2e-6));
    }
    // sealed with a key: the same answers; without the key, or with another key, an error
    ok(&["keygen", &d.p("key.txt")]);
    fails(&["keygen", &d.p("key.txt")]); // never overwrites a key
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(d.p("key.txt")).unwrap().permissions().mode() & 0o777, 0o600, "a key is readable by its owner only");
    }
    ok(&["seal", &d.p("m.tirs"), &d.p("m.sealed"), &d.p("key.txt")]);
    assert_eq!(ok(&["predict", &d.p("w.tirs"), &d.p("m.sealed"), &d.p("new.csv"), "--key", &d.p("key.txt")]), streamed);
    fails(&["predict", &d.p("w.tirs"), &d.p("m.sealed"), &d.p("new.csv")]);
    ok(&["keygen", &d.p("other.txt")]);
    let e = fails(&["predict", &d.p("w.tirs"), &d.p("m.sealed"), &d.p("new.csv"), "--key", &d.p("other.txt")]);
    assert!(e.contains("wrong key"), "{e}");
    // a table missing a column the model was fitted on
    std::fs::write(d.p("bad.csv"), "f0,colour\n1,red\n").unwrap();
    assert!(fails(&["predict", &d.p("w.tirs"), &d.p("m.tirs"), &d.p("bad.csv")]).contains("no column named"));
    assert!(fails(&["fit", &d.p("w.tirs"), &d.p("train.csv"), "--target", "nope", &d.p("m2.tirs")]).contains("no column named nope"));
}

#[test]
fn regression_from_the_command_line() {
    let d = Dir::new();
    ok(&["convert", root().join("tests/data/tiny-reg.ckpt").to_str().unwrap(), &d.p("r.tirs")]);
    let fx = data("tiny-reg-a");
    write_csv(&fx, d.0.join("train.csv").as_path(), d.0.join("new.csv").as_path(), true);
    ok(&["fit", &d.p("r.tirs"), &d.p("train.csv"), "--target", "y", &d.p("m.tirs"), "--estimators", "2"]);
    let out = ok(&["predict", &d.p("r.tirs"), &d.p("m.tirs"), &d.p("new.csv"), "--stats"]);
    let mut lines = out.lines();
    assert_eq!(lines.next(), Some("prediction,median,q05,q95"));
    for l in lines {
        let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
        assert!(v[2] <= v[1] && v[1] <= v[3], "q05 <= median <= q95: {l}");
    }
}

#[test]
fn split_and_join() {
    let d = Dir::new();
    let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(d.p("big.bin"), &data).unwrap();
    ok(&["split", &d.p("big.bin"), &d.p("parts"), "--max-mb", "1"]);
    assert!(Path::new(&d.p("parts.02.bin")).exists() && !Path::new(&d.p("parts.03.bin")).exists());
    ok(&["join", &d.p("parts.json"), &d.p("back.bin")]);
    assert_eq!(std::fs::read(d.p("back.bin")).unwrap(), data);
    std::fs::write(d.p("parts.01.bin"), b"short").unwrap();
    assert!(fails(&["join", &d.p("parts.json"), &d.p("back2.bin")]).contains("parts list says"));
}

#[test]
fn benchmark_commands_run() {
    let d = Dir::new();
    ok(&["convert", root().join("tests/data/tiny-clf.ckpt").to_str().unwrap(), &d.p("w.tirs")]);
    assert!(ok(&["bench-synth", "200", "16", "5", "3", &d.p("t.tirs")]).contains("200 training rows"));
    for kv in ["f32", "i8"] {
        assert!(ok(&["bench-fit", &d.p("w.tirs"), &d.p("t.tirs"), &d.p(&format!("c-{kv}.tirs")), "--kv", kv, "--mem", "0"]).contains("fit 200 x 5, 3 classes"));
    }
    let same = ok(&["bench-predict", &d.p("w.tirs"), &d.p("c-f32.tirs"), &d.p("t.tirs"), "--against", &d.p("c-f32.tirs"), "--stages", "--warm", "16", "--repeat", "2"]);
    assert!(same.contains("worst logit difference 0.0000, same answer on 16/16 rows"), "{same}");
    let one = ok(&["bench-predict", &d.p("w.tirs"), &d.p("c-i8.tirs"), &d.p("t.tirs"), "--resident", "--one-at-a-time", "--rows", "4", "--io", "4096", "--key-threads", "2"]);
    assert!(one.contains("predict 4 rows"), "{one}");
    assert!(ok(&["bench-attention", &d.p("w.tirs"), &d.p("c-f32.tirs"), &d.p("t.tirs"), "--rows", "3"]).contains("layer"));
    if !cfg!(feature = "gpu") {
        assert!(fails(&["bench-fit", &d.p("w.tirs"), &d.p("t.tirs"), &d.p("c.tirs"), "--gpu"]).contains("rebuild with --features gpu"));
    }
}

#[test]
fn usage_and_mistakes() {
    assert!(run(&[]).status.success(), "no command: the usage, successfully");
    assert!(String::from_utf8_lossy(&run(&["--help"]).stderr).contains("usage"));
    assert!(fails(&["frobnicate"]).contains("unknown command 'frobnicate'"));
    fails(&["convert", "/no/such/file.ckpt", "/no/such/out.tirs"]);
    let d = Dir::new();
    std::fs::write(d.p("junk.ckpt"), b"junk").unwrap();
    assert!(fails(&["convert", &d.p("junk.ckpt"), &d.p("out.tirs")]).contains("not a zip"));
}
