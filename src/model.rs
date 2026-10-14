//! The pretrained weights, as plain arrays, and the sizes that describe the network.
//!
//! Only the TabICLv2 classifier layout is accepted for now: grouped features (3 per group, circular), target-aware
//! column embedding, non-interleaved rotary positions in the row stage, pre-norm blocks, GELU, and the
//! "qassmax-mlp-elementwise" scalable softmax. `Model::load` refuses anything else rather than guess.

use crate::store::Bundle;
use std::io;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Scalable softmax ("QASSMax"): queries are multiplied by `base(log n) * (1 + tanh(query_mlp(q)))`.
pub struct Ssmax {
    pub b0w: Vec<f32>,
    pub b0b: Vec<f32>,
    pub b2w: Vec<f32>,
    pub b2b: Vec<f32>,
    pub q0w: Vec<f32>,
    pub q0b: Vec<f32>,
    pub q2w: Vec<f32>,
    pub q2b: Vec<f32>,
}

/// One pre-norm attention block: `x += attn(norm1 x); x += l2(gelu(l1(norm2 x)))`.
pub struct Block {
    pub d: usize,
    pub heads: usize,
    pub ff: usize,
    pub n1w: Vec<f32>,
    pub n1b: Vec<f32>,
    pub n2w: Vec<f32>,
    pub n2b: Vec<f32>,
    /// Query, key and value projections stacked: `[3d][d]` (or just the query's `[d][d]` if loaded `q_only`).
    pub in_w: Vec<f32>,
    pub in_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub ss: Option<Ssmax>,
}

impl Block {
    pub fn hd(&self) -> usize {
        self.d / self.heads
    }

    pub fn load(b: &Bundle, p: &str, heads: usize, q_only: bool) -> io::Result<Block> {
        let g = |n: &str| b.f32s(&format!("{p}.{n}"));
        let n1w = g("norm1.weight")?;
        let d = n1w.len();
        // some checkpoints train their layer norms without a bias
        let opt = |n: &str| if b.has(&format!("{p}.{n}")) { g(n) } else { Ok(vec![0.0f32; d]) };
        let (mut in_w, mut in_b) = (g("attn.in_proj_weight")?, g("attn.in_proj_bias")?);
        if q_only {
            in_w.truncate(d * d);
            in_w.shrink_to_fit();
            in_b.truncate(d);
        }
        let ss = if b.has(&format!("{p}.attn.ssmax_layer.base_mlp.0.weight")) {
            let s = |n: &str| g(&format!("attn.ssmax_layer.{n}"));
            Some(Ssmax {
                b0w: s("base_mlp.0.weight")?,
                b0b: s("base_mlp.0.bias")?,
                b2w: s("base_mlp.2.weight")?,
                b2b: s("base_mlp.2.bias")?,
                q0w: s("query_mlp.0.weight")?,
                q0b: s("query_mlp.0.bias")?,
                q2w: s("query_mlp.2.weight")?,
                q2b: s("query_mlp.2.bias")?,
            })
        } else {
            None
        };
        let l1b = g("linear1.bias")?;
        Ok(Block {
            d,
            heads,
            ff: l1b.len(),
            n1w,
            n1b: opt("norm1.bias")?,
            n2w: g("norm2.weight")?,
            n2b: opt("norm2.bias")?,
            in_w,
            in_b,
            out_w: g("attn.out_proj.weight")?,
            out_b: g("attn.out_proj.bias")?,
            l1w: g("linear1.weight")?,
            l1b,
            l2w: g("linear2.weight")?,
            l2b: g("linear2.bias")?,
            ss,
        })
    }
}

/// One induced set-attention block of the column stage.
pub struct Isab {
    /// Inducing points attend to the column's training cells (fitting only).
    pub a1: Option<Block>,
    /// Every cell attends to the inducing points' result (cached per column at fit time).
    pub a2: Block,
    pub ind: Vec<f32>,
}

pub struct Model {
    /// Width of the column and row stages (128).
    pub e: usize,
    /// Width of the in-context stage (512 = e * cls).
    pub d: usize,
    pub group: usize,
    pub inds: usize,
    pub cls: usize,
    pub max_classes: usize,
    pub col_heads: usize,
    pub row_heads: usize,
    pub icl_heads: usize,
    pub in_w: Vec<f32>,
    pub in_b: Vec<f32>,
    pub col: Vec<Isab>,
    /// Target embedding added to training cells in the column stage, one row of `e` per class.
    pub col_y: Vec<f32>,
    pub cls_tokens: Vec<f32>,
    pub row: Vec<Block>,
    pub rope_freqs: Vec<f32>,
    pub out_lnw: Vec<f32>,
    pub out_lnb: Vec<f32>,
    /// In-context blocks; empty when loaded for streamed prediction (see `predict::IclWeights`).
    pub icl: Vec<Block>,
    /// Target embedding added to training rows in the in-context stage, one row of `d` per class.
    pub icl_y: Vec<f32>,
    pub icl_layers: usize,
    pub lnw: Vec<f32>,
    pub lnb: Vec<f32>,
    pub dec0w: Vec<f32>,
    pub dec0b: Vec<f32>,
    pub dec2w: Vec<f32>,
    pub dec2b: Vec<f32>,
}

/// How much of the weight file to hold in memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Load {
    /// Everything: needed to fit.
    Fit,
    /// Everything prediction uses (about 85 MB, shared by all threads).
    Predict,
    /// Only the small column and row stages (about 4 MB); the in-context weights are read from disk as they are
    /// needed, a tensor at a time.
    PredictStreamed,
}

fn cfg_value<'a>(cfg: &'a str, key: &str) -> Option<&'a str> {
    let at = cfg.find(&format!("\"{key}\""))?;
    let rest = cfg[at..].split_once(':')?.1;
    Some(rest.split([',', '}']).next()?.trim().trim_matches('"'))
}

/// `[out][classes]` weight and `[out]` bias of a one-hot linear layer, turned into one embedding row per class.
fn class_table(w: &[f32], b: &[f32], classes: usize) -> Vec<f32> {
    let out = b.len();
    if classes == 0 {
        // regression: the target is a number; keep the layer as `[weight][bias]`
        return w.iter().chain(b).copied().collect();
    }
    let mut t = vec![0.0f32; classes * out];
    for c in 0..classes {
        for o in 0..out {
            t[c * out + o] = w[o * classes + c] + b[o];
        }
    }
    t
}

impl Model {
    /// Outputs the decoder produces: class scores, or quantiles for a regression model.
    pub fn outputs(&self) -> usize {
        self.dec2b.len()
    }

    /// Add the embedding of target `y` (a class index, or a standardised number for regression) to `cell`,
    /// using `table` (`col_y` or `icl_y`).
    #[inline(always)]
    pub fn add_y(&self, table: &[f32], y: f32, cell: &mut [f32]) {
        let w = cell.len();
        if self.max_classes > 0 {
            let row = &table[y as usize * w..(y as usize + 1) * w];
            for i in 0..w {
                cell[i] += row[i];
            }
        } else {
            for i in 0..w {
                cell[i] += table[i] * y + table[w + i];
            }
        }
    }

    pub fn load(bundle: &Bundle, how: Load) -> io::Result<Model> {
        let cfg = String::from_utf8(bundle.bytes("config.json")?).map_err(|e| bad(e.to_string()))?;
        let want = [
            ("col_feature_group", "same"),
            ("col_affine", "false"),
            ("col_target_aware", "true"),
            ("col_ssmax", "qassmax-mlp-elementwise"),
            ("icl_ssmax", "qassmax-mlp-elementwise"),
            ("row_rope_interleaved", "false"),
            ("norm_first", "true"),
            ("activation", "gelu"),
        ];
        for (k, v) in want {
            let got = cfg_value(&cfg, k).unwrap_or("(missing)");
            if got != v {
                return Err(bad(format!("unsupported checkpoint: {k} is {got}, this crate implements {v}")));
            }
        }
        let num = |k: &str| -> io::Result<usize> { cfg_value(&cfg, k).and_then(|v| v.parse().ok()).ok_or_else(|| bad(format!("config has no number {k}"))) };
        let (max_classes, cls) = (num("max_classes")?, num("row_num_cls")?);
        let (col_heads, row_heads, icl_heads) = (num("col_nhead")?, num("row_nhead")?, num("icl_nhead")?);
        let in_b = bundle.f32s("col_embedder.in_linear.bias")?;
        let e = in_b.len();
        let fit = how == Load::Fit;
        let mut col = Vec::new();
        for i in 0..num("col_num_blocks")? {
            let p = format!("col_embedder.tf_col.blocks.{i}");
            col.push(Isab {
                a1: if fit { Some(Block::load(bundle, &format!("{p}.multihead_attn1"), col_heads, false)?) } else { None },
                a2: Block::load(bundle, &format!("{p}.multihead_attn2"), col_heads, !fit)?,
                ind: if fit { bundle.f32s(&format!("{p}.ind_vectors"))? } else { Vec::new() },
            });
        }
        let row = (0..num("row_num_blocks")?).map(|i| Block::load(bundle, &format!("row_interactor.tf_row.blocks.{i}"), row_heads, false)).collect::<io::Result<Vec<_>>>()?;
        let icl_layers = num("icl_num_blocks")?;
        let icl = if how == Load::PredictStreamed {
            Vec::new()
        } else {
            (0..icl_layers).map(|i| Block::load(bundle, &format!("icl_predictor.tf_icl.blocks.{i}"), icl_heads, !fit)).collect::<io::Result<Vec<_>>>()?
        };
        let lnw = bundle.f32s("icl_predictor.ln.weight")?;
        Ok(Model {
            e,
            d: lnw.len(),
            group: num("col_feature_group_size")?,
            inds: num("col_num_inds")?,
            cls,
            max_classes,
            col_heads,
            row_heads,
            icl_heads,
            in_w: bundle.f32s("col_embedder.in_linear.weight")?,
            in_b,
            col,
            col_y: class_table(&bundle.f32s("col_embedder.y_encoder.weight")?, &bundle.f32s("col_embedder.y_encoder.bias")?, max_classes),
            cls_tokens: bundle.f32s("row_interactor.cls_tokens")?,
            row,
            rope_freqs: bundle.f32s("row_interactor.tf_row.rope.freqs")?,
            out_lnw: bundle.f32s("row_interactor.out_ln.weight")?,
            out_lnb: if bundle.has("row_interactor.out_ln.bias") { bundle.f32s("row_interactor.out_ln.bias")? } else { vec![0.0; e] },
            icl,
            icl_y: class_table(&bundle.f32s("icl_predictor.y_encoder.weight")?, &bundle.f32s("icl_predictor.y_encoder.bias")?, max_classes),
            icl_layers,
            lnw,
            lnb: if bundle.has("icl_predictor.ln.bias") { bundle.f32s("icl_predictor.ln.bias")? } else { vec![0.0; cls * e] },
            dec0w: bundle.f32s("icl_predictor.decoder.0.weight")?,
            dec0b: bundle.f32s("icl_predictor.decoder.0.bias")?,
            dec2w: bundle.f32s("icl_predictor.decoder.2.weight")?,
            dec2b: bundle.f32s("icl_predictor.decoder.2.bias")?,
        })
    }
}
