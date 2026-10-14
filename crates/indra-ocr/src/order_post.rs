//! Reading order (order_post.py): line features for the order model, decoding
//! of its pairwise logits, XY-cut, and block clustering.

use crate::py::{argmax, argsort_by, is_digit, is_upper, median_f32, round_int, str_is_digit, strip};

pub const N_GEOM: usize = 10;
pub const N_TEXT: usize = 6;
pub const N_FEATS: usize = N_GEOM + N_TEXT;

pub const BLOCK_CLASSES: [&str; 10] =
    ["Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group", "Table", "Caption", "Footnote", "Table-Of-Contents", "Form"];
pub const GRID_LABELS: [&str; 3] = ["Table", "Form", "Table-Of-Contents"];

pub fn class_id(label: &str) -> usize {
    BLOCK_CLASSES.iter().position(|&c| c == label).unwrap_or(0)
}

/// A dense row-major float32 matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match its shape");
        Mat { rows, cols, data }
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    /// The submatrix of rows ``r`` and columns ``c`` (np.ix_).
    pub fn select(&self, r: &[usize], c: &[usize]) -> Mat {
        let mut d = Vec::with_capacity(r.len() * c.len());
        for &i in r {
            for &j in c {
                d.push(self.at(i, j));
            }
        }
        Mat::new(r.len(), c.len(), d)
    }
    /// Column sums over the rows ``r`` (``m[r].sum(0)``, row by row in float32).
    pub fn sum_rows(&self, r: &[usize]) -> Vec<f32> {
        let mut s = self.row(r[0]).to_vec();
        for &i in &r[1..] {
            for (a, b) in s.iter_mut().zip(self.row(i)) {
                *a += b;
            }
        }
        s
    }
}

/// [N, 4] boxes -> [N, N_FEATS] features (row-major).
pub fn line_features(boxes: &[[f32; 4]], page_wh: (usize, usize), texts: Option<&[String]>) -> Vec<f32> {
    let n = boxes.len();
    let mut out = vec![0f32; n * N_FEATS];
    if n == 0 {
        return out;
    }
    let (wf, hf) = (page_wh.0 as f32, page_wh.1 as f32);
    let hs: Vec<f32> = boxes.iter().map(|b| (b[3] - b[1]).max(1e-3)).collect();
    let med_h = median_f32(&hs) as f64;
    let last = (med_h / page_wh.1 as f64 * 20.0) as f32;
    for (i, b) in boxes.iter().enumerate() {
        let w = (b[2] - b[0]).max(1e-3);
        let h = hs[i];
        let f = &mut out[i * N_FEATS..(i + 1) * N_FEATS];
        f[0] = b[0] / wf;
        f[1] = b[1] / hf;
        f[2] = b[2] / wf;
        f[3] = b[3] / hf;
        f[4] = (b[0] + b[2]) / (2.0 * wf);
        f[5] = (b[1] + b[3]) / (2.0 * hf);
        f[6] = w / wf;
        f[7] = (h / med_h as f32).ln();
        f[8] = (w / h).ln() / 4.0;
        f[9] = last;
        let Some(texts) = texts else { continue };
        let t = strip(&texts[i]);
        if t.is_empty() {
            continue;
        }
        let chars: Vec<char> = t.chars().collect();
        let len = chars.len();
        f[10] = ((len as f64).ln_1p() / 5.0) as f32;
        f[11] = (chars.iter().filter(|c| is_digit(**c)).count() as f64 / len as f64) as f32;
        let head: String = chars.iter().take(3).collect();
        let head = head.trim_end_matches(['.', ')']);
        f[12] = ("•-–▪◦*·→".contains(chars[0]) || (str_is_digit(head) && len > 3)) as u8 as f32;
        f[13] = (chars[len - 1] == '-') as u8 as f32;
        f[14] = ".!?:;".contains(chars[len - 1]) as u8 as f32;
        f[15] = is_upper(chars[0]) as u8 as f32;
    }
    out
}

/// np.logaddexp(0, x) in float32.
fn logaddexp0(x: f32) -> f32 {
    if x == 0.0 {
        return std::f32::consts::LN_2;
    }
    let tmp = -x;
    if tmp > 0.0 {
        (-tmp).exp().ln_1p()
    } else if tmp <= 0.0 {
        x + tmp.exp().ln_1p()
    } else {
        tmp
    }
}

/// Order from a pairwise logit matrix (a[i, j] > 0: i before j): the row-sum
/// ranking, improved by insertion moves.
pub fn decode_order(a: &Mat, sweeps: usize) -> Vec<usize> {
    let n = a.rows;
    if n <= 1 {
        return (0..n).collect();
    }
    let lp: Vec<f32> = a.data.iter().map(|&v| -logaddexp0(-v)).collect();
    let lpm = Mat::new(n, n, lp);
    let rows: Vec<f32> = (0..n).map(|i| crate::py::pairwise_sum_f32(lpm.row(i))).collect();
    let cols = lpm.sum_rows(&(0..n).collect::<Vec<_>>());
    let score: Vec<f32> = (0..n).map(|i| -rows[i] + cols[i]).collect();
    let mut order = argsort_by(&score, |v| v as f64);
    let mut gains = vec![0f32; n];
    for _ in 0..sweeps {
        let mut improved = false;
        for x in order.clone() {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            let m = rest.len();
            // gains[k]: x inserted before rest[k]
            let mut acc = 0f32;
            gains[0] = 0.0;
            for k in 0..m {
                acc += lpm.at(rest[k], x);
                gains[k + 1] = acc;
            }
            let mut acc = 0f32;
            let mut suffix = vec![0f32; m + 1];
            for k in (0..m).rev() {
                acc += lpm.at(x, rest[k]);
                suffix[k] = acc;
            }
            for k in 0..=m {
                gains[k] += suffix[k];
            }
            let k = argmax(&gains[..=m]);
            let cur = order.iter().position(|&v| v == x).unwrap();
            if gains[k] > gains[cur] + 1e-6 {
                let mut o = rest;
                o.insert(k, x);
                order = o;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// Rows top to bottom (centres within half a line height share a row), left to right.
pub fn order_top_down(boxes: &[[f32; 4]]) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }
    let hs: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median_f32(&hs);
    let div = if h * 0.5 > 1e-3 { h * 0.5 } else { 1e-3f32 };
    let key = |i: usize| -> (i64, f32) {
        let cy = (boxes[i][1] + boxes[i][3]) / 2.0;
        (round_int((cy / div) as f64), boxes[i][0])
    };
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&a, &b| {
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
    });
    idx
}

/// Recursive XY-cut at the widest whitespace gap on either axis.
pub fn order_xycut(boxes: &[[f32; 4]]) -> Vec<usize> {
    fn best_cut(boxes: &[[f32; 4]], ids: &[usize], axis: usize) -> (f32, isize, Vec<usize>) {
        let (lo, hi) = if axis == 1 { (1, 3) } else { (0, 2) };
        let o = argsort_by(ids, |i| boxes[i][lo] as f64);
        let mut end = boxes[ids[o[0]]][hi];
        let (mut best_gap, mut best_k) = (0f32, -1isize);
        for k in 1..o.len() {
            let gap = boxes[ids[o[k]]][lo] - end;
            if gap > best_gap {
                best_gap = gap;
                best_k = k as isize;
            }
            end = end.max(boxes[ids[o[k]]][hi]);
        }
        (best_gap, best_k, o)
    }
    fn rec(boxes: &[[f32; 4]], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let (gy, ky, oy) = best_cut(boxes, &ids, 1);
        let (gx, kx, ox) = best_cut(boxes, &ids, 0);
        if ky < 0 && kx < 0 {
            let b: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
            return order_top_down(&b).into_iter().map(|i| ids[i]).collect();
        }
        let (k, o) = if gy >= gx { (ky as usize, oy) } else { (kx as usize, ox) };
        let mut out = rec(boxes, o[..k].iter().map(|&i| ids[i]).collect());
        out.extend(rec(boxes, o[k..].iter().map(|&i| ids[i]).collect()));
        out
    }
    rec(boxes, (0..boxes.len()).collect())
}

/// Average-linkage clustering of pairwise "same group" logits: merge the two
/// groups with the highest mean logit while it is positive.
pub fn cluster(logits: &Mat) -> Vec<Vec<usize>> {
    let n = logits.rows;
    if n == 0 {
        return vec![];
    }
    let mut s: Vec<f64> = logits.data.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        s[i * n + i] = 0.0;
    }
    let mut size = vec![1f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m = s.clone();
    for i in 0..n {
        m[i * n + i] = f64::NEG_INFINITY;
    }
    let row_argmax = |m: &[f64], j: usize| argmax(&m[j * n..(j + 1) * n]);
    let mut best: Vec<usize> = (0..n).map(|j| row_argmax(&m, j)).collect();
    let mut best_v: Vec<f64> = (0..n).map(|j| m[j * n + best[j]]).collect();
    for _ in 0..n.saturating_sub(1) {
        let a0 = argmax(&best_v);
        let b0 = best[a0];
        if best_v[a0] <= 0.0 {
            break;
        }
        let (a, b) = (a0.min(b0), a0.max(b0));
        for j in 0..n {
            s[a * n + j] += s[b * n + j];
        }
        for i in 0..n {
            s[i * n + a] += s[i * n + b];
        }
        s[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let mb = std::mem::take(&mut members[b]);
        members[a].extend(mb);
        let mut row: Vec<f64> = (0..n).map(|j| s[a * n + j] / (size[a] * size[j])).collect();
        for j in 0..n {
            if !alive[j] {
                row[j] = f64::NEG_INFINITY;
            }
        }
        row[a] = f64::NEG_INFINITY;
        for j in 0..n {
            m[a * n + j] = row[j];
            m[j * n + a] = row[j];
        }
        for j in 0..n {
            m[b * n + j] = f64::NEG_INFINITY;
            m[j * n + b] = f64::NEG_INFINITY;
        }
        best_v[b] = f64::NEG_INFINITY;
        best[a] = argmax(&row);
        best_v[a] = row[best[a]];
        for j in 0..n {
            if alive[j] && j != a && (best[j] == a || best[j] == b) {
                best[j] = row_argmax(&m, j);
                best_v[j] = m[j * n + best[j]];
            }
        }
        for j in 0..n {
            if alive[j] && j != a && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }
    (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            let mut v = members[i].clone();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Reading order with the layout heads: blocks by clustering the same-block
/// logits, lines inside a block by XY-cut (row by row in grid blocks), blocks by
/// the mean pairwise order logit between them.
pub fn decode_blocks_order(a: &Mat, same: &Mat, boxes: &[[f32; 4]], cls: Option<&Mat>) -> Vec<usize> {
    let n = a.rows;
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(same);
    let grid: Vec<usize> = GRID_LABELS.iter().map(|l| class_id(l)).collect();
    let mut inner = Vec::with_capacity(blocks.len());
    for b in &blocks {
        let bx: Vec<[f32; 4]> = b.iter().map(|&i| boxes[i]).collect();
        let is_grid = cls.is_some_and(|c| grid.contains(&argmax(&c.sum_rows(b))));
        let local = if is_grid { order_top_down(&bx) } else { order_xycut(&bx) };
        inner.push(local.into_iter().map(|k| b[k]).collect::<Vec<_>>());
    }
    let m = blocks.len();
    let mut ab = Mat::zeros(m, m);
    for p in 0..m {
        for q in p + 1..m {
            let mut s = 0f64;
            for &i in &blocks[p] {
                for &j in &blocks[q] {
                    s += a.at(i, j) as f64;
                }
            }
            let v = s / (blocks[p].len() * blocks[q].len()) as f64;
            ab.data[p * m + q] = v as f32;
            ab.data[q * m + p] = (-v) as f32;
        }
    }
    decode_order(&ab, 4).into_iter().flat_map(|k| inner[k].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xycut_two_columns() {
        let boxes = [[0., 0., 10., 2.], [20., 0., 30., 2.], [0., 3., 10., 5.], [20., 3., 30., 5.]];
        assert_eq!(order_xycut(&boxes), vec![0, 2, 1, 3]);
    }

    #[test]
    fn cluster_groups() {
        let l = Mat::new(3, 3, vec![0., 5., -5., 5., 0., -5., -5., -5., 0.]);
        assert_eq!(cluster(&l), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn order_from_logits() {
        // 2 before 0 before 1
        let a = Mat::new(3, 3, vec![0., 4., -4., -4., 0., -4., 4., 4., 0.]);
        assert_eq!(decode_order(&a, 4), vec![2, 0, 1]);
    }
}
