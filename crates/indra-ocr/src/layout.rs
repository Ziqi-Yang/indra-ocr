//! Page text from the ordered lines and the layout heads (layout_post.py):
//! blocks along the reading order, rendered by class (headings, paragraphs,
//! lists, HTML tables; page headers and footers left out by default).

use crate::order_post::{BLOCK_CLASSES, Mat, class_id, cluster};
use crate::py::{argmax, is_alpha, is_decimal, is_lower, is_space, median_f32, median_f64, round1, strip};

fn cls_of(cls: &Mat, blk: &[usize]) -> usize {
    argmax(&cls.sum_rows(blk))
}

/// Han, kana, CJK punctuation and fullwidth forms: written without spaces.
pub fn is_cjk(c: char) -> bool {
    let o = c as u32;
    (0x3000..=0x30FF).contains(&o)
        || (0x3400..=0x9FFF).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xFF00..=0xFFEF).contains(&o)
        || (0x20000..=0x2FA1F).contains(&o)
}

/// The lines of one paragraph joined: a hyphen at a line end before a
/// lowercase letter joins the word; CJK lines join without a space.
pub fn join_lines<S: AsRef<str>>(texts: &[S]) -> String {
    let mut out = String::new();
    for t in texts {
        let t = strip(t.as_ref());
        let Some(first) = t.chars().next() else { continue };
        let mut rev = out.chars().rev();
        let (last, prev) = (rev.next(), rev.next());
        if out.chars().count() > 1 && last == Some('-') && prev.is_some_and(is_alpha) && is_alpha(first) && is_lower(first) {
            out.pop();
            out.push_str(t);
        } else if out.is_empty() || (last.is_some_and(is_cjk) && is_cjk(first)) {
            out.push_str(t);
        } else {
            out.push(' ');
            out.push_str(t);
        }
    }
    out
}

/// Runs of the reading order with a positive same-block logit between neighbours.
pub fn split_blocks(order: &[usize], same: &Mat) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = vec![];
    for &i in order {
        if let Some(b) = blocks.last_mut()
            && same.at(*b.last().unwrap(), i) > 0.0
        {
            b.push(i);
            continue;
        }
        blocks.push(vec![i]);
    }
    blocks
}

/// html.escape(text, quote=False)
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// ^\$[^$]+\$$ : a line that is one formula.
fn whole_math(t: &str) -> bool {
    let t = t.strip_suffix('\n').unwrap_or(t);
    t.len() >= 3 && t.starts_with('$') && t.ends_with('$') && !t[1..t.len() - 1].contains('$')
}

/// A list item marker at the start of a line (layout_post._ITEM).
fn is_item(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let mut i = 0;
    while i < c.len() && is_space(c[i]) {
        i += 1;
    }
    let boundary = |j: usize| j >= c.len() || is_space(c[j]);
    let at = |j: usize| c.get(j).copied();
    // bullet
    if at(i).is_some_and(|ch| "•-–▪◦*·→".contains(ch)) && boundary(i + 1) {
        return true;
    }
    for start in [i, i + 1] {
        if start == i + 1 && at(i) != Some('(') {
            continue;
        }
        // \d{1,3}[.)]
        let mut j = start;
        while j < c.len() && j - start < 3 && is_decimal(c[j]) {
            j += 1;
        }
        if j > start && matches!(at(j), Some('.' | ')')) && boundary(j + 1) {
            return true;
        }
        // [a-zA-Z][.)]
        if at(start).is_some_and(|ch| ch.is_ascii_alphabetic()) && matches!(at(start + 1), Some('.' | ')')) && boundary(start + 2) {
            return true;
        }
    }
    false
}

fn col(boxes: &[[f32; 4]], idx: &[usize], k: usize) -> Vec<f32> {
    idx.iter().map(|&i| boxes[i][k]).collect()
}

fn median_height(boxes: &[[f32; 4]]) -> f32 {
    median_f32(&boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>())
}

/// Cell lines grouped into rows by vertical centre, top to bottom, left to right.
fn rows_of(boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let h = median_height(boxes) as f64;
    let mut rows: Vec<Vec<usize>> = vec![];
    for i in crate::py::argsort_by(&cy, |v| v as f64) {
        if let Some(r) = rows.last_mut() {
            let mean = crate::py::pairwise_sum_f32(&r.iter().map(|&k| cy[k]).collect::<Vec<_>>()) / r.len() as f32;
            if (cy[i] - mean).abs() < (0.5 * h) as f32 {
                r.push(i);
                continue;
            }
        }
        rows.push(vec![i]);
    }
    for r in rows.iter_mut() {
        r.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]));
    }
    rows
}

/// Line rows of wrapped cells joined into one table row.
fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let h = median_height(boxes) as f64;
    let gaps: Vec<f64> = rows
        .windows(2)
        .map(|w| {
            let top = w[1].iter().map(|&k| boxes[k][1]).fold(f32::INFINITY, f32::min);
            let bot = w[0].iter().map(|&k| boxes[k][3]).fold(f32::NEG_INFINITY, f32::max);
            (top - bot) as f64
        })
        .collect();
    let med = median_f64(&gaps);
    let mut out = vec![rows[0].clone()];
    let mut prev = &rows[0];
    for (g, r) in gaps.iter().zip(&rows[1..]) {
        if *g < 0.5 * med && *g < 0.6 * h && r.len() < prev.len() {
            out.last_mut().unwrap().extend(r);
        } else {
            out.push(r.clone());
        }
        prev = r;
    }
    out
}

fn row_html(tag: &str, cells: &std::collections::BTreeMap<usize, (usize, String)>, width: usize) -> String {
    let mut s = String::from("<tr>");
    let mut c = 0;
    while c < width {
        if let Some((n, text)) = cells.get(&c) {
            let attr = if *n > 1 { format!(" colspan=\"{n}\"") } else { String::new() };
            s += &format!("<{tag}{attr}>{}</{tag}>", html_escape(text));
            c += n;
        } else {
            s += &format!("<{tag}></{tag}>");
            c += 1;
        }
    }
    s + "</tr>"
}

/// An HTML table from cell lines (one text line per cell). A first row that
/// is one line across most of the table is a title before the table.
pub fn table_html(boxes: &[[f32; 4]], texts: &[String]) -> String {
    let rows = rows_of(boxes);
    let x_max = boxes.iter().map(|b| b[2]).fold(f32::NEG_INFINITY, f32::max);
    let x_min = boxes.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min);
    let span_x = x_max - x_min;
    let widest = rows.iter().map(|r| r.len()).max().unwrap();
    if rows.len() > 2 && rows[0].len() == 1 && widest > 1 {
        let t = rows[0][0];
        if boxes[t][2] - boxes[t][0] > 0.6 * span_x {
            let keep: Vec<usize> = (0..texts.len()).filter(|&k| k != t).collect();
            let kb: Vec<[f32; 4]> = keep.iter().map(|&k| boxes[k]).collect();
            let kt: Vec<String> = keep.iter().map(|&k| texts[k].clone()).collect();
            return format!("{}\n\n{}", texts[t], table_html(&kb, &kt));
        }
    }
    let width = widest;
    let full: Vec<&Vec<usize>> = rows.iter().filter(|r| r.len() == width).collect();
    let rows = merge_wrapped(rows.clone(), boxes);
    let bands: Vec<[f32; 2]> = (0..width)
        .map(|c| {
            [
                median_f32(&full.iter().map(|r| boxes[r[c]][0]).collect::<Vec<_>>()),
                median_f32(&full.iter().map(|r| boxes[r[c]][2]).collect::<Vec<_>>()),
            ]
        })
        .collect();
    let mut edges = vec![f32::NEG_INFINITY];
    for c in 0..width - 1 {
        edges.push((bands[c][1] + bands[c + 1][0]) / 2.0);
    }
    edges.push(f32::INFINITY);
    // np.searchsorted(edges, x, side="right") - 1
    let band_of = |x: f32| edges.iter().filter(|&&e| e <= x).count() as isize - 1;
    let mut out = vec!["<table>".to_string()];
    for (ri, r) in rows.iter().enumerate() {
        let tag = if ri == 0 && rows.len() > 1 { "th" } else { "td" };
        let mut cells: std::collections::BTreeMap<usize, (usize, String)> = Default::default();
        let mut sorted = r.clone();
        sorted.sort_by(|&a, &b| boxes[a][1].total_cmp(&boxes[b][1]).then(boxes[a][0].total_cmp(&boxes[b][0])));
        for k in sorted {
            let (x0, x1) = (boxes[k][0], boxes[k][2]);
            let first = band_of(x0);
            let last = band_of(x1);
            let lo = first.max(0) as usize;
            let hi = last.min(width as isize - 1);
            let mut span: Vec<usize> = vec![];
            if hi >= lo as isize {
                for c in lo..=hi as usize {
                    if x1.min(bands[c][1]) - x0.max(bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0]) {
                        span.push(c);
                    }
                }
            }
            if span.is_empty() {
                span.push((first.max(0) as usize).min(width - 1));
            }
            let c0 = span[0];
            let entry = match cells.get(&c0) {
                None => (span.len(), texts[k].clone()),
                Some((n, t)) => ((*n).max(span.len()), format!("{t} {}", texts[k])),
            };
            cells.insert(c0, entry);
        }
        out.push(row_html(tag, &cells, width));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// Whether the cell head joined side-by-side lines in at most 10% of its cells.
pub fn heads_plausible(boxes: &[[f32; 4]], tab_cell: &Mat) -> bool {
    let cells = cluster(tab_cell);
    let mut bad = 0;
    for c in &cells {
        if c.len() < 2 {
            continue;
        }
        let mut min_overlap = f32::INFINITY;
        for &i in c {
            for &j in c {
                let ov = boxes[i][2].min(boxes[j][2]) - boxes[i][0].max(boxes[j][0]);
                min_overlap = min_overlap.min(ov);
            }
        }
        if min_overlap < 0.0 {
            bad += 1;
        }
    }
    bad as f64 <= 0.1 * cells.len() as f64
}

/// An HTML table from the order model's table heads (same row / column / cell
/// logits for the lines of one table block).
pub fn table_html_heads(boxes: &[[f32; 4]], texts: &[String], tab: &[Mat; 3]) -> String {
    let cells = cluster(&tab[2]);
    let cb: Vec<[f32; 4]> = cells
        .iter()
        .map(|c| {
            [
                col(boxes, c, 0).into_iter().fold(f32::INFINITY, f32::min),
                col(boxes, c, 1).into_iter().fold(f32::INFINITY, f32::min),
                col(boxes, c, 2).into_iter().fold(f32::NEG_INFINITY, f32::max),
                col(boxes, c, 3).into_iter().fold(f32::NEG_INFINITY, f32::max),
            ]
        })
        .collect();
    let ctext: Vec<String> = cells
        .iter()
        .map(|c| {
            let mut s = c.clone();
            s.sort_by(|&a, &b| boxes[a][1].total_cmp(&boxes[b][1]).then(boxes[a][0].total_cmp(&boxes[b][0])));
            s.iter().map(|&k| texts[k].as_str()).collect::<Vec<_>>().join(" ")
        })
        .collect();
    let m = cells.len();
    let cell_logits = |k: usize| -> Mat {
        let mut out = Mat::zeros(m, m);
        for a in 0..m {
            for b in a + 1..m {
                let sub = tab[k].select(&cells[a], &cells[b]);
                let v = crate::py::pairwise_sum_f32(&sub.data) / sub.data.len() as f32;
                out.data[a * m + b] = v;
                out.data[b * m + a] = v;
            }
        }
        out
    };
    let min_of = |g: &[usize], k: usize| g.iter().map(|&c| cb[c][k]).fold(f32::INFINITY, f32::min);
    let mut rows = cluster(&cell_logits(0));
    rows.sort_by(|a, b| min_of(a, 1).total_cmp(&min_of(b, 1)));
    let mut cols = cluster(&cell_logits(1));
    cols.sort_by(|a, b| min_of(a, 0).total_cmp(&min_of(b, 0)));
    let mut row_of = vec![0; m];
    let mut col_of = vec![0; m];
    for (r, g) in rows.iter().enumerate() {
        for &c in g {
            row_of[c] = r;
        }
    }
    for (k, g) in cols.iter().enumerate() {
        for &c in g {
            col_of[c] = k;
        }
    }
    let centers: Vec<f32> = cols.iter().map(|g| median_f32(&g.iter().map(|&c| (cb[c][0] + cb[c][2]) / 2.0).collect::<Vec<_>>())).collect();
    let h = median_height(boxes);
    let mut grid: std::collections::BTreeMap<(usize, usize), (usize, String)> = Default::default();
    for c in 0..m {
        let c0 = col_of[c];
        let mut span = 1;
        while c0 + span < cols.len() && cb[c][0] - h <= centers[c0 + span] && centers[c0 + span] <= cb[c][2] + h {
            span += 1;
        }
        let key = (row_of[c], c0);
        let entry = match grid.get(&key) {
            Some((n, t)) => ((*n).max(span), format!("{t} {}", ctext[c])),
            None => (span, ctext[c].clone()),
        };
        grid.insert(key, entry);
    }
    let mut out = vec!["<table>".to_string()];
    for r in 0..rows.len() {
        let tag = if r == 0 && rows.len() > 1 { "th" } else { "td" };
        let cells: std::collections::BTreeMap<usize, (usize, String)> =
            grid.iter().filter(|((rr, _), _)| *rr == r).map(|((_, c), v)| (*c, v.clone())).collect();
        out.push(row_html(tag, &cells, cols.len()));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// One layout block of a page, in reading order.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Block {
    pub label: String,
    pub bbox: [f64; 4],
    pub text: String,
}

/// The layout blocks behind ``page_markdown``, in reading order.
pub fn page_blocks(boxes: &[[f32; 4]], texts: &[String], order: &[usize], cls: &Mat, same: &Mat) -> Vec<Block> {
    split_blocks(order, same)
        .into_iter()
        .map(|blk| {
            let fold = |k: usize, init: f32, f: fn(f32, f32) -> f32| blk.iter().map(|&i| boxes[i][k]).fold(init, f);
            Block {
                label: BLOCK_CLASSES[cls_of(cls, &blk)].to_string(),
                bbox: [
                    round1(fold(0, f32::INFINITY, f32::min) as f64),
                    round1(fold(1, f32::INFINITY, f32::min) as f64),
                    round1(fold(2, f32::NEG_INFINITY, f32::max) as f64),
                    round1(fold(3, f32::NEG_INFINITY, f32::max) as f64),
                ],
                text: blk.iter().map(|&i| texts[i].as_str()).collect::<Vec<_>>().join("\n"),
            }
        })
        .collect()
}

/// Table heads of the lines of one block: [3, n, n] logits.
fn tab_select(tab: &[Mat; 3], blk: &[usize]) -> [Mat; 3] {
    [tab[0].select(blk, blk), tab[1].select(blk, blk), tab[2].select(blk, blk)]
}

/// Cells (box, text) of each line that holds several (table rows read as one line).
pub type Cells = std::collections::HashMap<usize, Vec<([f32; 4], String)>>;

/// Markdown for one page.
#[allow(clippy::too_many_arguments)]
pub fn page_markdown(
    boxes: &[[f32; 4]],
    texts: &[String],
    order: &[usize],
    cls: &Mat,
    same: &Mat,
    keep_furniture: bool,
    cells: Option<&Cells>,
    tab: Option<&[Mat; 3]>,
) -> String {
    let (header, footer) = (class_id("Page-Header"), class_id("Page-Footer"));
    let (section, list, table) = (class_id("Section-Header"), class_id("List-Group"), class_id("Table"));
    let mut parts: Vec<String> = vec![];
    for blk in split_blocks(order, same) {
        let c = cls_of(cls, &blk);
        let tb: Vec<&str> = blk.iter().map(|&i| texts[i].as_str()).collect();
        if (c == header || c == footer) && !keep_furniture {
            continue;
        }
        if tb.iter().all(|t| whole_math(strip(t))) {
            parts.push(
                tb.iter()
                    .map(|t| {
                        let t = strip(t);
                        format!("$${}$$", &t[1..t.len() - 1])
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            continue;
        }
        let bb: Vec<[f32; 4]> = blk.iter().map(|&i| boxes[i]).collect();
        if c == section {
            parts.push(format!("## {}", join_lines(&tb)));
        } else if c == table && blk.len() > 1 && tab.is_some_and(|t| heads_plausible(&bb, &tab_select(t, &blk)[2])) {
            let owned: Vec<String> = tb.iter().map(|s| s.to_string()).collect();
            parts.push(table_html_heads(&bb, &owned, &tab_select(tab.unwrap(), &blk)));
        } else if c == table && blk.len() > 1 {
            let (mut cbx, mut ctx) = (vec![], vec![]);
            for &i in &blk {
                match cells.and_then(|m| m.get(&i)).filter(|v| !v.is_empty()) {
                    Some(v) => {
                        for (b, t) in v {
                            cbx.push(*b);
                            ctx.push(t.clone());
                        }
                    }
                    None => {
                        cbx.push(boxes[i]);
                        ctx.push(texts[i].clone());
                    }
                }
            }
            parts.push(table_html(&cbx, &ctx));
        } else if c == list {
            let mut items: Vec<Vec<&str>> = vec![];
            for t in &tb {
                if items.is_empty() || is_item(t) {
                    items.push(vec![t]);
                } else {
                    items.last_mut().unwrap().push(t);
                }
            }
            parts.push(items.iter().map(|it| join_lines(it)).collect::<Vec<_>>().join("\n"));
        } else {
            parts.push(join_lines(&tb));
        }
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins() {
        assert_eq!(join_lines(&["tumori-", "genesis is"]), "tumorigenesis is");
        assert_eq!(join_lines(&["中文", "字符"]), "中文字符");
        assert_eq!(join_lines(&["a", " ", "b"]), "a b");
    }

    #[test]
    fn items() {
        for t in ["• x", "1. x", "(12) x", "a) x", "-", "(b."] {
            assert!(is_item(t), "{t}");
        }
        for t in ["1234. x", "ab) x", "-x", "1.5 kg"] {
            assert!(!is_item(t), "{t}");
        }
    }

    #[test]
    fn simple_table() {
        let boxes = [[0., 0., 10., 10.], [50., 0., 60., 10.], [0., 20., 10., 30.], [50., 20., 60., 30.]];
        let texts: Vec<String> = ["A", "B", "1", "2<3"].iter().map(|s| s.to_string()).collect();
        assert_eq!(table_html(&boxes, &texts), "<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>1</td><td>2&lt;3</td></tr>\n</table>");
    }
}
