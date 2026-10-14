//! Python and NumPy semantics the pipeline depends on: str predicates and
//! whitespace, round(), np.median, np.percentile, and NumPy's pairwise float
//! summation.

use crate::unitab;

fn in_table(t: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    t.binary_search_by(|&(a, b)| {
        if b < cp {
            std::cmp::Ordering::Less
        } else if a > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

/// str.isalpha of one character.
pub fn is_alpha(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_table(&unitab::ALPHA, c)
}
/// str.isdigit of one character.
pub fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(&unitab::DIGIT, c)
}
/// str.isdecimal of one character (what re's \\d matches).
pub fn is_decimal(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(&unitab::DECIMAL, c)
}
/// str.isupper of one character.
pub fn is_upper(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    in_table(&unitab::UPPER, c)
}
/// str.islower of one character.
pub fn is_lower(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_lowercase();
    }
    in_table(&unitab::LOWER, c)
}
/// str.isspace of one character.
pub fn is_space(c: char) -> bool {
    in_table(&unitab::SPACE, c)
}
/// U+FFFD, private use, surrogate, unassigned or control (but tab).
pub fn is_bad_char(c: char) -> bool {
    in_table(&unitab::BAD, c)
}
/// unicodedata.bidirectional(c) in ("R", "AL")
pub fn bidi_rtl(c: char) -> bool {
    in_table(&unitab::BIDI_RTL, c)
}
/// unicodedata.bidirectional(c) == "L"
pub fn bidi_ltr(c: char) -> bool {
    in_table(&unitab::BIDI_LTR, c)
}

/// str.isalpha of a string: non-empty and every character alphabetic.
pub fn str_is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_alpha)
}
/// str.isdigit of a string.
pub fn str_is_digit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit)
}

/// str.strip() (Python whitespace).
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// " ".join(s.split())
pub fn squash_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split(is_space).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

/// Python round(x) for a float: nearest integer, ties to even.
pub fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python round(x, 1): the correctly rounded one-decimal value.
pub fn round1(x: f64) -> f64 {
    round_n(x, 1)
}

/// Python round(x, n) for small n (via the shortest correctly rounded decimal).
pub fn round_n(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.n$}").parse().unwrap()
}

/// np.median of float32 values (mean of the two middle values in float32).
pub fn median_f32(v: &[f32]) -> f32 {
    assert!(!v.is_empty(), "median of nothing");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// np.median of float64 values.
pub fn median_f64(v: &[f64]) -> f64 {
    assert!(!v.is_empty(), "median of nothing");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// np.percentile(v, q) with the default linear method.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    assert!(!v.is_empty(), "percentile of nothing");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let idx = q / 100.0 * (s.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let t = idx - lo as f64;
    let (a, b) = (s[lo], s[hi]);
    let d = b - a;
    // NumPy's _lerp: from the nearer end, for symmetry.
    if t >= 0.5 { b - d * (1.0 - t) } else { a + d * t }
}

/// np.add.reduce of a contiguous float32 vector (NumPy's pairwise summation:
/// blocks of 128 with 8 accumulators).
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = -0.0f32;
        for &v in a {
            r += v;
        }
        return r;
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
}

/// np.add.reduce of a contiguous float64 vector (pairwise, as pairwise_sum_f32).
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = -0.0f64;
        for &v in a {
            r += v;
        }
        return r;
    }
    if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
}

/// Indices that sort ``v`` ascending; ties keep their order.
pub fn argsort_by<T: Copy>(v: &[T], mut key: impl FnMut(T) -> f64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| key(v[i]).total_cmp(&key(v[j])));
    idx
}

/// Index of the first maximum (np.argmax).
pub fn argmax<T: PartialOrd + Copy>(v: &[T]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_python() {
        assert_eq!(round1(0.25), 0.2);
        assert_eq!(round1(0.35), 0.3); // 0.35 is 0.34999.. in binary
        assert_eq!(round1(2.675), 2.7);
        assert_eq!(round1(-1.05), -1.1);
        assert_eq!(round_int(2.5), 2);
        assert_eq!(round_int(3.5), 4);
    }

    #[test]
    fn string_predicates() {
        assert!(is_alpha('é') && !is_alpha('1') && is_alpha('中'));
        assert!(is_digit('²') && is_digit('٣') && !is_digit('½'));
        assert!(is_space('\u{a0}') && is_space('\u{1c}') && !is_space('\u{200b}'));
        assert_eq!(squash_spaces("  a \t b\u{3000}c "), "a b c");
        assert_eq!(strip("\u{85}x\n"), "x");
    }

    #[test]
    fn numpy_stats() {
        assert_eq!(median_f32(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 90.0), 4.6);
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        assert!((pairwise_sum_f32(&v) - 49950.0).abs() < 0.1);
    }
}
