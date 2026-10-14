//! JSON written as Python's ``json.dumps(obj, ensure_ascii=False)`` writes it:
//! ", " and ": " separators and Python's float repr ("1e-05", "12.0").

use std::io::{self, Write};

use serde::Serialize;

/// Python's repr of a float.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits and the decimal exponent
    let e = format!("{:e}", x.abs());
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp >= n - 1 {
            format!("{digits}{}.0", "0".repeat((exp - (n - 1)) as usize))
        } else if exp >= 0 {
            format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn write_f64<W: ?Sized + Write>(&mut self, w: &mut W, v: f64) -> io::Result<()> {
        w.write_all(float_repr(v).as_bytes())
    }
    fn write_f32<W: ?Sized + Write>(&mut self, w: &mut W, v: f32) -> io::Result<()> {
        w.write_all(float_repr(v as f64).as_bytes())
    }
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// ``value`` as Python's json.dumps(value, ensure_ascii=False).
pub fn dumps<T: Serialize + ?Sized>(value: &T) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
    value.serialize(&mut ser).expect("serializable");
    String::from_utf8(buf).expect("JSON is UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs() {
        for (x, s) in [
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (12.0, "12.0"),
            (123.456, "123.456"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (-0.5, "-0.5"),
            (0.1, "0.1"),
            (2.5e-7, "2.5e-07"),
        ] {
            assert_eq!(float_repr(x), s);
        }
        let v = serde_json::json!({"a": [1, 2.0], "b": "é"});
        assert_eq!(dumps(&v), r#"{"a": [1, 2.0], "b": "é"}"#);
    }
}
