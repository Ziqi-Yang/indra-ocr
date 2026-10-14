//! A minimal ONNX reader: the graph's nodes, their attributes, the
//! initializers and Constant nodes. Enough to read the weights of a known
//! architecture and to check that the graph is the one the engine implements.

use std::collections::HashMap;
use std::path::Path;

use crate::{EngineError, Result};

/// A protobuf wire-format reader over a byte slice.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

enum Value<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.pos).ok_or_else(|| EngineError::Model("truncated protobuf".into()))?;
            self.pos += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(EngineError::Model("bad protobuf varint".into()))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| EngineError::Model("truncated protobuf".into()))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// The next field: (number, value), or None at the end.
    fn field(&mut self) -> Result<Option<(u32, Value<'a>)>> {
        if self.pos >= self.b.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let (num, wire) = ((key >> 3) as u32, key & 7);
        let v = match wire {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                Value::Bytes(self.take(n)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            w => return Err(EngineError::Model(format!("unsupported protobuf wire type {w}"))),
        };
        Ok(Some((num, v)))
    }
}

fn string(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Repeated int64 (packed or not).
fn push_ints(out: &mut Vec<i64>, v: Value) -> Result<()> {
    match v {
        Value::Varint(x) => out.push(x as i64),
        Value::Bytes(b) => {
            let mut r = Reader::new(b);
            while r.pos < b.len() {
                out.push(r.varint()? as i64);
            }
        }
        _ => return Err(EngineError::Model("bad repeated int64".into())),
    }
    Ok(())
}

/// Repeated float (packed or not).
fn push_floats(out: &mut Vec<f32>, v: Value) -> Result<()> {
    match v {
        Value::Fixed32(x) => out.push(f32::from_bits(x)),
        Value::Bytes(b) => out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        _ => return Err(EngineError::Model("bad repeated float".into())),
    }
    Ok(())
}

/// A tensor's shape and values (as f32 or i64, by its data type).
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub f32: Vec<f32>,
    pub i64: Vec<i64>,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

const FLOAT: i64 = 1;
const INT32: i64 = 6;
const INT64: i64 = 7;
const BOOL: i64 = 9;
const DOUBLE: i64 = 11;

fn tensor(b: &[u8]) -> Result<Tensor> {
    let mut r = Reader::new(b);
    let (mut t, mut dtype, mut raw): (Tensor, i64, &[u8]) = (Tensor::default(), 0, &[]);
    let mut dims: Vec<i64> = vec![];
    let mut i32s: Vec<i64> = vec![];
    let mut doubles: Vec<f64> = vec![];
    while let Some((num, v)) = r.field()? {
        match (num, v) {
            (1, v) => push_ints(&mut dims, v)?,
            (2, Value::Varint(x)) => dtype = x as i64,
            (4, v) => push_floats(&mut t.f32, v)?,
            (5, v) => push_ints(&mut i32s, v)?,
            (7, v) => push_ints(&mut t.i64, v)?,
            (8, Value::Bytes(s)) => t.name = string(s),
            (9, Value::Bytes(s)) => raw = s,
            (10, Value::Bytes(s)) => doubles.extend(s.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))),
            (14, Value::Varint(1)) => return Err(EngineError::Model(format!("tensor {}: external data is not supported", t.name))),
            _ => {}
        }
    }
    t.dims = dims.iter().map(|&d| d.max(0) as usize).collect();
    match dtype {
        FLOAT if !raw.is_empty() => t.f32 = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        DOUBLE => {
            let d: Vec<f64> = if raw.is_empty() { doubles } else { raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() };
            t.f32 = d.into_iter().map(|v| v as f32).collect();
        }
        INT64 if !raw.is_empty() => t.i64 = raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        INT32 | BOOL => t.i64 = if raw.is_empty() { i32s } else if dtype == BOOL { raw.iter().map(|&v| v as i64).collect() } else { raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect() },
        _ => {}
    }
    Ok(t)
}

/// A node attribute.
#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Str(String),
    Tensor(Tensor),
}

fn attribute(b: &[u8]) -> Result<(String, Attr)> {
    let mut r = Reader::new(b);
    let mut name = String::new();
    let (mut i, mut f, mut s, mut t) = (None, None, None, None);
    let (mut ints, mut floats) = (vec![], vec![]);
    let mut ty = 0;
    while let Some((num, v)) = r.field()? {
        match (num, v) {
            (1, Value::Bytes(x)) => name = string(x),
            (2, Value::Fixed32(x)) => f = Some(f32::from_bits(x)),
            (3, Value::Varint(x)) => i = Some(x as i64),
            (4, Value::Bytes(x)) => s = Some(string(x)),
            (5, Value::Bytes(x)) => t = Some(tensor(x)?),
            (7, v) => push_floats(&mut floats, v)?,
            (8, v) => push_ints(&mut ints, v)?,
            (20, Value::Varint(x)) => ty = x,
            _ => {}
        }
    }
    let a = match ty {
        1 => Attr::Float(f.unwrap_or(0.0)),
        2 => Attr::Int(i.unwrap_or(0)),
        3 => Attr::Str(s.unwrap_or_default()),
        4 => Attr::Tensor(t.unwrap_or_default()),
        6 => Attr::Floats(floats),
        7 => Attr::Ints(ints),
        _ => match (i, f, s, t) {
            (Some(i), ..) => Attr::Int(i),
            (_, Some(f), ..) => Attr::Float(f),
            (_, _, Some(s), _) => Attr::Str(s),
            (_, _, _, Some(t)) => Attr::Tensor(t),
            _ if !ints.is_empty() => Attr::Ints(ints),
            _ => Attr::Floats(floats),
        },
    };
    Ok((name, a))
}

/// One graph node.
#[derive(Clone, Debug, Default)]
pub struct Node {
    pub op: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

impl Node {
    pub fn int(&self, k: &str) -> Option<i64> {
        match self.attrs.get(k) {
            Some(Attr::Int(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn float(&self, k: &str) -> Option<f32> {
        match self.attrs.get(k) {
            Some(Attr::Float(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn ints(&self, k: &str) -> Option<&[i64]> {
        match self.attrs.get(k) {
            Some(Attr::Ints(v)) => Some(v),
            _ => None,
        }
    }
}

fn node(b: &[u8]) -> Result<Node> {
    let mut r = Reader::new(b);
    let mut n = Node::default();
    while let Some((num, v)) = r.field()? {
        match (num, v) {
            (1, Value::Bytes(x)) => n.inputs.push(string(x)),
            (2, Value::Bytes(x)) => n.outputs.push(string(x)),
            (3, Value::Bytes(x)) => n.name = string(x),
            (4, Value::Bytes(x)) => n.op = string(x),
            (5, Value::Bytes(x)) => {
                let (k, a) = attribute(x)?;
                n.attrs.insert(k, a);
            }
            _ => {}
        }
    }
    Ok(n)
}

fn value_info_name(b: &[u8]) -> Result<String> {
    let mut r = Reader::new(b);
    while let Some((num, v)) = r.field()? {
        if let (1, Value::Bytes(x)) = (num, v) {
            return Ok(string(x));
        }
    }
    Ok(String::new())
}

/// An ONNX graph: nodes in order, initializers and Constant outputs by name.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors: HashMap<String, Tensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Graph {
    pub fn load(path: &Path) -> Result<Graph> {
        let b = std::fs::read(path).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))?;
        Graph::parse(&b).map_err(|e| EngineError::Model(format!("{}: {e}", path.display())))
    }

    pub fn parse(b: &[u8]) -> Result<Graph> {
        let mut r = Reader::new(b);
        let mut g = Graph::default();
        while let Some((num, v)) = r.field()? {
            if let (7, Value::Bytes(gb)) = (num, v) {
                let mut gr = Reader::new(gb);
                let mut inits: Vec<String> = vec![];
                while let Some((gn, gv)) = gr.field()? {
                    match (gn, gv) {
                        (1, Value::Bytes(x)) => g.nodes.push(node(x)?),
                        (5, Value::Bytes(x)) => {
                            let t = tensor(x)?;
                            inits.push(t.name.clone());
                            g.tensors.insert(t.name.clone(), t);
                        }
                        (11, Value::Bytes(x)) => g.inputs.push(value_info_name(x)?),
                        (12, Value::Bytes(x)) => g.outputs.push(value_info_name(x)?),
                        _ => {}
                    }
                }
                // Initializers may be listed as inputs (old exports): keep the real ones.
                g.inputs.retain(|i| !inits.contains(i));
            }
        }
        // Constant nodes become named tensors.
        for n in &g.nodes {
            if n.op == "Constant"
                && let Some(Attr::Tensor(t)) = n.attrs.get("value")
            {
                let mut t = t.clone();
                t.name = n.outputs[0].clone();
                g.tensors.insert(t.name.clone(), t);
            }
        }
        g.nodes.retain(|n| n.op != "Constant");
        Ok(g)
    }

    /// The nodes of type ``op``, in order.
    pub fn ops<'a>(&'a self, op: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.nodes.iter().filter(move |n| n.op == op)
    }

    /// The tensor called ``name`` (an initializer or a Constant output).
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| EngineError::Model(format!("no tensor {name:?}")))
    }

    /// Input ``i`` of ``node`` as a tensor.
    pub fn arg(&self, node: &Node, i: usize) -> Result<&Tensor> {
        let name = node.inputs.get(i).ok_or_else(|| EngineError::Model(format!("{} has no input {i}", node.op)))?;
        self.tensor(name)
    }
}
