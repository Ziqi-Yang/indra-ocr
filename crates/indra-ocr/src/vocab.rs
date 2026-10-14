//! The recognizer's token vocabulary (vocab.py): class 0 is the CTC blank,
//! class i + 1 is ``chars[i]``, one character or one LaTeX command.

use std::collections::HashMap;

use crate::py::{is_alpha, str_is_alpha};

#[derive(Clone, Debug)]
pub struct Vocab {
    pub chars: Vec<String>,
    pub index: HashMap<String, u32>,
    pub has_commands: bool,
}

#[derive(serde::Deserialize)]
struct VocabJson {
    #[serde(default)]
    blank: u32,
    chars: Vec<String>,
}

impl Vocab {
    pub fn new(chars: Vec<String>) -> Self {
        let index = chars.iter().enumerate().map(|(i, c)| (c.clone(), i as u32 + 1)).collect();
        let has_commands = chars.iter().any(|c| c.chars().count() > 1);
        Vocab { chars, index, has_commands }
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let d: VocabJson = serde_json::from_str(s).map_err(|e| format!("bad vocabulary: {e}"))?;
        if d.blank != 0 {
            return Err("only blank=0 vocabularies are supported".into());
        }
        Ok(Self::new(d.chars))
    }

    /// Whether a single character is a token of this vocabulary.
    pub fn has_char(&self, c: char) -> bool {
        let mut b = [0u8; 4];
        self.index.contains_key(c.encode_utf8(&mut b) as &str)
    }

    /// Token of class ``id`` (1-based), if it exists.
    pub fn token(&self, id: u32) -> Option<&str> {
        if id == 0 {
            return None;
        }
        self.chars.get(id as usize - 1).map(|s| s.as_str())
    }

    pub fn decode(&self, ids: impl IntoIterator<Item = u32>) -> String {
        let toks: Vec<&str> = ids.into_iter().filter_map(|i| self.token(i)).collect();
        if !self.has_commands {
            return toks.concat();
        }
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            // "\alpha" + "x" must not read back as "\alphax"
            let nxt = toks.get(k + 1).copied().unwrap_or("");
            let mut tc = t.chars();
            if t.chars().count() > 2
                && tc.next() == Some('\\')
                && str_is_alpha(tc.as_str())
                && nxt.chars().next().is_some_and(is_alpha)
            {
                out.push(' ');
            }
        }
        out
    }
}
