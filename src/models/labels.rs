use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Labels {
    values: BTreeMap<usize, String>,
}

impl Labels {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LabelError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| LabelError::Read {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&contents)
    }

    pub fn parse(contents: &str) -> Result<Self, LabelError> {
        let mut values = BTreeMap::new();
        let mut next_id = 0;
        for raw_line in contents.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, label) = match line.split_once(' ') {
                Some((first, rest))
                    if first.chars().all(|character| character.is_ascii_digit()) =>
                {
                    let id = first
                        .parse::<usize>()
                        .map_err(|_| LabelError::InvalidId(first.into()))?;
                    let label = rest.trim();
                    if label.is_empty() {
                        (next_id, line)
                    } else {
                        (id, label)
                    }
                }
                _ => (next_id, line),
            };
            values.insert(id, label.to_owned());
            next_id = next_id.max(id + 1);
        }
        if values.is_empty() {
            return Err(LabelError::Empty);
        }
        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, class_id: usize) -> String {
        self.values
            .get(&class_id)
            .cloned()
            .unwrap_or_else(|| class_id.to_string())
    }
}

#[derive(Debug, Error)]
pub enum LabelError {
    #[error("failed to read labels from {path}: {source}")]
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("label file is empty")]
    Empty,
    #[error("invalid label id {0:?}")]
    InvalidId(String),
}
