use std::collections::{BTreeMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::models::ModelKind;
pub use crate::preprocess::PreprocessMode;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    pub models: Vec<ModelConfig>,
}

impl Config {
    pub fn parse(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source)?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&source)
    }

    pub fn default_model(&self) -> &ModelConfig {
        self.models
            .iter()
            .find(|model| model.default)
            .expect("validated config has one default model")
    }

    pub fn model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|model| model.name == name)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        self.server.validate()?;
        self.runtime.validate()?;

        if self.models.is_empty() {
            return Err(ConfigError::validation("at least one model is required"));
        }

        let mut names = HashSet::new();
        let mut default_count = 0;
        for model in &self.models {
            model.validate()?;
            if !names.insert(model.name.as_str()) {
                return Err(ConfigError::validation(format!(
                    "model name {:?} is configured more than once",
                    model.name
                )));
            }
            if model.default {
                default_count += 1;
            }
        }
        if default_count != 1 {
            return Err(ConfigError::validation(format!(
                "exactly one model must be default, found {default_count}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: String,
    pub max_body_bytes: usize,
    pub request_timeout_ms: u64,
    pub queue_depth: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8700".into(),
            max_body_bytes: 8_000_000,
            request_timeout_ms: 2_000,
            queue_depth: 8,
        }
    }
}

impl ServerConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.bind.parse::<SocketAddr>().map_err(|error| {
            ConfigError::validation(format!("server.bind {:?} is invalid: {error}", self.bind))
        })?;
        if self.max_body_bytes == 0 {
            return Err(ConfigError::validation(
                "server.max_body_bytes must be positive",
            ));
        }
        if self.request_timeout_ms == 0 {
            return Err(ConfigError::validation(
                "server.request_timeout_ms must be positive",
            ));
        }
        if self.queue_depth == 0 {
            return Err(ConfigError::validation(
                "server.queue_depth must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub tflite_library: String,
    pub edgetpu_library: String,
    pub device: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            tflite_library: "libtensorflowlite_c.so".into(),
            edgetpu_library: "libedgetpu.so.1".into(),
            device: "usb".into(),
        }
    }
}

impl RuntimeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.tflite_library.trim().is_empty() {
            return Err(ConfigError::validation(
                "runtime.tflite_library must not be empty",
            ));
        }
        if self.edgetpu_library.trim().is_empty() {
            return Err(ConfigError::validation(
                "runtime.edgetpu_library must not be empty",
            ));
        }
        if !valid_device(&self.device) {
            return Err(ConfigError::validation(format!(
                "runtime.device {:?} must be usb, usb:0, pci, or pci:0",
                self.device
            )));
        }
        Ok(())
    }
}

fn valid_device(device: &str) -> bool {
    matches!(device, "usb" | "usb:0" | "pci" | "pci:0")
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default)]
    pub default: bool,
    pub path: PathBuf,
    pub labels: PathBuf,
    #[serde(default, rename = "type")]
    pub model_type: Option<ModelKind>,
    #[serde(default)]
    pub input_size: Option<u32>,
    #[serde(default)]
    pub preprocess: PreprocessMode,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_nms_iou")]
    pub nms_iou: f32,
    #[serde(default = "default_max_detections")]
    pub max_detections: usize,
    #[serde(default)]
    pub class_thresholds: BTreeMap<String, f32>,
}

fn default_threshold() -> f32 {
    0.12
}

fn default_nms_iou() -> f32 {
    0.5
}

fn default_max_detections() -> usize {
    50
}

impl ModelConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.name.trim().is_empty() {
            return Err(ConfigError::validation("model.name must not be empty"));
        }
        if self.path.as_os_str().is_empty() {
            return Err(ConfigError::validation(format!(
                "model {:?} has an empty path",
                self.name
            )));
        }
        if self.labels.as_os_str().is_empty() {
            return Err(ConfigError::validation(format!(
                "model {:?} has an empty labels path",
                self.name
            )));
        }
        if self.input_size == Some(0) {
            return Err(ConfigError::validation(format!(
                "model {:?} input_size must be positive",
                self.name
            )));
        }
        if !self.threshold.is_finite() || !(0.0..1.0).contains(&self.threshold) {
            return Err(ConfigError::validation(format!(
                "model {:?} threshold must be between 0 and 1 (exclusive)",
                self.name
            )));
        }
        if !self.nms_iou.is_finite() || !(0.0..=1.0).contains(&self.nms_iou) {
            return Err(ConfigError::validation(format!(
                "model {:?} nms_iou must be between 0 and 1",
                self.name
            )));
        }
        if self.max_detections == 0 {
            return Err(ConfigError::validation(format!(
                "model {:?} max_detections must be positive",
                self.name
            )));
        }
        for (class, threshold) in &self.class_thresholds {
            if class.trim().is_empty() {
                return Err(ConfigError::validation(format!(
                    "model {:?} has an empty class threshold name",
                    self.name
                )));
            }
            if !threshold.is_finite() || !(0.0 < *threshold && *threshold <= 1.0) {
                return Err(ConfigError::validation(format!(
                    "model {:?} class threshold {:?} must be greater than 0 and at most 1",
                    self.name, class
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid config: {0}")]
    Validation(String),
}

impl ConfigError {
    fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_CONFIG: &str = r#"
[runtime]
device = "usb"

[[models]]
name = "detector"
default = true
path = "model.tflite"
labels = "labels.txt"
threshold = 0.5
"#;

    #[test]
    fn config_rejects_invalid_values() {
        Config::parse(include_str!("../tpue.toml")).expect("shipped config should parse");

        let cases = [
            (
                "two default models",
                format!(
                    "{VALID_CONFIG}\n[[models]]\nname = \"other\"\ndefault = true\npath = \"other.tflite\"\nlabels = \"labels.txt\"\n"
                ),
            ),
            (
                "threshold outside (0, 1)",
                VALID_CONFIG.replace("threshold = 0.5", "threshold = 1.0"),
            ),
            (
                "unsupported device",
                VALID_CONFIG.replace("device = \"usb\"", "device = \"cpu\""),
            ),
        ];

        for (case, source) in cases {
            let error = Config::parse(&source).expect_err(case);
            assert!(
                matches!(error, ConfigError::Validation(_)),
                "{case}: {error}"
            );
        }
    }
}
