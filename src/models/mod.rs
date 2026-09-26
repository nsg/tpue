mod labels;
mod ssd;
mod yolo_generic;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub use labels::{LabelError, Labels};
pub use ssd::Ssd;
use thiserror::Error;
pub use yolo_generic::YoloGeneric;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2d {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Box2d {
    pub fn width(self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn iou(self, other: Self) -> f32 {
        let intersection = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0)
            * (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let union = self.width() * self.height() + other.width() * other.height() - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: Box2d,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantization {
    pub scale: f32,
    pub zero_point: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    I8(Vec<i8>),
    F32(Vec<f32>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
    pub quantization: Option<Quantization>,
}

impl RawTensor {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn validate(&self) -> Result<(), PostprocessError> {
        let actual = match &self.data {
            TensorData::I8(values) => values.len(),
            TensorData::F32(values) => values.len(),
        };
        if actual != self.element_count() {
            return Err(PostprocessError::InvalidTensor(format!(
                "tensor {:?} shape {:?} requires {} values, found {actual}",
                self.name,
                self.shape,
                self.element_count()
            )));
        }
        if matches!(self.data, TensorData::I8(_))
            && self
                .quantization
                .is_none_or(|quantization| quantization.scale <= 0.0)
        {
            return Err(PostprocessError::InvalidTensor(format!(
                "tensor {:?} has invalid quantization parameters",
                self.name
            )));
        }
        Ok(())
    }

    fn value(&self, index: usize) -> f32 {
        match &self.data {
            TensorData::I8(values) => {
                let quantization = self.quantization.expect("validated quantization");
                (i32::from(values[index]) - quantization.zero_point) as f32 * quantization.scale
            }
            TensorData::F32(values) => values[index],
        }
    }
}

#[derive(Clone, Debug)]
pub struct PostprocessOptions {
    pub threshold: f32,
    pub nms_iou: f32,
    pub max_detections: usize,
    pub class_thresholds: HashMap<String, f32>,
    pub classes: Option<HashSet<String>>,
}

impl Default for PostprocessOptions {
    fn default() -> Self {
        Self {
            threshold: 0.4,
            nms_iou: 0.5,
            max_detections: 100,
            class_thresholds: HashMap::new(),
            classes: None,
        }
    }
}

impl PostprocessOptions {
    fn validate(&self) -> Result<(), PostprocessError> {
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err(PostprocessError::InvalidOptions(
                "threshold must be finite and between 0 and 1".into(),
            ));
        }
        if !self.nms_iou.is_finite() || !(0.0..=1.0).contains(&self.nms_iou) {
            return Err(PostprocessError::InvalidOptions(
                "nms_iou must be finite and between 0 and 1".into(),
            ));
        }
        if self.max_detections == 0 {
            return Err(PostprocessError::InvalidOptions(
                "max_detections must be positive".into(),
            ));
        }
        if self
            .class_thresholds
            .values()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(PostprocessError::InvalidOptions(
                "class thresholds must be finite and between 0 and 1".into(),
            ));
        }
        Ok(())
    }
}

pub trait Postprocessor: Send + Sync {
    fn postprocess(
        &self,
        outputs: &[RawTensor],
        options: &PostprocessOptions,
    ) -> Result<Vec<Detection>, PostprocessError>;
}

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    YoloGeneric,
    Ssd,
}

impl ModelKind {
    pub fn infer(outputs: &[RawTensor]) -> Result<Self, PostprocessError> {
        if outputs
            .iter()
            .any(|tensor| tensor.shape.len() == 3 && tensor.shape[2] == 64)
        {
            return Ok(Self::YoloGeneric);
        }
        let has_boxes = outputs
            .iter()
            .any(|tensor| tensor.shape.len() == 3 && tensor.shape[2] == 4);
        let has_count = outputs.iter().any(|tensor| tensor.shape.as_slice() == [1]);
        if outputs.len() == 4 && has_boxes && has_count {
            return Ok(Self::Ssd);
        }
        Err(PostprocessError::UnrecognizedOutputs(
            outputs.iter().map(|tensor| tensor.shape.clone()).collect(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum PostprocessError {
    #[error("invalid postprocess options: {0}")]
    InvalidOptions(String),
    #[error("invalid output tensor: {0}")]
    InvalidTensor(String),
    #[error("unrecognized output tensor shapes: {0:?}")]
    UnrecognizedOutputs(Vec<Vec<usize>>),
    #[error("model output class count {actual} does not match {expected} labels")]
    ClassCount { expected: usize, actual: usize },
    #[error("model input size {input_size} implies {expected} anchors, output contains {actual}")]
    AnchorCount {
        input_size: u32,
        expected: usize,
        actual: usize,
    },
}

fn stable_nms_by<T>(
    mut candidates: Vec<T>,
    iou_threshold: f32,
    confidence: impl Fn(&T) -> f32,
    bbox: impl Fn(&T) -> Box2d,
) -> Vec<T> {
    candidates.sort_by(|left, right| {
        confidence(right)
            .partial_cmp(&confidence(left))
            .unwrap_or(Ordering::Equal)
    });
    let mut kept = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if kept
            .iter()
            .all(|prior| bbox(prior).iou(bbox(&candidate)) <= iou_threshold)
        {
            kept.push(candidate);
        }
    }
    kept
}

fn filter_detections(
    detections: impl IntoIterator<Item = Detection>,
    options: &PostprocessOptions,
) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|detection| {
            options
                .classes
                .as_ref()
                .is_none_or(|classes| classes.contains(&detection.class_name))
        })
        .filter(|detection| {
            options
                .class_thresholds
                .get(&detection.class_name)
                .is_none_or(|threshold| detection.confidence > *threshold)
        })
        .take(options.max_detections)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(class_id: usize, class_name: &str, confidence: f32, bbox: Box2d) -> Detection {
        Detection {
            class_id,
            class_name: class_name.into(),
            confidence,
            bbox,
        }
    }

    #[test]
    fn nms_is_class_agnostic_and_allowlist_applies_after() {
        let bbox = Box2d {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        };
        let nms = stable_nms_by(
            vec![
                detection(1, "car", 0.9, bbox),
                detection(0, "person", 0.8, bbox),
            ],
            0.5,
            |detection| detection.confidence,
            |detection| detection.bbox,
        );
        let options = PostprocessOptions {
            classes: Some(HashSet::from(["person".into()])),
            ..PostprocessOptions::default()
        };

        assert!(filter_detections(nms, &options).is_empty());
    }
}
