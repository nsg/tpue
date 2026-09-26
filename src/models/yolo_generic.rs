use super::{
    Box2d, Detection, Labels, PostprocessError, PostprocessOptions, Postprocessor, RawTensor,
    TensorData, stable_nms_by,
};

const REG_MAX: usize = 16;
const STRIDES: [u32; 3] = [8, 16, 32];

#[derive(Clone, Debug)]
pub struct YoloGeneric {
    input_size: u32,
    labels: Labels,
    anchors: Vec<(f32, f32, f32)>,
}

struct Candidate {
    class_id: usize,
    confidence: f32,
    bbox: Box2d,
}

impl YoloGeneric {
    pub fn new(input_size: u32, labels: Labels) -> Result<Self, PostprocessError> {
        if input_size == 0
            || STRIDES
                .iter()
                .any(|stride| !input_size.is_multiple_of(*stride))
        {
            return Err(PostprocessError::InvalidOptions(format!(
                "YOLO input size {input_size} must be positive and divisible by 32"
            )));
        }
        let mut anchors = Vec::new();
        for stride in STRIDES {
            let feature_size = input_size / stride;
            for y in 0..feature_size {
                for x in 0..feature_size {
                    anchors.push((x as f32 + 0.5, y as f32 + 0.5, stride as f32));
                }
            }
        }
        Ok(Self {
            input_size,
            labels,
            anchors,
        })
    }

    fn find_outputs<'a>(
        &self,
        outputs: &'a [RawTensor],
    ) -> Result<(&'a RawTensor, &'a RawTensor, Option<&'a RawTensor>), PostprocessError> {
        for tensor in outputs {
            tensor.validate()?;
        }
        if !(2..=3).contains(&outputs.len()) {
            return Err(PostprocessError::UnrecognizedOutputs(
                outputs.iter().map(|tensor| tensor.shape.clone()).collect(),
            ));
        }
        let mut boxes = None;
        let mut scores = None;
        let mut auxiliary = None;
        let mut valid = true;
        for tensor in outputs {
            if tensor.shape.len() != 3 || tensor.shape[0] != 1 {
                valid = false;
            } else if tensor.shape[2] == 64 {
                valid &= boxes.replace(tensor).is_none();
            } else if tensor.shape[2] == self.labels.len() {
                valid &= scores.replace(tensor).is_none();
            } else if tensor.shape[2] == 1 {
                valid &= auxiliary.replace(tensor).is_none();
            } else {
                valid = false;
            }
        }
        if !valid || boxes.is_none() || scores.is_none() {
            return Err(PostprocessError::UnrecognizedOutputs(
                outputs.iter().map(|tensor| tensor.shape.clone()).collect(),
            ));
        }
        let boxes = boxes.expect("checked box output");
        let scores = scores.expect("checked score output");
        if boxes.shape[1] != scores.shape[1] {
            return Err(PostprocessError::InvalidTensor(
                "YOLO box and class tensors have different anchor counts".into(),
            ));
        }
        if boxes.shape[1] != self.anchors.len() {
            return Err(PostprocessError::AnchorCount {
                input_size: self.input_size,
                expected: self.anchors.len(),
                actual: boxes.shape[1],
            });
        }
        if let Some(auxiliary) = auxiliary
            && auxiliary.shape[1] != boxes.shape[1]
        {
            return Err(PostprocessError::InvalidTensor(
                "YOLO auxiliary tensor has a different anchor count".into(),
            ));
        }
        Ok((boxes, scores, auxiliary))
    }

    fn decode_distances(boxes: &RawTensor, anchor_index: usize) -> [f32; 4] {
        let mut distances = [0.0; 4];
        let anchor_offset = anchor_index * 4 * REG_MAX;
        match &boxes.data {
            TensorData::I8(values) => {
                let quantization = boxes.quantization.expect("validated quantization");
                let values = &values[anchor_offset..anchor_offset + 4 * REG_MAX];
                let (distributions, remainder) = values.as_chunks::<REG_MAX>();
                debug_assert!(remainder.is_empty());
                for (distance, distribution) in distances.iter_mut().zip(distributions) {
                    let maximum = distribution
                        .iter()
                        .map(|value| {
                            (i32::from(*value) - quantization.zero_point) as f32
                                * quantization.scale
                        })
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut denominator = 0.0;
                    let mut numerator = 0.0;
                    for (bin, value) in distribution.iter().enumerate() {
                        let value = (i32::from(*value) - quantization.zero_point) as f32
                            * quantization.scale;
                        let exponential = (value - maximum).exp();
                        denominator += exponential;
                        numerator += exponential * bin as f32;
                    }
                    *distance = numerator / denominator;
                }
            }
            TensorData::F32(values) => {
                let values = &values[anchor_offset..anchor_offset + 4 * REG_MAX];
                let (distributions, remainder) = values.as_chunks::<REG_MAX>();
                debug_assert!(remainder.is_empty());
                for (distance, distribution) in distances.iter_mut().zip(distributions) {
                    let maximum = distribution
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut denominator = 0.0;
                    let mut numerator = 0.0;
                    for (bin, value) in distribution.iter().enumerate() {
                        let exponential = (*value - maximum).exp();
                        denominator += exponential;
                        numerator += exponential * bin as f32;
                    }
                    *distance = numerator / denominator;
                }
            }
        }
        distances
    }

    fn candidate(
        &self,
        boxes: &RawTensor,
        anchor_index: usize,
        class_id: usize,
        confidence: f32,
    ) -> Candidate {
        let (anchor_x, anchor_y, stride) = self.anchors[anchor_index];
        let [left, top, right, bottom] = Self::decode_distances(boxes, anchor_index);
        Candidate {
            class_id,
            confidence,
            bbox: Box2d {
                x1: (anchor_x - left) * stride,
                y1: (anchor_y - top) * stride,
                x2: (anchor_x + right) * stride,
                y2: (anchor_y + bottom) * stride,
            },
        }
    }
}

impl Postprocessor for YoloGeneric {
    fn postprocess(
        &self,
        outputs: &[RawTensor],
        options: &PostprocessOptions,
    ) -> Result<Vec<Detection>, PostprocessError> {
        options.validate()?;
        let (boxes, scores, auxiliary) = self.find_outputs(outputs)?;
        let class_count = self.labels.len();
        let mut candidates = Vec::new();
        let minimum_logit = (options.threshold / (1.0 - options.threshold)).ln();
        match &scores.data {
            TensorData::I8(values) => {
                let quantization = scores.quantization.expect("validated quantization");
                let minimum_quantized =
                    (minimum_logit / quantization.scale + quantization.zero_point as f32) as i32;
                let auxiliary_gate = auxiliary.and_then(|tensor| match &tensor.data {
                    TensorData::I8(maxima) => {
                        let quantization = tensor.quantization.expect("validated quantization");
                        let minimum = (minimum_logit / quantization.scale
                            + quantization.zero_point as f32)
                            as i32;
                        Some((maxima.as_slice(), minimum))
                    }
                    TensorData::F32(_) => None,
                });
                for (anchor_index, logits) in values.chunks_exact(class_count).enumerate() {
                    if auxiliary_gate
                        .is_some_and(|(maxima, minimum)| i32::from(maxima[anchor_index]) < minimum)
                    {
                        continue;
                    }
                    let mut class_id = 0;
                    let mut maximum = logits[0];
                    for (candidate_id, &candidate) in logits.iter().enumerate().skip(1) {
                        if candidate > maximum {
                            class_id = candidate_id;
                            maximum = candidate;
                        }
                    }
                    if auxiliary_gate.is_none() && i32::from(maximum) < minimum_quantized {
                        continue;
                    }
                    let logit =
                        (i32::from(maximum) - quantization.zero_point) as f32 * quantization.scale;
                    let confidence = 1.0 / (1.0 + (-logit).exp());
                    if confidence <= options.threshold {
                        continue;
                    }
                    candidates.push(self.candidate(boxes, anchor_index, class_id, confidence));
                }
            }
            TensorData::F32(values) => {
                for (anchor_index, logits) in values.chunks_exact(class_count).enumerate() {
                    let mut class_id = 0;
                    let mut logit = logits[0];
                    for (candidate_id, &candidate) in logits.iter().enumerate().skip(1) {
                        if candidate > logit {
                            class_id = candidate_id;
                            logit = candidate;
                        }
                    }
                    let confidence = 1.0 / (1.0 + (-logit).exp());
                    if confidence <= options.threshold {
                        continue;
                    }
                    candidates.push(self.candidate(boxes, anchor_index, class_id, confidence));
                }
            }
        }
        let candidates = stable_nms_by(
            candidates,
            options.nms_iou,
            |candidate| candidate.confidence,
            |candidate| candidate.bbox,
        );
        Ok(candidates
            .into_iter()
            .filter_map(|candidate| {
                let class_name = self.labels.get(candidate.class_id);
                if options
                    .classes
                    .as_ref()
                    .is_some_and(|classes| !classes.contains(&class_name))
                    || options
                        .class_thresholds
                        .get(&class_name)
                        .is_some_and(|threshold| candidate.confidence <= *threshold)
                {
                    return None;
                }
                Some(Detection {
                    class_id: candidate.class_id,
                    class_name,
                    confidence: candidate.confidence,
                    bbox: candidate.bbox,
                })
            })
            .take(options.max_detections)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Quantization;

    #[test]
    fn yolo_decodes_one_anchor() {
        let processor = YoloGeneric::new(32, Labels::parse("person\ncar").unwrap()).unwrap();
        let anchor_count = 4 * 4 + 2 * 2 + 1;
        let anchor_index = 10;
        let box_quantization = Quantization {
            scale: 1.0,
            zero_point: -7,
        };
        let mut box_values = vec![i8::MIN; anchor_count * 4 * REG_MAX];
        for (side, bin) in [1, 2, 1, 1].into_iter().enumerate() {
            box_values[(anchor_index * 4 + side) * REG_MAX + bin] = i8::MAX;
        }
        let score_quantization = Quantization {
            scale: 0.25,
            zero_point: -4,
        };
        let mut score_values = vec![-12; anchor_count * 2];
        score_values[anchor_index * 2 + 1] = 4;
        let outputs = vec![
            RawTensor {
                name: "boxes".into(),
                shape: vec![1, anchor_count, 64],
                data: TensorData::I8(box_values),
                quantization: Some(box_quantization),
            },
            RawTensor {
                name: "scores".into(),
                shape: vec![1, anchor_count, 2],
                data: TensorData::I8(score_values),
                quantization: Some(score_quantization),
            },
        ];

        let detections = processor
            .postprocess(&outputs, &PostprocessOptions::default())
            .unwrap();

        assert_eq!(detections.len(), 1);
        let detection = &detections[0];
        assert_eq!(
            (detection.class_id, detection.class_name.as_str()),
            (1, "car")
        );
        let expected_score = 1.0 / (1.0 + (-2.0_f32).exp());
        assert!((detection.confidence - expected_score).abs() <= 1e-4);
        let expected_box = Box2d {
            x1: 12.0,
            y1: 4.0,
            x2: 28.0,
            y2: 28.0,
        };
        for (actual, expected) in [
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        ]
        .into_iter()
        .zip([
            expected_box.x1,
            expected_box.y1,
            expected_box.x2,
            expected_box.y2,
        ]) {
            assert!((actual - expected).abs() <= 1e-3);
        }
    }
}
