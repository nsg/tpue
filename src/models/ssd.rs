use super::{
    Box2d, Detection, Labels, PostprocessError, PostprocessOptions, Postprocessor, RawTensor,
    filter_detections,
};

#[derive(Clone, Debug)]
pub struct Ssd {
    input_size: u32,
    labels: Labels,
}

impl Ssd {
    pub fn new(input_size: u32, labels: Labels) -> Result<Self, PostprocessError> {
        if input_size == 0 {
            return Err(PostprocessError::InvalidOptions(
                "SSD input size must be positive".into(),
            ));
        }
        Ok(Self { input_size, labels })
    }

    fn values(tensor: &RawTensor) -> Vec<f32> {
        (0..tensor.element_count())
            .map(|index| tensor.value(index))
            .collect()
    }

    fn is_integer_like(values: &[f32]) -> bool {
        values
            .iter()
            .all(|value| value.is_finite() && (*value - value.round()).abs() <= 1e-5)
    }
}

impl Postprocessor for Ssd {
    fn postprocess(
        &self,
        outputs: &[RawTensor],
        options: &PostprocessOptions,
    ) -> Result<Vec<Detection>, PostprocessError> {
        options.validate()?;
        for tensor in outputs {
            tensor.validate()?;
        }
        if outputs.len() != 4 {
            return Err(PostprocessError::UnrecognizedOutputs(
                outputs.iter().map(|tensor| tensor.shape.clone()).collect(),
            ));
        }
        let boxes: Vec<_> = outputs
            .iter()
            .filter(|tensor| {
                tensor.shape.len() == 3 && tensor.shape[0] == 1 && tensor.shape[2] == 4
            })
            .collect();
        let counts: Vec<_> = outputs
            .iter()
            .filter(|tensor| tensor.shape.as_slice() == [1])
            .collect();
        let vectors: Vec<_> = outputs
            .iter()
            .filter(|tensor| tensor.shape.len() == 2 && tensor.shape[0] == 1)
            .collect();
        if boxes.len() != 1 || counts.len() != 1 || vectors.len() != 2 {
            return Err(PostprocessError::UnrecognizedOutputs(
                outputs.iter().map(|tensor| tensor.shape.clone()).collect(),
            ));
        }
        let boxes = Self::values(boxes[0]);
        let first = Self::values(vectors[0]);
        let second = Self::values(vectors[1]);
        let (classes, scores) = match (
            Self::is_integer_like(&first),
            Self::is_integer_like(&second),
        ) {
            (true, false) => (&first, &second),
            (false, true) => (&second, &first),
            _ => (&first, &second),
        };
        let count = counts[0]
            .value(0)
            .max(0.0)
            .floor()
            .min((boxes.len() / 4).min(classes.len()).min(scores.len()) as f32)
            as usize;
        let mut detections = Vec::new();
        for index in 0..count {
            let score = scores[index];
            if score < options.threshold {
                break;
            }
            let class_id = classes[index] as usize;
            let offset = index * 4;
            detections.push(Detection {
                class_id,
                class_name: self.labels.get(class_id),
                confidence: score,
                bbox: Box2d {
                    x1: boxes[offset + 1] * self.input_size as f32,
                    y1: boxes[offset] * self.input_size as f32,
                    x2: boxes[offset + 3] * self.input_size as f32,
                    y2: boxes[offset + 2] * self.input_size as f32,
                },
            });
        }
        Ok(filter_detections(detections, options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TensorData;

    fn tensor(name: &str, shape: &[usize], values: &[f32]) -> RawTensor {
        RawTensor {
            name: name.into(),
            shape: shape.into(),
            data: TensorData::F32(values.into()),
            quantization: None,
        }
    }

    #[test]
    fn ssd_decodes_postprocess_tensors() {
        let processor = Ssd::new(100, Labels::parse("person\ncar").unwrap()).unwrap();
        let outputs = vec![
            tensor(
                "boxes",
                &[1, 2, 4],
                &[0.125, 0.25, 0.5, 0.75, 0.0, 0.0, 1.0, 1.0],
            ),
            tensor("classes", &[1, 2], &[1.0, 0.0]),
            tensor("scores", &[1, 2], &[0.75, 0.99]),
            tensor("count", &[1], &[1.0]),
        ];

        let detections = processor
            .postprocess(&outputs, &PostprocessOptions::default())
            .unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(
            (detections[0].class_id, detections[0].class_name.as_str()),
            (1, "car")
        );
        assert_eq!(detections[0].confidence, 0.75);
        assert_eq!(
            detections[0].bbox,
            Box2d {
                x1: 25.0,
                y1: 12.5,
                x2: 75.0,
                y2: 50.0,
            }
        );
    }
}
