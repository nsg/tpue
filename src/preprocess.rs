use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::Box2d;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PreprocessMode {
    #[default]
    Letterbox,
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreprocessTransform {
    pub source_width: u32,
    pub source_height: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_left: u32,
    pub pad_top: u32,
}

impl PreprocessTransform {
    pub fn map_box(self, bbox: Box2d) -> Box2d {
        let scale_x = self.resized_width as f32 / self.source_width as f32;
        let scale_y = self.resized_height as f32 / self.source_height as f32;
        Box2d {
            x1: ((bbox.x1 - self.pad_left as f32) / scale_x).clamp(0.0, self.source_width as f32),
            y1: ((bbox.y1 - self.pad_top as f32) / scale_y).clamp(0.0, self.source_height as f32),
            x2: ((bbox.x2 - self.pad_left as f32) / scale_x).clamp(0.0, self.source_width as f32),
            y2: ((bbox.y2 - self.pad_top as f32) / scale_y).clamp(0.0, self.source_height as f32),
        }
    }

    pub fn map_box_normalized(self, bbox: Box2d) -> Box2d {
        let mapped = self.map_box(bbox);
        Box2d {
            x1: mapped.x1 / self.source_width as f32,
            y1: mapped.y1 / self.source_height as f32,
            x2: mapped.x2 / self.source_width as f32,
            y2: mapped.y2 / self.source_height as f32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub pixels: Vec<u8>,
    pub transform: PreprocessTransform,
}

#[derive(Debug)]
pub struct Preprocessor {
    resizer: Resizer,
    resize_options: ResizeOptions,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            resizer: Resizer::new(),
            resize_options: ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        }
    }

    pub fn preprocess(
        &mut self,
        image: &DynamicImage,
        input_width: u32,
        input_height: u32,
        mode: PreprocessMode,
    ) -> Result<PreparedImage, PreprocessError> {
        let rgb = image.to_rgb8();
        let (source_width, source_height) = rgb.dimensions();
        if source_width == 0 || source_height == 0 {
            return Err(PreprocessError::EmptyImage);
        }
        if input_width == 0 || input_height == 0 {
            return Err(PreprocessError::InvalidInputSize);
        }
        let (resized_width, resized_height) = match mode {
            PreprocessMode::Stretch => (input_width, input_height),
            PreprocessMode::Letterbox => {
                let scale = (input_width as f64 / source_width as f64)
                    .min(input_height as f64 / source_height as f64);
                (
                    (source_width as f64 * scale).round().max(1.0) as u32,
                    (source_height as f64 * scale).round().max(1.0) as u32,
                )
            }
        };
        let pad_left = (input_width - resized_width) / 2;
        let pad_top = (input_height - resized_height) / 2;
        let source =
            Image::from_vec_u8(source_width, source_height, rgb.into_raw(), PixelType::U8x3)?;
        let mut resized = Image::new(resized_width, resized_height, PixelType::U8x3);
        self.resizer
            .resize(&source, &mut resized, &self.resize_options)?;
        let pixels = if mode == PreprocessMode::Stretch {
            resized.into_vec()
        } else {
            let mut letterboxed = vec![114; input_width as usize * input_height as usize * 3];
            let resized = resized.buffer();
            let row_bytes = resized_width as usize * 3;
            for row in 0..resized_height as usize {
                let source_start = row * row_bytes;
                let destination_start =
                    ((pad_top as usize + row) * input_width as usize + pad_left as usize) * 3;
                letterboxed[destination_start..destination_start + row_bytes]
                    .copy_from_slice(&resized[source_start..source_start + row_bytes]);
            }
            letterboxed
        };
        Ok(PreparedImage {
            pixels,
            transform: PreprocessTransform {
                source_width,
                source_height,
                input_width,
                input_height,
                resized_width,
                resized_height,
                pad_left,
                pad_top,
            },
        })
    }
}

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("image dimensions must be positive")]
    EmptyImage,
    #[error("model input dimensions must be positive")]
    InvalidInputSize,
    #[error("invalid RGB image buffer: {0}")]
    InvalidImageBuffer(#[from] fast_image_resize::ImageBufferError),
    #[error("could not resize image: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_maps_box_back_to_source() {
        let transform = PreprocessTransform {
            source_width: 640,
            source_height: 360,
            input_width: 512,
            input_height: 512,
            resized_width: 512,
            resized_height: 288,
            pad_left: 0,
            pad_top: 112,
        };

        let mapped = transform.map_box(Box2d {
            x1: 80.0,
            y1: 148.0,
            x2: 400.0,
            y2: 292.0,
        });

        assert_eq!(
            mapped,
            Box2d {
                x1: 100.0,
                y1: 45.0,
                x2: 500.0,
                y2: 225.0,
            }
        );

        let clamped = transform.map_box(Box2d {
            x1: -10.0,
            y1: 0.0,
            x2: 600.0,
            y2: 520.0,
        });

        assert_eq!(
            clamped,
            Box2d {
                x1: 0.0,
                y1: 0.0,
                x2: 640.0,
                y2: 360.0,
            }
        );
    }
}
