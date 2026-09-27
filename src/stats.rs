use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::server::DetectionResponse;

const WINDOW: usize = 4096;
const RATE_WINDOW: Duration = Duration::from_secs(60);
const STAGES: usize = 6;

pub struct Stats {
    started: Instant,
    queue_depth: usize,
    in_flight: AtomicUsize,
    queued: AtomicUsize,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    succeeded: u64,
    errors: BTreeMap<u16, u64>,
    last_error: Option<(Instant, u16, String)>,
    last_request: Option<Instant>,
    finished: VecDeque<Instant>,
    images: u64,
    image_bytes: u64,
    image_pixels: u64,
    detections: u64,
    images_with_detections: u64,
    classes: BTreeMap<String, u64>,
    models: BTreeMap<String, u64>,
    stage_sums: [f64; STAGES],
    samples: VecDeque<Sample>,
}

struct Sample {
    at: Instant,
    ms: [f64; STAGES],
}

pub struct InFlight<'a>(&'a Stats);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub uptime_s: u64,
    pub requests: RequestStats,
    pub pipeline: PipelineStats,
    pub images: ImageStats,
    pub detections: DetectionStats,
    pub models: BTreeMap<String, u64>,
    pub timing_ms: TimingStats,
}

#[derive(Debug, Serialize)]
pub struct RequestStats {
    pub total: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub last_minute: usize,
    pub last_request_age_s: Option<f64>,
    pub errors: BTreeMap<u16, u64>,
    pub last_error: Option<LastError>,
}

#[derive(Debug, Serialize)]
pub struct LastError {
    pub status: u16,
    pub message: String,
    pub age_s: f64,
}

#[derive(Debug, Serialize)]
pub struct PipelineStats {
    pub in_flight: usize,
    pub queued: usize,
    pub queue_depth: usize,
    pub busy_last_minute: f64,
}

#[derive(Debug, Serialize)]
pub struct ImageStats {
    pub received: u64,
    pub bytes: u64,
    pub pixels: u64,
    pub mean_bytes: f64,
    pub mean_megapixels: f64,
}

#[derive(Debug, Serialize)]
pub struct DetectionStats {
    pub total: u64,
    pub images_with_detections: u64,
    pub by_class: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
pub struct TimingStats {
    pub window: usize,
    pub decode: StageStats,
    pub queue: StageStats,
    pub preprocess: StageStats,
    pub inference: StageStats,
    pub postprocess: StageStats,
    pub total: StageStats,
}

#[derive(Debug, Default, Serialize)]
pub struct StageStats {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Stats {
    pub fn new(queue_depth: usize) -> Self {
        Self {
            started: Instant::now(),
            queue_depth,
            in_flight: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            inner: Mutex::default(),
        }
    }

    pub fn begin(&self) -> InFlight<'_> {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        InFlight(self)
    }

    pub fn enqueued(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dequeued(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_image_bytes(&self, bytes: usize) {
        let mut inner = self.lock();
        inner.images += 1;
        inner.image_bytes += bytes as u64;
    }

    pub fn record_success(&self, response: &DetectionResponse) {
        let now = Instant::now();
        let timing = &response.timing_ms;
        let ms = [
            timing.decode,
            timing.queue,
            timing.preprocess,
            timing.inference,
            timing.postprocess,
            timing.total,
        ];
        let mut inner = self.lock();
        inner.finish(now);
        inner.succeeded += 1;
        inner.image_pixels += u64::from(response.image.width) * u64::from(response.image.height);
        inner.detections += response.detections.len() as u64;
        if !response.detections.is_empty() {
            inner.images_with_detections += 1;
        }
        for detection in &response.detections {
            *inner.classes.entry(detection.class.clone()).or_default() += 1;
        }
        *inner.models.entry(response.model.clone()).or_default() += 1;
        for (sum, value) in inner.stage_sums.iter_mut().zip(ms) {
            *sum += value;
        }
        if inner.samples.len() == WINDOW {
            inner.samples.pop_front();
        }
        inner.samples.push_back(Sample { at: now, ms });
    }

    pub fn record_error(&self, status: u16, message: &str) {
        let now = Instant::now();
        let mut inner = self.lock();
        inner.finish(now);
        *inner.errors.entry(status).or_default() += 1;
        inner.last_error = Some((now, status, message.to_owned()));
    }

    pub fn snapshot(&self) -> StatsResponse {
        let now = Instant::now();
        let inner = self.lock();
        let failed = inner.errors.values().sum::<u64>();
        let recent = |at: &Instant| now.duration_since(*at) <= RATE_WINDOW;
        let busy_ms = inner
            .samples
            .iter()
            .filter(|sample| recent(&sample.at))
            .fold(0.0, |sum, sample| {
                sum + sample.ms[2] + sample.ms[3] + sample.ms[4]
            });
        let observed = now.duration_since(self.started).min(RATE_WINDOW);
        let stage = |index: usize| inner.stage(index);
        let per_image = |value: u64, count: u64| {
            if count == 0 {
                0.0
            } else {
                value as f64 / count as f64
            }
        };
        StatsResponse {
            uptime_s: self.started.elapsed().as_secs(),
            requests: RequestStats {
                total: inner.succeeded + failed,
                succeeded: inner.succeeded,
                failed,
                last_minute: inner.finished.iter().filter(|at| recent(at)).count(),
                last_request_age_s: inner
                    .last_request
                    .map(|at| now.duration_since(at).as_secs_f64()),
                errors: inner.errors.clone(),
                last_error: inner
                    .last_error
                    .as_ref()
                    .map(|(at, status, message)| LastError {
                        status: *status,
                        message: message.clone(),
                        age_s: now.duration_since(*at).as_secs_f64(),
                    }),
            },
            pipeline: PipelineStats {
                in_flight: self.in_flight.load(Ordering::Relaxed),
                queued: self.queued.load(Ordering::Relaxed),
                queue_depth: self.queue_depth,
                busy_last_minute: if observed.is_zero() {
                    0.0
                } else {
                    (busy_ms / (observed.as_secs_f64() * 1_000.0)).min(1.0)
                },
            },
            images: ImageStats {
                received: inner.images,
                bytes: inner.image_bytes,
                pixels: inner.image_pixels,
                mean_bytes: per_image(inner.image_bytes, inner.images),
                mean_megapixels: per_image(inner.image_pixels, inner.succeeded) / 1e6,
            },
            detections: DetectionStats {
                total: inner.detections,
                images_with_detections: inner.images_with_detections,
                by_class: inner.classes.clone(),
            },
            models: inner.models.clone(),
            timing_ms: TimingStats {
                window: inner.samples.len(),
                decode: stage(0),
                queue: stage(1),
                preprocess: stage(2),
                inference: stage(3),
                postprocess: stage(4),
                total: stage(5),
            },
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("stats lock poisoned")
    }
}

impl Inner {
    fn finish(&mut self, now: Instant) {
        self.last_request = Some(now);
        while self
            .finished
            .front()
            .is_some_and(|at| now.duration_since(*at) > RATE_WINDOW)
        {
            self.finished.pop_front();
        }
        self.finished.push_back(now);
    }

    fn stage(&self, index: usize) -> StageStats {
        if self.samples.is_empty() {
            return StageStats::default();
        }
        let mut values = self
            .samples
            .iter()
            .map(|sample| sample.ms[index])
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let percentile = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
        StageStats {
            mean: self.stage_sums[index] / self.succeeded as f64,
            p50: percentile(0.5),
            p95: percentile(0.95),
            max: values[values.len() - 1],
        }
    }
}
