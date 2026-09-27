use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::DynamicImage;
use pulldown_cmark::{Options as MarkdownOptions, Parser as MarkdownParser, html};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tower_http::trace::TraceLayer;

use crate::config::{Config, ModelConfig};
use crate::models::{
    Labels, ModelKind, PostprocessOptions, Postprocessor, Quantization, RawTensor, Ssd, TensorData,
    YoloGeneric,
};
use crate::preprocess::{PreprocessMode, Preprocessor};
use crate::stats::{Stats, StatsResponse};
use crate::tflite::{Device, ElementType, Interpreter, Model, Runtime};

#[derive(Debug, Clone, Serialize)]
pub struct DetectionResponse {
    pub detections: Vec<ApiDetection>,
    pub model: String,
    pub image: ImageSize,
    pub timing_ms: Timing,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiDetection {
    pub class: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    pub decode: f64,
    pub queue: f64,
    pub preprocess: f64,
    pub inference: f64,
    pub postprocess: f64,
    pub total: f64,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
    device: String,
    uptime_s: u64,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<ModelResponse>,
}

#[derive(Debug, Serialize)]
struct ModelResponse {
    name: String,
    file: String,
    input_size: Option<u32>,
    labels: String,
    model_type: Option<String>,
    preprocess: String,
    threshold: f32,
    nms_iou: f32,
    max_detections: usize,
    class_thresholds: BTreeMap<String, f32>,
    default: bool,
}

#[derive(Debug, Default, Deserialize)]
struct DetectQuery {
    threshold: Option<f32>,
    classes: Option<String>,
    max_detections: Option<usize>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JsonRequest {
    image: String,
    threshold: Option<f32>,
    classes: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct RequestOptions {
    threshold: Option<f32>,
    classes: Option<HashSet<String>>,
    max_detections: Option<usize>,
}

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    engine: Engine,
    stats: Arc<Stats>,
    started: Instant,
}

#[derive(Clone)]
struct Engine {
    sender: mpsc::SyncSender<Job>,
    health: Arc<Mutex<EngineHealth>>,
    stats: Arc<Stats>,
}

#[derive(Debug, Default)]
struct EngineHealth {
    ready: bool,
    error: Option<String>,
    input_sizes: HashMap<String, u32>,
}

struct Job {
    model: String,
    image: DynamicImage,
    decode_ms: f64,
    queued: Instant,
    options: RequestOptions,
    response: oneshot::Sender<Result<DetectionResponse, String>>,
}

struct LoadedModel {
    interpreter: Interpreter,
    processor: Box<dyn Postprocessor>,
    config: ModelConfig,
    input_width: u32,
    input_height: u32,
    input_type: ElementType,
}

impl Engine {
    fn start(config: Arc<Config>, stats: Arc<Stats>) -> Self {
        let (sender, receiver) = mpsc::sync_channel(config.server.queue_depth);
        let health = Arc::new(Mutex::new(EngineHealth::default()));
        let worker_health = Arc::clone(&health);
        let worker_stats = Arc::clone(&stats);
        let preprocessor = Preprocessor::new();
        std::thread::Builder::new()
            .name("tpue-inference".into())
            .spawn(move || {
                inference_thread(config, receiver, worker_health, worker_stats, preprocessor)
            })
            .expect("could not start inference thread");
        Self {
            sender,
            health,
            stats,
        }
    }

    async fn detect(
        &self,
        model: String,
        image: DynamicImage,
        decode_ms: f64,
        options: RequestOptions,
        timeout: Duration,
    ) -> Result<DetectionResponse, ApiError> {
        let (response, receiver) = oneshot::channel();
        self.stats.enqueued();
        self.sender
            .try_send(Job {
                model,
                image,
                decode_ms,
                queued: Instant::now(),
                options,
                response,
            })
            .map_err(|error| {
                self.stats.dequeued();
                match error {
                    mpsc::TrySendError::Full(_) => ApiError::unavailable("inference queue is full"),
                    mpsc::TrySendError::Disconnected(_) => {
                        ApiError::unavailable("inference thread is unavailable")
                    }
                }
            })?;
        tokio::time::timeout(timeout, receiver)
            .await
            .map_err(|_| ApiError::timeout("inference request timed out"))?
            .map_err(|_| ApiError::unavailable("inference thread stopped"))?
            .map_err(ApiError::unavailable)
    }
}

fn inference_thread(
    config: Arc<Config>,
    receiver: mpsc::Receiver<Job>,
    health: Arc<Mutex<EngineHealth>>,
    stats: Arc<Stats>,
    mut preprocessor: Preprocessor,
) {
    let runtime = match Runtime::load(
        &config.runtime.tflite_library,
        &config.runtime.edgetpu_library,
    ) {
        Ok(runtime) => runtime,
        Err(error) => {
            set_health_error(&health, error.to_string());
            return;
        }
    };
    let device = match Device::from_str(&config.runtime.device) {
        Ok(device) => device,
        Err(error) => {
            set_health_error(&health, error.to_string());
            return;
        }
    };
    let mut models = HashMap::new();
    let default = config.default_model().clone();
    match load_model(&runtime, device, default.clone()).and_then(|mut loaded| {
        warm_up(&mut loaded)?;
        Ok(loaded)
    }) {
        Ok(loaded) => {
            health
                .lock()
                .expect("health lock poisoned")
                .input_sizes
                .insert(default.name.clone(), loaded.input_width);
            models.insert(default.name.clone(), loaded);
            let mut state = health.lock().expect("health lock poisoned");
            state.ready = true;
            state.error = None;
        }
        Err(error) => set_health_error(&health, error.to_string()),
    }

    while let Ok(job) = receiver.recv() {
        stats.dequeued();
        let result = run_job(
            &runtime,
            device,
            &config,
            &mut models,
            &health,
            &mut preprocessor,
            &job,
        );
        {
            let mut state = health.lock().expect("health lock poisoned");
            match &result {
                Ok(_) => {
                    state.ready = true;
                    state.error = None;
                }
                Err(error) => {
                    state.ready = false;
                    state.error = Some(error.to_string());
                }
            }
        }
        let _ = job.response.send(result.map_err(|error| error.to_string()));
    }
}

fn load_model(runtime: &Runtime, device: Device, config: ModelConfig) -> Result<LoadedModel> {
    let model = Model::load(runtime, &config.path)
        .with_context(|| format!("could not load model {}", config.path.display()))?;
    let interpreter = Interpreter::new(model, device, 1)
        .with_context(|| format!("could not initialize model {}", config.name))?;
    if interpreter.input_count()? != 1 {
        bail!("model {} must have exactly one input tensor", config.name);
    }
    let input = interpreter.input(0)?;
    let shape = input.shape()?;
    if shape.len() != 4 || shape[0] != 1 || shape[1] == 0 || shape[1] != shape[2] || shape[3] != 3 {
        bail!(
            "model {} has invalid NHWC input shape {shape:?}",
            config.name
        );
    }
    let input_type = input.element_type();
    if !matches!(input_type, ElementType::UInt8 | ElementType::Int8) {
        bail!("model {} input must be uint8 or int8", config.name);
    }
    let input_size = u32::try_from(shape[1]).context("model input size exceeds u32")?;
    if config
        .input_size
        .is_some_and(|expected| expected != input_size)
    {
        bail!(
            "model {} configured input_size does not match tensor size {input_size}",
            config.name
        );
    }
    let kind = match config.model_type {
        Some(kind) => kind,
        None => infer_model_kind(&interpreter)?,
    };
    let labels = Labels::load(&config.labels)
        .with_context(|| format!("could not load labels {}", config.labels.display()))?;
    let processor: Box<dyn Postprocessor> = match kind {
        ModelKind::YoloGeneric => Box::new(YoloGeneric::new(input_size, labels)?),
        ModelKind::Ssd => Box::new(Ssd::new(input_size, labels)?),
    };
    Ok(LoadedModel {
        interpreter,
        processor,
        config,
        input_width: input_size,
        input_height: input_size,
        input_type,
    })
}

fn infer_model_kind(interpreter: &Interpreter) -> Result<ModelKind> {
    let mut shapes = Vec::new();
    for index in 0..interpreter.output_count()? {
        shapes.push(interpreter.output(index)?.shape()?);
    }
    if shapes
        .iter()
        .any(|shape| shape.len() == 3 && shape[2] == 64)
    {
        return Ok(ModelKind::YoloGeneric);
    }
    if shapes.len() == 4
        && shapes.iter().any(|shape| shape.len() == 3 && shape[2] == 4)
        && shapes.iter().any(|shape| shape.as_slice() == [1])
    {
        return Ok(ModelKind::Ssd);
    }
    bail!("cannot identify model type from output shapes {shapes:?}")
}

fn warm_up(model: &mut LoadedModel) -> Result<()> {
    let size = model.interpreter.input(0)?.byte_size();
    model.interpreter.write_input(0, &vec![0; size])?;
    model.interpreter.invoke()?;
    Ok(())
}

fn run_job(
    runtime: &Runtime,
    device: Device,
    config: &Config,
    models: &mut HashMap<String, LoadedModel>,
    health: &Arc<Mutex<EngineHealth>>,
    preprocessor: &mut Preprocessor,
    job: &Job,
) -> Result<DetectionResponse> {
    let queue_ms = elapsed_ms(job.queued);
    if !models.contains_key(&job.model) {
        let model_config = config
            .model(&job.model)
            .cloned()
            .ok_or_else(|| anyhow!("unknown model {:?}", job.model))?;
        let loaded = load_model(runtime, device, model_config)?;
        health
            .lock()
            .expect("health lock poisoned")
            .input_sizes
            .insert(job.model.clone(), loaded.input_width);
        models.insert(job.model.clone(), loaded);
    }
    let model = models.get_mut(&job.model).expect("model inserted");
    let preprocess_started = Instant::now();
    let prepared = preprocessor.preprocess(
        &job.image,
        model.input_width,
        model.input_height,
        model.config.preprocess,
    )?;
    let preprocess_ms = elapsed_ms(preprocess_started);
    let mut input = prepared.pixels;
    if model.input_type == ElementType::Int8 {
        for value in &mut input {
            *value ^= 128;
        }
    }
    let inference_started = Instant::now();
    model.interpreter.write_input(0, &input)?;
    model.interpreter.invoke()?;
    let inference_ms = elapsed_ms(inference_started);
    let postprocess_started = Instant::now();
    let outputs = read_outputs(&model.interpreter)?;
    let options = postprocess_options(&model.config, &job.options);
    let detections = model.processor.postprocess(&outputs, &options)?;
    let detections = detections
        .into_iter()
        .map(|detection| {
            let bbox = prepared.transform.map_box_normalized(detection.bbox);
            ApiDetection {
                class: detection.class_name,
                confidence: detection.confidence.clamp(0.0, 1.0),
                x: bbox.x1.clamp(0.0, 1.0),
                y: bbox.y1.clamp(0.0, 1.0),
                w: (bbox.x2 - bbox.x1).clamp(0.0, 1.0),
                h: (bbox.y2 - bbox.y1).clamp(0.0, 1.0),
            }
        })
        .collect::<Vec<_>>();
    let postprocess_ms = elapsed_ms(postprocess_started);
    Ok(DetectionResponse {
        detections,
        model: job.model.clone(),
        image: ImageSize {
            width: prepared.transform.source_width,
            height: prepared.transform.source_height,
        },
        timing_ms: Timing {
            decode: job.decode_ms,
            queue: queue_ms,
            preprocess: preprocess_ms,
            inference: inference_ms,
            postprocess: postprocess_ms,
            total: job.decode_ms + queue_ms + preprocess_ms + inference_ms + postprocess_ms,
        },
    })
}

fn postprocess_options(config: &ModelConfig, request: &RequestOptions) -> PostprocessOptions {
    PostprocessOptions {
        threshold: request.threshold.unwrap_or(config.threshold),
        nms_iou: config.nms_iou,
        max_detections: request.max_detections.unwrap_or(config.max_detections),
        class_thresholds: config
            .class_thresholds
            .iter()
            .map(|(name, threshold)| (name.clone(), *threshold))
            .collect(),
        classes: request.classes.clone(),
    }
}

fn read_outputs(interpreter: &Interpreter) -> Result<Vec<RawTensor>> {
    let output_count = interpreter.output_count()?;
    let mut outputs = Vec::with_capacity(output_count);
    for index in 0..output_count {
        let tensor = interpreter.output(index)?;
        let shape = tensor.shape()?;
        let name = tensor.name().unwrap_or_else(|| format!("output_{index}"));
        let (data, quantization) = match tensor.element_type() {
            ElementType::Int8 => {
                let parameters = tensor.quantization();
                (
                    TensorData::I8(tensor.to_i8_vec()?),
                    Some(Quantization {
                        scale: parameters.scale,
                        zero_point: parameters.zero_point,
                    }),
                )
            }
            ElementType::Float32 => {
                let bytes = tensor.to_vec()?;
                if !bytes.len().is_multiple_of(4) {
                    bail!(
                        "float output {name:?} has invalid byte length {}",
                        bytes.len()
                    );
                }
                let values = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                (TensorData::F32(values), None)
            }
            other => bail!("unsupported output tensor type {other:?} for {name:?}"),
        };
        outputs.push(RawTensor {
            name,
            shape,
            data,
            quantization,
        });
    }
    Ok(outputs)
}

fn set_health_error(health: &Arc<Mutex<EngineHealth>>, error: String) {
    let mut state = health.lock().expect("health lock poisoned");
    state.ready = false;
    state.error = Some(error);
}

pub async fn serve(config: Config) -> Result<()> {
    let bind = config.server.bind.clone();
    let max_body_bytes = config.server.max_body_bytes;
    let config = Arc::new(config);
    let stats = Arc::new(Stats::new(config.server.queue_depth));
    let state = AppState {
        engine: Engine::start(Arc::clone(&config), Arc::clone(&stats)),
        config,
        stats,
        started: Instant::now(),
    };
    let app = Router::new()
        .route("/", get(root))
        .route("/docs", get(docs))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models_endpoint))
        .route("/v1/stats", get(stats_endpoint))
        .route("/openapi.json", get(openapi))
        .route("/v1/detect", post(detect_endpoint))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("could not bind {bind}"))?;
    tracing::info!(%bind, "server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

pub async fn detect_file(config: Config, path: &Path) -> Result<DetectionResponse> {
    let bytes =
        std::fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
    let decode_started = Instant::now();
    let image = tokio::task::spawn_blocking(move || image::load_from_memory(&bytes))
        .await
        .context("image decoder task stopped")??;
    let decode_ms = elapsed_ms(decode_started);
    let config = Arc::new(config);
    let model = config.default_model().name.clone();
    let timeout =
        Duration::from_secs(30).max(Duration::from_millis(config.server.request_timeout_ms));
    let stats = Arc::new(Stats::new(config.server.queue_depth));
    Engine::start(Arc::clone(&config), stats)
        .detect(
            model,
            image,
            decode_ms,
            RequestOptions {
                threshold: None,
                classes: None,
                max_detections: None,
            },
            timeout,
        )
        .await
        .map_err(|error| anyhow!(error.message))
}

pub fn models_json(config: &Config) -> serde_json::Value {
    serde_json::to_value(ModelsResponse {
        models: config
            .models
            .iter()
            .map(|model| model_response(model, None))
            .collect(),
    })
    .expect("models response serializes")
}

async fn detect_endpoint(State(state): State<AppState>, request: Request) -> Response {
    let _in_flight = state.stats.begin();
    match detect_request(&state, request).await {
        Ok(response) => {
            state.stats.record_success(&response);
            tracing::info!(
                model = %response.model,
                width = response.image.width,
                height = response.image.height,
                detections = response.detections.len(),
                total_ms = response.timing_ms.total,
                inference_ms = response.timing_ms.inference,
                "detection complete"
            );
            Json(response).into_response()
        }
        Err(error) => {
            state
                .stats
                .record_error(error.status.as_u16(), &error.message);
            error.into_response()
        }
    }
}

async fn detect_request(state: &AppState, request: Request) -> Result<DetectionResponse, ApiError> {
    let request_started = Instant::now();
    let mut query = Query::<DetectQuery>::try_from_uri(request.uri())
        .map_err(|error| ApiError::bad_request(error.to_string()))?
        .0;
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let (bytes, json_overrides) = if content_type.starts_with("multipart/form-data") {
        let mut multipart = Multipart::from_request(request, state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let mut image = None;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            if field.name() == Some("image") {
                image = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| ApiError::bad_request(error.to_string()))?,
                );
                break;
            }
        }
        (
            image.ok_or_else(|| ApiError::bad_request("multipart field image is required"))?,
            None,
        )
    } else {
        if !matches!(
            content_type.split(';').next().unwrap_or(""),
            "image/jpeg" | "image/png" | "application/json"
        ) {
            return Err(ApiError::unsupported("unsupported Content-Type"));
        }
        let bytes = to_bytes(request.into_body(), state.config.server.max_body_bytes)
            .await
            .map_err(|_| ApiError::too_large("request body exceeds max_body_bytes"))?;
        if content_type.starts_with("application/json") {
            let json: JsonRequest = serde_json::from_slice(&bytes)
                .map_err(|error| ApiError::bad_request(format!("invalid JSON: {error}")))?;
            let decoded = BASE64
                .decode(&json.image)
                .map_err(|error| ApiError::bad_request(format!("invalid base64 image: {error}")))?;
            (Bytes::from(decoded), Some(json))
        } else {
            (bytes, None)
        }
    };
    if bytes.is_empty() {
        return Err(ApiError::bad_request("image is empty"));
    }
    state.stats.record_image_bytes(bytes.len());
    if let Some(json) = json_overrides {
        if query.threshold.is_none() {
            query.threshold = json.threshold;
        }
        if query.classes.is_none() {
            query.classes = json.classes.map(|classes| classes.join(","));
        }
    }
    let options = request_options(query.threshold, query.classes, query.max_detections)?;
    let model = query
        .model
        .unwrap_or_else(|| state.config.default_model().name.clone());
    if state.config.model(&model).is_none() {
        return Err(ApiError::bad_request(format!("unknown model {model:?}")));
    }
    let decode_started = Instant::now();
    let image = tokio::task::spawn_blocking(move || image::load_from_memory(&bytes))
        .await
        .map_err(|_| ApiError::bad_request("image decoder task stopped"))?
        .map_err(|error| ApiError::bad_request(format!("could not decode image: {error}")))?;
    let decode_ms = elapsed_ms(decode_started);
    let timeout = Duration::from_millis(state.config.server.request_timeout_ms);
    let mut response = state
        .engine
        .detect(model, image, decode_ms, options, timeout)
        .await?;
    response.timing_ms.total = elapsed_ms(request_started);
    Ok(response)
}

fn request_options(
    threshold: Option<f32>,
    classes: Option<String>,
    max_detections: Option<usize>,
) -> Result<RequestOptions, ApiError> {
    if threshold.is_some_and(|value| !value.is_finite() || !(0.0 < value && value < 1.0)) {
        return Err(ApiError::bad_request("threshold must be between 0 and 1"));
    }
    if max_detections == Some(0) {
        return Err(ApiError::bad_request("max_detections must be positive"));
    }
    let classes = classes
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_lowercase)
                .collect::<HashSet<_>>()
        })
        .filter(|classes| !classes.is_empty());
    Ok(RequestOptions {
        threshold,
        classes,
        max_detections,
    })
}

async fn healthz(State(state): State<AppState>) -> Response {
    let health = state.engine.health.lock().expect("health lock poisoned");
    if health.ready {
        Json(HealthResponse {
            status: "ok",
            model: state.config.default_model().name.clone(),
            device: state.config.runtime.device.clone(),
            uptime_s: state.started.elapsed().as_secs(),
        })
        .into_response()
    } else {
        ApiError::unavailable(
            health
                .error
                .clone()
                .unwrap_or_else(|| "model is warming up".into()),
        )
        .into_response()
    }
}

async fn models_endpoint(State(state): State<AppState>) -> Json<ModelsResponse> {
    let health = state.engine.health.lock().expect("health lock poisoned");
    Json(ModelsResponse {
        models: state
            .config
            .models
            .iter()
            .map(|model| model_response(model, health.input_sizes.get(&model.name).copied()))
            .collect(),
    })
}

fn model_response(model: &ModelConfig, input_size: Option<u32>) -> ModelResponse {
    ModelResponse {
        name: model.name.clone(),
        file: model.path.display().to_string(),
        input_size: input_size.or(model.input_size),
        labels: model.labels.display().to_string(),
        model_type: model.model_type.map(|kind| match kind {
            ModelKind::YoloGeneric => "yolo-generic".into(),
            ModelKind::Ssd => "ssd".into(),
        }),
        preprocess: match model.preprocess {
            PreprocessMode::Letterbox => "letterbox".into(),
            PreprocessMode::Stretch => "stretch".into(),
        },
        threshold: model.threshold,
        nms_iou: model.nms_iou,
        max_detections: model.max_detections,
        class_thresholds: model.class_thresholds.clone(),
        default: model.default,
    }
}

async fn stats_endpoint(State(state): State<AppState>) -> Json<StatsResponse> {
    Json(state.stats.snapshot())
}

async fn root() -> Html<&'static str> {
    Html(include_str!("stats.html"))
}

async fn docs() -> Html<String> {
    let mut rendered = String::new();
    let parser = MarkdownParser::new_ext(
        include_str!("../docs/api.md"),
        MarkdownOptions::ENABLE_TABLES | MarkdownOptions::ENABLE_STRIKETHROUGH,
    );
    html::push_html(&mut rendered, parser);
    Html(include_str!("docs.html").replace("{{docs}}", &rendered))
}

async fn openapi() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(include_str!("../docs/openapi.json")))
        .expect("valid OpenAPI response")
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    fn too_large(message: impl Into<String>) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, message)
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNSUPPORTED_MEDIA_TYPE, message)
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, message)
    }

    fn timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, message)
    }

    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}
