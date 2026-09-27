<div align="center">
  <h1>tpue</h1>
  <p>Object detection service for a Google Coral USB Edge TPU, written in Rust.</p>
</div>

<div align="center">

[![AI usage: vibe](https://nsg.github.io/aibadge/vibe.svg)](https://nsg.github.io/aibadge/#vibe)

</div>

## About

tpue turns a Coral USB Edge TPU into a small HTTP service. A camera system
posts a frame, and the service answers with JSON listing the objects it found,
with class, confidence, and bounding box. It is built for a home camera setup
where the question is simply "is there a person, a vehicle, or an animal in
this frame", without sending footage to a GPU or cloud service.

One inference thread owns the Coral. The HTTP API accepts JPEG, PNG, JSON
base64, and multipart input; the CLI also supports one-shot detection.

## Quick Start

```bash
cargo build --release
```

The binary is `target/release/tpue`. It has no build-time dependencies beyond
Rust; the Edge TPU runtime and the TensorFlow Lite C library are loaded at
startup from the system, see [Runtime requirements](#runtime-requirements).

Run it with a configuration file:

```bash
tpue serve --config tpue.toml
tpue detect --config tpue.toml frame.jpg
tpue models --config tpue.toml
```

## Models

Models are not bundled in the binary and are not downloaded by the service.
Each entry in the configuration names a `.tflite` file and a label file on
disk, and the service reads them at runtime.

At startup the service loads the model marked `default = true`, attaches it to
the Coral, and runs one warm-up inference so the first request does not pay
the parameter upload. Other configured models are loaded on their first
request and stay resident afterwards. All models share the single Coral, so
switching models costs a re-upload of parameters to the device.

The models tpue was developed and measured with, all compiled for the Edge TPU:

| Model | Model file | Label file |
|---|---|---|
| YOLOv9-s ReLU6 512, 17 COCO classes, primary | [`yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite`](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.5/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite) | [`labels-coco17.txt`](https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/labels-coco17.txt) |
| YOLOv9-s ReLU6 320, 17 COCO classes, faster alternative | [`yolov9-s-relu6-best_320_int8_edgetpu.tflite`](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.0/yolov9-s-relu6-best_320_int8_edgetpu.tflite) | same as above |
| SSDLite MobileDet, 90-index COCO | [`ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite`](https://github.com/google-coral/test_data/raw/release-frogfish/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite) | [Frigate v0.18.0 `labelmap.txt`](https://raw.githubusercontent.com/blakeblackshear/frigate/v0.18.0/labelmap.txt) |

Neither the model files nor their label files are in this repository; they
belong to the models. Download both for each model you configure and point
the configuration at them. A label file has one class name per line, either
plain or prefixed with its numeric index. Any Edge TPU compiled model with a
supported output layout works: `yolo-generic` for YOLO heads with DFL boxes
and class logits, `ssd` for the four-tensor TFLite detection postprocess.

## Configuration

Pass a configuration with `--config`, or set `TPUE_CONFIG`. The checked-in
[`tpue.toml`](tpue.toml) lists the three models above with the 512 model as
default. The essential parts:

```toml
[server]
bind = "0.0.0.0:8700"
max_body_bytes = 8_000_000
request_timeout_ms = 2000
queue_depth = 8

[runtime]
tflite_library = "libtensorflowlite_c.so"
edgetpu_library = "libedgetpu.so.1"
device = "usb"

[[models]]
name = "yolov9s-512"
default = true
path = "/opt/tpue/models/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite"
labels = "/opt/tpue/models/labels-coco17.txt"
type = "yolo-generic"
input_size = 512
preprocess = "letterbox"
threshold = 0.12
nms_iou = 0.5
max_detections = 50
[models.class_thresholds]
person = 0.12
car = 0.18
dog = 0.22
cat = 0.09
bicycle = 0.17
truck = 0.22
bus = 0.68
```

The thresholds are low on purpose. The quantized YOLOv9 model reports
compressed confidence scores, and the values above were tuned for best F1 per
class on a COCO validation subset. A conventional 0.5 floor would discard most
true detections.

`preprocess` is `letterbox` (keeps the aspect ratio, pads with grey) or
`stretch` (plain resize). The library paths are resolved by the system loader
unless given as absolute paths. Devices may be `usb`, `usb:0`, `pci`, or
`pci:0`.

## API

Send raw JPEG bytes for the lowest overhead:

```bash
curl --fail-with-body -H 'Content-Type: image/jpeg' \
  --data-binary @frame.jpg http://127.0.0.1:8700/v1/detect
```

`GET /healthz` reports readiness, `GET /v1/models` lists configured models,
`GET /v1/stats` returns request counters and stage timings,
`GET /openapi.json` returns the OpenAPI document, `GET /` shows live
statistics, and `GET /docs` serves the reference as HTML. See [docs/api.md](docs/api.md) for request forms,
overrides, response fields, concurrency, and errors.

## Runtime requirements

The service runs on x86-64 Linux with the Coral in a USB 3 port. It needs two
shared libraries on the host, and both must come from the same TensorFlow
version because the Edge TPU runtime is compiled against a specific
TensorFlow Lite ABI.

**Edge TPU runtime**, `libedgetpu.so.1`. Google's original packages are
unmaintained; use the [feranick/libedgetpu](https://github.com/feranick/libedgetpu/releases)
builds, release `16.0TF2.19.1-1`, package `libedgetpu1-std` for your
distribution. It installs the library and the udev rule for the Coral's two
USB IDs. Do not install `libedgetpu1-max`; it overclocks the device and needs
cooler ambient air.

**TensorFlow Lite C library**, `libtensorflowlite_c.so`, built from TensorFlow
v2.19.1 source:

```bash
git clone --depth 1 --branch v2.19.1 https://github.com/tensorflow/tensorflow.git
cmake -S tensorflow/tensorflow/lite/c -B tflite-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=ON \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_INSTALL=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,noexecstack"
cmake --build tflite-build --parallel 8 --target tensorflowlite_c
```

The `noexecstack` linker flag matters: without it the v2.19.1 build is marked
as requiring an executable stack, and glibc 2.41 and newer refuse to load it.

What goes where on the target host:

| Item | Place |
|---|---|
| `target/release/tpue` | anywhere, for example `/opt/tpue/bin/` |
| `tpue.toml` | anywhere; pass it with `--config` |
| model `.tflite` files and label files | the paths named in the configuration |
| `libtensorflowlite_c.so` | a directory the system loader searches, for example `/usr/local/lib`, then run `ldconfig` |
| `libedgetpu.so.1` and its udev rule | installed by the package above |

The user running the service needs access to the Coral; the udev rule grants
it to the `plugdev` group. The device can be opened by only one process at a
time. Verify the device before starting: it enumerates as `1a6e:089a` when
plugged in and as `18d1:9302` once the runtime has uploaded its firmware.

## License

MIT, see `LICENSE`. The YOLOv9 model files are MIT licensed by their author
and the MobileDet model comes from Google's Coral test data under Apache-2.0.
