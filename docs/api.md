# tpue HTTP API

tpue exposes an unauthenticated HTTP API intended for a trusted LAN. Endpoint
responses are JSON; the documentation page at `/` is HTML.

## Detect objects

`POST /v1/detect` accepts one image in any of these forms:

| Content-Type | Body |
|---|---|
| `image/jpeg` | Raw JPEG bytes |
| `image/png` | Raw PNG bytes |
| `application/json` | `{"image":"<base64>","threshold":0.2,"classes":["person","car"]}` |
| `multipart/form-data` | File data in the `image` field |

Raw JPEG is the preferred format because it avoids base64 and multipart
overhead. Send images as they are: any size and aspect ratio is accepted, and
the service resizes internally, so there is no benefit in resizing or padding
on the client. The body limit is `server.max_body_bytes`, 8 MB by default.

Optional query parameters override configured request defaults:

| Parameter | Value |
|---|---|
| `threshold` | Global score floor, strictly between 0 and 1. Only set this deliberately: the configured per-class thresholds are tuned for the quantized models, whose confidence scale is compressed, and a conventional floor such as 0.5 discards most true detections. Filter with `classes` instead when only some labels matter. |
| `classes` | Comma-separated lower-case class allowlist |
| `max_detections` | Positive result limit |
| `model` | Configured model name |

```bash
curl --fail-with-body \
  -H 'Content-Type: image/jpeg' \
  --data-binary @frame.jpg \
  'http://127.0.0.1:8700/v1/detect?classes=person,car&threshold=0.2'
```

A successful response has this shape:

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.83,
      "x": 0.41,
      "y": 0.22,
      "w": 0.09,
      "h": 0.31
    }
  ],
  "model": "yolov9s-512",
  "image": {"width": 640, "height": 360},
  "timing_ms": {
    "decode": 1.8,
    "preprocess": 0.6,
    "inference": 34.9,
    "postprocess": 0.4,
    "total": 38.1
  }
}
```

Coordinates use a top-left origin and are normalized to the submitted image.
Each value is clamped to 0 through 1; `w` and `h` are width and height, not
bottom-right coordinates. If the submitted image was a crop taken at fraction
`(cx, cy, cw, ch)` of a larger frame, the box in that frame is
`(cx + x*cw, cy + y*ch, w*cw, h*ch)`. The class names come from the model's
label file; `GET /v1/models` lists them. Results are sorted by descending confidence. The
service performs class-agnostic NMS before applying a request class allowlist.
Configured per-class thresholds apply after the global score floor.

Configured stretch and letterbox preprocessing use `fast_image_resize` with
its convolution Bilinear filter. Letterbox mode preserves aspect ratio, adds
centered grey-114 padding, and maps result boxes back to the submitted image.

## Concurrency and latency

The service has one inference lane and a small queue (`server.queue_depth`).
Concurrent requests are accepted and served in order; when the queue is full
the request is rejected with `503`. Each request is bounded by
`server.request_timeout_ms` and answered with `504` when exceeded. Latency is
dominated by the Edge TPU inference plus JPEG decoding of large images, so
clients get the best throughput by sending one request at a time.

## Service state

`GET /healthz` returns `200` after the default model has completed its warm-up:

```json
{"status":"ok","model":"yolov9s-512","device":"usb","uptime_s":123}
```

It returns `503` while the model is not ready or if the device disappears.

`GET /v1/models` lists every configured model with its name, file, input size,
label source, thresholds, preprocessing mode, and whether it is the default.
Only the default model is loaded at startup; another model loads on its first
request and remains resident.

`GET /openapi.json` returns the hand-written OpenAPI 3.0.3 description embedded
at build time. `GET /` renders this document as HTML.

## Errors

Errors use `{"error":"message"}` and one of these status codes:

| Status | Meaning |
|---:|---|
| `400` | Empty or undecodable image, malformed request, or invalid override |
| `413` | Request exceeds `server.max_body_bytes` |
| `415` | Unsupported `Content-Type` |
| `503` | Model/device unavailable or inference queue full |
| `504` | Inference exceeded `server.request_timeout_ms` |

There is no authentication or TLS termination. Bind to a trusted interface or
place the service behind an authenticated reverse proxy when it is reachable
outside the LAN.
