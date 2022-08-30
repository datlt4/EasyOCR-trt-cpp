# Trtexec-DynamicShape

## Create yolov4 engine

``` bash
./Trtexec \
    --onnx <yolov4-onnx-file> \
    --engine <yolov4-engine-file> \
    --inputName "000_net" \
    --minShape 1x3x608x608 \
    --optShape 1x3x608x608 \
    --maxShape 1x3x608x608 \
    --workspace 1024
```
## Create AlphaPose engine

```bash
./Trtexec \
    --onnx <alpha-pose-onnx-file> \
    --engine <alpha-pose-engine-file> \
    --inputName "input" \
    --minShape 1x3x256x192 \
    --optShape 8x3x256x192 \
    --maxShape 32x3x256x192 \
    --workspace 1024
    --dynamicOnnx
```

## Create DeepSort engine

```bash
./Trtexec \
    --onnx <deepsort-onnx-file> \
    --engine <deepsort-engine-file> \
    --inputName "input" \
    --minShape 1x3x128x64 \
    --optShape 8x3x128x64 \
    --maxShape 32x3x128x64 \
    --workspace 1024 \
    --dynamicOnnx
```

## Create EasyOCR-recognition engine

```bash
./Trtexec \
    --onnx <easyocr-recognition-onnx-file> \
    --engine <easyocr-recognition-engine-file> \
    --inputName "input1" \
    --minShape 1x1x64x64 \
    --optShape 1x1x64x1280 \
    --maxShape 1x1x64x2560 \
    --workspace 1024 \
    --dynamicOnnx
```

## Create EasyOCR-recognition engine

```bash
./Trtexec \
    --onnx <easyocr-detection-onnx-file> \
    --engine <easyocr-detection-engine-file> \
    --inputName "input" \
    --minShape 1x3x360x480 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x720x720 \
    --workspace 1024 \
    --dynamicOnnx
```
