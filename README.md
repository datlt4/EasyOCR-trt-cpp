# EasyOCR-trt-cpp

## Clone

```bash
git clone --recursive https://github.com/LuongTanDat/EasyOCR-trt-cpp.git
cd EasyOCR-trt-cpp
```

## Download models

1. Download from `dvc`

```bash
dvc pull
```

2. Download from `jaided.ai` [model hub](https://www.jaided.ai/easyocr/modelhub/)

- [CRAFT](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip)
- [english_g2](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip)

## Convert `EasyOCR` to `onnx`

[Reference](https://github.com/JaidedAI/EasyOCR/issues/746)

1. Install environment

```bash
cd EasyOCR-python
python3 -m pip install -r requirements.txt
```

2. Convert


```bash
# python3 -c "import easyocr; reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./saved_model', download_enabled=False); print(reader.readtext('examples/full_image/Lorem-ipsum-dolor.png', detail=0))"
# - Two output `onnx` file located in `saved_model/recognitionModel.onnx` and `saved_model/detectionModel.onnx`.

cd EasyOCR-python
python3 to_onnx.py --pth "saved_model/craft_mlt_25k.pth" --onnx "saved_model/detectionModel.onnx" --device "cuda" --detection
python3 to_onnx.py --pth "saved_model/english_g2.pth" --onnx "saved_model/recognitionModel.onnx" --device "cuda" --recognition
```


## Convert `onnx` to `tensorRT`

### Create EasyOCR-text-detection engine

```bash
./TrtExec-bin \
    --onnx ../../weights/detectionModel.onnx \
    --engine ../../weights/detectionModel.engine \
    --inputName "input" \
    --minShape 1x3x360x480 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x720x720 \
    --workspace 1024 \
    --dynamicOnnx
```

### Create EasyOCR-recognition engine

```bash
./TrtExec-bin \
    --onnx ../../weights/recognitionModel.onnx \
    --engine ../../weights/recognitionModel.engine \
    --inputName "input1" \
    --minShape 1x1x64x64 \
    --optShape 1x1x64x1280 \
    --maxShape 1x1x64x2560 \
    --workspace 1024 \
    --dynamicOnnx
```
