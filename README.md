# onnxbert
```
./onnxbert_mobile.sh
cd bert
adb push torch.onnx /data/local/tmp
adb push torch_opt.onnx /data/local/tmp
cd ../build
adb push onnx_model /data/local/tmp
adb shell /data/local/tmp/onnx_model
```
