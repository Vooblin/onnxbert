#!/bin/sh

set -e -x

cur_dir="$(dirname $(readlink -f $0))"
cd $cur_dir

mkdir -p "./Android/Sdk"
cd "./Android/Sdk"

if [ ! -d "tools" ]
then
  wget "https://dl.google.com/android/repository/commandlinetools-linux-6200805_latest.zip"
  unzip "commandlinetools-linux-6200805_latest.zip"
  rm "commandlinetools-linux-6200805_latest.zip"
fi

FLAGS="--sdk_root=."

yes | ./tools/bin/sdkmanager $FLAGS "ndk;21.0.6113669" "platform-tools"

cd $cur_dir

if [ ! -d "sltbench" ]
then
    git clone --recursive "https://github.com/ivafanas/sltbench.git"
fi

if [ ! -d "onnxruntime" ]
then
  git clone --recursive "https://github.com/microsoft/onnxruntime.git"
fi
cd onnxruntime
git submodule update --recursive

if [ ! -d "build" ]
then
  ./build.sh --android --android_sdk_path "$cur_dir/Android/Sdk" --android_ndk_path "$cur_dir/Android/Sdk/ndk/21.0.6113669" --android_abi "arm64-v8a" --android_api 24 --config MinSizeRel
fi

cd $cur_dir

pip3 install "torch==1.4.0+cpu" "torchvision==0.5.0+cpu" -f "https://download.pytorch.org/whl/torch_stable.html"
if [ ! -d "bert" ]
then
  mkdir -p bert
  cd bert
  wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json"
  wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
  cd $cur_dir
fi
python3 get_bert.py
python3 "./onnxruntime/onnxruntime/python/tools/bert/bert_model_optimization.py" --input "./bert/torch.onnx" --output "./bert/torch_opt.onnx" --model_type bert --num_heads 16 --hidden_size 1024 --sequence_length 128

rm $cur_dir/build -r
mkdir -p $cur_dir/build
cd $cur_dir/build
cmake .. -DCMAKE_TOOLCHAIN_FILE="$cur_dir/Android/Sdk/ndk/21.0.6113669/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=24
make
$cur_dir/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip $cur_dir/build/onnx_model
