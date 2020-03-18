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

if [ ! -d "cmake-3.16.5" ]
then
  wget "https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz"
  tar -xf "cmake-3.16.5.tar.gz"
  rm "cmake-3.16.5.tar.gz"
fi
cd "cmake-3.16.5"
if [ ! -d "bin" ]
then
  ./bootstrap -- -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_USE_OPENSSL=OFF
  sudo make
  sudo make install
fi

cd $cur_dir

if [ ! -d "onnxruntime" ]
then
  git clone --recursive "https://github.com/microsoft/onnxruntime.git"
fi
cd onnxruntime
git submodule update --recursive

if [ ! -d "build" ]
then
  ./build.sh --android --android_sdk_path "$cur_dir/Android/Sdk" --android_ndk_path "$cur_dir/Android/Sdk/ndk/21.0.6113669" --android_abi "arm64-v8a" --android_api 24
fi

cd $cur_dir

pip3 install "torch==1.4.0+cpu" "torchvision==0.5.0+cpu" -f "https://download.pytorch.org/whl/torch_stable.html"
if [ ! -d "bert" ]
then
  mkdir -p bert
  cd bert
  wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
  wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"
  mv "bert-base-uncased-pytorch_model.bin" "pytorch_model.bin"
  cd $cur_dir
fi
python3 get_bert.py
python3 "./onnxruntime/onnxruntime/python/tools/bert/bert_model_optimization.py" --input "./bert/torch.onnx" --output "./bert/torch_opt.onnx" --model_type bert --num_heads 12 --hidden_size 768 --sequence_length 128

mkdir -p build
cd build
cmake ..
