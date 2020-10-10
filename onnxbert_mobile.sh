#!/bin/sh

set -e -x

PROJECT_DIR="$(dirname $(readlink -f $0))"
cd $PROJECT_DIR
git submodule update --init

mkdir -p $PROJECT_DIR/thirdparty/Android/Sdk
cd $PROJECT_DIR/thirdparty/Android/Sdk

if [ ! -d tools ]
then
  wget https://dl.google.com/android/repository/commandlinetools-linux-6609375_latest.zip
  unzip commandlinetools-linux-6609375_latest.zip
  rm commandlinetools-linux-6609375_latest.zip
fi

if [ ! -d platform-tools ]
then
  FLAGS=--sdk_root=.

  yes | ./tools/bin/sdkmanager $FLAGS "ndk;21.3.6528147" platform-tools
fi

cd $PROJECT_DIR/thirdparty

if [ ! -d cmake ]
then
  wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz
  tar -xf cmake-3.18.4-Linux-x86_64.tar.gz
  rm cmake-3.18.4-Linux-x86_64.tar.gz
  mv cmake-3.18.4-Linux-x86_64 cmake
fi

ANDROID_HOME=$PROJECT_DIR/thirdparty/Android/Sdk
NDK_HOME=$ANDROID_HOME/ndk/21.3.6528147
PATH=$PROJECT_DIR/thirdparty/cmake/bin:$PATH

cd $PROJECT_DIR

python3 $PROJECT_DIR/get_bert.py --size tiny
python3 $PROJECT_DIR/get_bert.py --size base

python3 $PROJECT_DIR/thirdparty/onnxruntime/tools/python/convert_onnx_models_to_ort.py $PROJECT_DIR/bert

$PROJECT_DIR/thirdparty/onnxruntime/build.sh --android --android_sdk_path $ANDROID_HOME --android_ndk_path $NDK_HOME --android_abi arm64-v8a --android_api 29 --config=MinSizeRel --build_shared_lib --android_cpp_shared --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config $PROJECT_DIR/bert/required_operators.config --parallel --use_openmp

rm $PROJECT_DIR/build -r
mkdir -p $PROJECT_DIR/build
cd $PROJECT_DIR/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=29
make
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip $PROJECT_DIR/build/onnx_model
