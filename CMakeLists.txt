cmake_minimum_required(VERSION 3.16)

project(model)

set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/thirdparty/onnxruntime)
set(ONNXRUNTIME_BUILD_DIR ${ONNXRUNTIME_DIR}/build/Android/MinSizeRel)
set(ANDROID_HOME ${PROJECT_SOURCE_DIR}/thirdparty/Android/Sdk)
set(NDK_HOME ${ANDROID_HOME}/ndk/21.3.6528147)

include_directories(
  ${ONNXRUNTIME_DIR}/include/onnxruntime
  ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session
  ${PROJECT_SOURCE_DIR}/thirdparty/sltbench/install/include/sltbench)

include(ExternalProject)
set(ANDROID_FLAGS -DCMAKE_TOOLCHAIN_FILE=${NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=29)
externalproject_add(sltbench
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/sltbench
    CONFIGURE_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/thirdparty/sltbench/install -DCMAKE_BUILD_TYPE=Release ${ANDROID_FLAGS} ${PROJECT_SOURCE_DIR}/thirdparty/sltbench
    BUILD_COMMAND cmake --build . --target install
    INSTALL_COMMAND "")

add_library(cpp_shared_lib SHARED IMPORTED)
add_library(onnxruntime_lib SHARED IMPORTED)
add_library(omp_lib STATIC IMPORTED)
add_library(log_lib SHARED IMPORTED)

set_property(TARGET cpp_shared_lib PROPERTY IMPORTED_LOCATION ${NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so)
set_property(TARGET onnxruntime_lib PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime.so)
set_property(TARGET omp_lib PROPERTY IMPORTED_LOCATION ${NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/lib/linux/aarch64/libomp.a)
set_property(TARGET log_lib PROPERTY IMPORTED_LOCATION ${NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/liblog.so)

add_executable(onnx_model model.cpp)
add_dependencies(onnx_model sltbench)

add_library(sltbench_lib STATIC IMPORTED)
set_property(TARGET sltbench_lib PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/thirdparty/sltbench/install/lib/libsltbench.a)

target_link_libraries(onnx_model PUBLIC sltbench_lib cpp_shared_lib onnxruntime_lib omp_lib log_lib)
