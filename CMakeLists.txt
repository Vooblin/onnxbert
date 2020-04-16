cmake_minimum_required(VERSION 3.16)

project(model)

set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/onnxruntime)
set(ANDROID_HOME ${PROJECT_SOURCE_DIR}/Android/Sdk)

include_directories(
  ${ONNXRUNTIME_DIR}/include/onnxruntime
  ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session
  ${PROJECT_SOURCE_DIR}/sltbench/install/include/sltbench)

include(ExternalProject)
set(ANDROID_FLAGS -DCMAKE_TOOLCHAIN_FILE=${ANDROID_HOME}/ndk/21.0.6113669/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=24)
externalproject_add(sltbench
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/sltbench
    CONFIGURE_COMMAND cmake -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/sltbench/install -DCMAKE_BUILD_TYPE=Release ${ANDROID_FLAGS} ${PROJECT_SOURCE_DIR}/sltbench
    BUILD_COMMAND cmake --build . --target install
    INSTALL_COMMAND "")

add_library(lib1 STATIC IMPORTED)
add_library(lib2 STATIC IMPORTED)
add_library(lib3 STATIC IMPORTED)
add_library(lib4 STATIC IMPORTED)
add_library(lib5 STATIC IMPORTED)
add_library(lib6 STATIC IMPORTED)
add_library(lib7 STATIC IMPORTED)
add_library(lib8 STATIC IMPORTED)
add_library(lib9 STATIC IMPORTED)
add_library(lib10 STATIC IMPORTED)
add_library(lib11 STATIC IMPORTED)
add_library(lib12 STATIC IMPORTED)
add_library(lib13 STATIC IMPORTED)
add_library(lib14 STATIC IMPORTED)
add_library(lib15 STATIC IMPORTED)
add_library(lib16 STATIC IMPORTED)

set_property(TARGET lib1 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_session.a)
set_property(TARGET lib2 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_test_utils.a)
set_property(TARGET lib3 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_optimizer.a)
set_property(TARGET lib4 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_providers.a)
set_property(TARGET lib5 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_util.a)
set_property(TARGET lib6 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_graph.a)
set_property(TARGET lib7 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_framework.a)
set_property(TARGET lib8 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_common.a)
set_property(TARGET lib9 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/onnx/libonnx.a)
set_property(TARGET lib10 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/onnx/libonnx_proto.a)
set_property(TARGET lib11 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/external/protobuf/cmake/libprotobuf-lite.a)
set_property(TARGET lib12 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/external/nsync/libnsync_cpp.a)
set_property(TARGET lib13 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/external/re2/libre2.a)
set_property(TARGET lib14 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/MinSizeRel/libonnxruntime_mlas.a)
set_property(TARGET lib15 PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/lib/linux/aarch64/libomp.a)
set_property(TARGET lib16 PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/24/liblog.so)

add_executable(onnx_model model.cpp)
add_dependencies(onnx_model sltbench)

add_library(sltbench_lib STATIC IMPORTED)
set_property(TARGET sltbench_lib PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/sltbench/install/lib/libsltbench.a)

target_link_libraries(onnx_model PUBLIC sltbench_lib lib1 lib2 lib3 lib4 lib5 lib6 lib7 lib8 lib9 lib10 lib11 lib12 lib13 lib14 lib15 lib16)
