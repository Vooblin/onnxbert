cmake_minimum_required(VERSION 3.16)

project(model)

set(ONNXRUNTIME_DIR ./onnxruntime)

include_directories(
  ${ONNXRUNTIME_DIR}/include/onnxruntime
  ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)

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

set_property(TARGET lib1 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_session.a)
set_property(TARGET lib2 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_test_utils.a)
set_property(TARGET lib3 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_optimizer.a)
set_property(TARGET lib4 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_providers.a)
set_property(TARGET lib5 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_util.a)
set_property(TARGET lib6 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_graph.a)
set_property(TARGET lib7 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_common.a)
set_property(TARGET lib8 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/onnx/libonnx.a)
set_property(TARGET lib9 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/onnx/libonnx_proto.a)
set_property(TARGET lib10 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/external/protobuf/cmake/libprotobuf-lited.a)
set_property(TARGET lib11 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/external/nsync/libnsync_cpp.a)
set_property(TARGET lib12 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/external/re2/libre2.a)
set_property(TARGET lib13 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_framework.a)
set_property(TARGET lib14 PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/build/Linux/Debug/libonnxruntime_mlas.a)
set_property(TARGET lib15 PROPERTY IMPORTED_LOCATION ./Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/9.0.8/lib/linux/aarch64/libomp.a)

add_executable(model model.cpp)

find_library(log-lib log)

target_link_libraries(model PUBLIC lib1 lib2 lib3 lib4 lib5 lib6 lib7 lib8 lib9 lib10 lib11 lib12 lib13 lib14 lib15 ${log-lib})
