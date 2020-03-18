#include "onnxruntime_cxx_api.h"
#include <vector>
#include <cstdio>
#include <iostream>
#include <ctime>

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  //*************************************************************************
  // create session and load model into memory
  const char* model_path = "/data/local/tmp/torch.onnx";

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of outputs = %zu\n", num_output_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Output %d : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Output %d : type=%d\n", i, type);

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    for (int j = 0; j < output_node_dims.size(); j++)
      printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
  }

  const int batch_size = 1;
  const int max_seq_length = 128;
  typedef long ttype;
  //size_t input_0_tensor_size = batch_size;
  size_t input_1_tensor_size = batch_size * max_seq_length;
  size_t input_2_tensor_size = batch_size * max_seq_length;
  size_t input_3_tensor_size = batch_size * max_seq_length;

  //std::vector<int64_t> input_0_node_dims{batch_size};
  std::vector<int64_t> input_1_node_dims{batch_size, max_seq_length};
  std::vector<int64_t> input_2_node_dims{batch_size, max_seq_length};
  std::vector<int64_t> input_3_node_dims{batch_size, max_seq_length};

  //std::vector<long> input_0_tensor_values(input_0_tensor_size);
  std::vector<ttype> input_1_tensor_values(input_1_tensor_size);
  std::vector<ttype> input_2_tensor_values(input_2_tensor_size);
  std::vector<ttype> input_3_tensor_values(input_3_tensor_size);

  //for (unsigned int i = 0; i < input_0_tensor_size; i++)
  //  input_0_tensor_values[i] = i;
  for (unsigned int i = 0; i < input_1_tensor_size; i++)
    input_1_tensor_values[i] = i + 1;
  for (unsigned int i = 0; i < input_2_tensor_size; i++)
    input_2_tensor_values[i] = 1;
  for (unsigned int i = 0; i < input_3_tensor_size; i++)
    input_3_tensor_values[i] = 0;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  //Ort::Value input_0_tensor = Ort::Value::CreateTensor<long>(memory_info, input_0_tensor_values.data(), input_0_tensor_size, input_0_node_dims.data(), 1);
  Ort::Value input_1_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_1_tensor_values.data(), input_1_tensor_size, input_1_node_dims.data(), 2);
  Ort::Value input_2_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_2_tensor_values.data(), input_2_tensor_size, input_2_node_dims.data(), 2);
  Ort::Value input_3_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_3_tensor_values.data(), input_3_tensor_size, input_3_node_dims.data(), 2);

  //Ort::Value input_tensors[4] = {std::move(input_0_tensor), std::move(input_1_tensor), std::move(input_2_tensor), std::move(input_3_tensor)};
  Ort::Value input_tensors[3] = {std::move(input_1_tensor), std::move(input_2_tensor), std::move(input_3_tensor)};

  struct timeval start, end; 
  gettimeofday(&start, NULL);
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors, 3, output_node_names.data(), 2);
  gettimeofday(&end, NULL);
  double time_taken;
  time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec -
                              start.tv_usec)) * 1e-6;
  std::cout << "Time taken by program is : " << std::fixed
       << time_taken;
  std::cout << " sec" << std::endl;

  float* output_0 = output_tensors[0].GetTensorMutableData<float>();
  float* output_1 = output_tensors[1].GetTensorMutableData<float>();
  for (int i = 0; i < 10; ++i) {
      std::cout << output_0[i] << "   " << output_1[i] << std::endl;
  }
  return 0;
}
