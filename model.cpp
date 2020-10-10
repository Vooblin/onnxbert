#include "onnxruntime_cxx_api.h"
#include <vector>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <sys/time.h>

#include <Bench.h>

class BertFixture
{
public:
	typedef Ort::Session Type;

    BertFixture() {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(8);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* model_path = "/data/local/tmp/bert_base.ort";
        session.push_back(Ort::Session(env, model_path, session_options));
    }

	Type& SetUp()
	{
		return session[0];
	}

	void TearDown() {}

private:
    std::vector<Ort::Session> session;
};

void bert_benchmark(BertFixture::Type& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;
  for (int i = 0; i < num_input_nodes; i++) {
    char* input_name = session.GetInputName(i, allocator);
    input_node_names[i] = input_name;
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    input_node_dims = tensor_info.GetShape();
  }
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;
  for (int i = 0; i < num_output_nodes; i++) {
    char* output_name = session.GetOutputName(i, allocator);
    output_node_names[i] = output_name;
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    output_node_dims = tensor_info.GetShape();
  }
  const int batch_size = 1;
  const int max_seq_length = 512;
  typedef long ttype;
  size_t input_1_tensor_size = batch_size * max_seq_length;
  size_t input_2_tensor_size = batch_size * max_seq_length;
  size_t input_3_tensor_size = batch_size * max_seq_length;
  std::vector<int64_t> input_1_node_dims{batch_size, max_seq_length};
  std::vector<int64_t> input_2_node_dims{batch_size, max_seq_length};
  std::vector<int64_t> input_3_node_dims{batch_size, max_seq_length};
  std::vector<ttype> input_1_tensor_values(input_1_tensor_size);
  std::vector<ttype> input_2_tensor_values(input_2_tensor_size);
  std::vector<ttype> input_3_tensor_values(input_3_tensor_size);
  for (unsigned int i = 0; i < input_1_tensor_size; i++)
    input_1_tensor_values[i] = i + 1;
  for (unsigned int i = 0; i < input_2_tensor_size; i++)
    input_2_tensor_values[i] = 1;
  for (unsigned int i = 0; i < input_3_tensor_size; i++)
    input_3_tensor_values[i] = 0;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_1_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_1_tensor_values.data(), input_1_tensor_size, input_1_node_dims.data(), 2);
  Ort::Value input_2_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_2_tensor_values.data(), input_2_tensor_size, input_2_node_dims.data(), 2);
  Ort::Value input_3_tensor = Ort::Value::CreateTensor<ttype>(memory_info, input_3_tensor_values.data(), input_3_tensor_size, input_3_node_dims.data(), 2);
  Ort::Value input_tensors[3] = {std::move(input_1_tensor), std::move(input_2_tensor), std::move(input_3_tensor)};
  session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors, 3, output_node_names.data(), 2);
}

SLTBENCH_FUNCTION_WITH_FIXTURE(bert_benchmark, BertFixture)

SLTBENCH_MAIN()
