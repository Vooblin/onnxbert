from transformers import *
import torch

config = BertConfig.from_json_file('./bert/bert-large-uncased-config.json')
model_path = "./bert/"
model = BertModel(config)
model.eval()
device = torch.device("cpu")
model.to(device)
dummy_input0 = torch.LongTensor(1, 128).fill_(1).to(device)
dummy_input1 = torch.LongTensor(1, 128).fill_(1).to(device)
dummy_input2 = torch.LongTensor(1, 128).fill_(0).to(device)
dummy_input = (dummy_input0, dummy_input1, dummy_input2)
output_path = "./bert/torch.onnx"
torch.onnx.export(model,
                  dummy_input,
                  output_path,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ["input_ids", "input_mask", "segment_ids"],
                  output_names = ["output"],
                  dynamic_axes = {'input_ids' : {0 : 'batch_size'},
                                  'input_mask' : {0 : 'batch_size'},
                                  'segment_ids' : {0 : 'batch_size'},
                                  'output': {0 : 'batch_size'}})
