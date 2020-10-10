import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--size', choices=['tiny', 'base', 'large'], default='tiny', help='Size of BERT model')
args = parser.parse_args()

import torch
from transformers import BertConfig, BertModel

if args.size == 'tiny':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    bert_name_or_path = os.path.join(os.path.join(cur_dir, 'bert'), 'bert-tiny-uncased-config.json')
elif args.size == 'base':
    bert_name_or_path = "bert-base-uncased"
else:
    bert_name_or_path = "bert-large-uncased"

config = BertConfig.from_pretrained(bert_name_or_path)
model = BertModel(config)
model.eval()
device = torch.device("cpu")
model.to(device)
dummy_input0 = torch.LongTensor(1, 512).fill_(1).to(device)
dummy_input1 = torch.LongTensor(1, 512).fill_(1).to(device)
dummy_input2 = torch.LongTensor(1, 512).fill_(0).to(device)
dummy_input = (dummy_input0, dummy_input1, dummy_input2)
output_path = './bert/bert_{}.onnx'.format(args.size)
torch.onnx.export(model,
                  dummy_input,
                  output_path,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names = ["input_ids", "input_mask", "segment_ids"],
                  output_names = ["output"],
                  dynamic_axes = {'input_ids' : {0 : 'batch_size'},
                                  'input_mask' : {0 : 'batch_size'},
                                  'segment_ids' : {0 : 'batch_size'},
                                  'output': {0 : 'batch_size'}})
