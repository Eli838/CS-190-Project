import torch
import torch.nn as nn
import math

import bitsandbytes as bnb
#from bnb.nn import Linear8bitLt

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)

int8_model = nn.Sequential(
    bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False),
    bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)
)

print("unquantized")
print(fp16_model.state_dict()['0.weight'][0])
#print(bnb.triton.quantize_rowwise.quantize_rowwise)
testing, testmax = bnb.triton.quantize_rowwise.quantize_rowwise(fp16_model.state_dict()['0.weight'][0])

print("quantized")
print(testing[0])

print("dequantized")
print(bnb.triton.dequantize_rowwise.dequantize_rowwise(testing,testmax)[0])
#int8_model.load_state_dict(fp16_model.state_dict())
#print(torch.cuda.is_available())
#int8_model = int8_model.to(0) # Quantization happens here
#print(int8_model.state_dict()['0.weight'][0])