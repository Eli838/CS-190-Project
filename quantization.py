import torch
import torch.nn as nn
import math


def quantize_rowwise(x: torch.Tensor):
    abso = torch.abs(x)
    output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
    output = x[0]  / output_maxs[0,None]

    return output, output_maxs

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = x * state_x
    return output


fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)


print("   unquantized")
print(fp16_model.state_dict()['0.weight'][0])
#print(bnb.triton.quantize_rowwise.quantize_rowwise)
testing, testmax = quantize_rowwise(fp16_model.state_dict()['0.weight'])

print("quantized")
print(testing[0])

print("dequantized")
print(dequantize_rowwise(testing,testmax)[0])
#int8_model.load_state_dict(fp16_model.state_dict())
#print(torch.cuda.is_available())
#int8_model = int8_model.to(0) # Quantization happens here
#print(int8_model.state_dict()['0.weight'][0])