import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import struct

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# input : tensor output: binary string
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

#input: mantissa bitstring
#output: float value
def calc_mantissa(mantissa):
    res = 0
    for k in range(len(mantissa)):
        if mantissa[k] == '1':
          res += 2**(-k-1)
    return res

#input exp: bitstring
# new_exp_len: new length of exp
def calc_exp(exp, new_exp_len):
    limit = 2**(new_exp_len) - 1
    # if exp is more than new_exp_len limit, truncate to new_exp_len limit.
    bias = 2**(len(exp)-1) - 1
    val = int(exp,2) - bias
    if val > limit:
        return limit
    if val < -limit + 1:
        return -limit + 1

    return val

def round_fp8(x, exp = 4):
  '''
  Quantizes input tensor to FP8 data format
  inputs  x:      original tensor
          exp:    number of bits used for exponent field
                  e.g. E5M2 has 5 exp bits, E4M3 has 4 exp bits
  output  x_32:   quantized tensor
  '''

  x_fp8 = copy.deepcopy(x)


  result = 1.0
  bin_str = binary(x)

  bin_mantissa = bin_str[9:32]
  res_mantissa = bin_mantissa[:7-exp]    
  result += calc_mantissa(res_mantissa)

  bin_exp = bin_str[1:9]
  exp_int = calc_exp(bin_exp, exp)
  result *= 2**exp_int

  if bin_str[0] == '1':
    result *= -1
  
  return result

def quantize_rowwise(x: torch.Tensor):
    abso = torch.abs(x)
    output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
    output = x / output_maxs[0,None]
    output.apply_(round_fp8)
    return output, output_maxs

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = x * state_x
    return output


fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)


print("   unquantized")
print(fp16_model.state_dict()['0.weight'].shape)
#print(bnb.triton.quantize_rowwise.quantize_rowwise)
testing, testmax = quantize_rowwise(fp16_model.state_dict()['0.weight'])

print("quantized")
print(testing.shape)
print(testing[0])
print(testing[1])

print("dequantized")
print(dequantize_rowwise(testing,testmax).shape)
#int8_model.load_state_dict(fp16_model.state_dict())
#print(torch.cuda.is_available())
#int8_model = int8_model.to(0) # Quantization happens here
#print(int8_model.state_dict()['0.weight'][0])
