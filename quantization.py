

import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import struct
import torch.nn.init as init

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


def bisection_quantization(num, bits = 7):
  val = abs(num)
  inversed_bits = []
  # Bisection tree quantization
  range_min, range_max = 0, 1
  for _ in range(bits):
      mid = (range_min + range_max) / 2
      if val >= mid:
          inversed_bits.append(1)
          range_min = mid
      else:
          inversed_bits.append(0)
          range_max = mid

  quantized_val = 0
  for k, bit in enumerate(inversed_bits):
      if bit:
          quantized_val += 2**-(k + 1)

  return quantized_val

#the input is normalized tensor x,
def round_dt8(x, exp = 4):
  val = copy.deepcopy(x)
  num_levels = 2 ** (7)


  sign_bit = 0 if val >= 0 else 1
  val = abs(val)
  exp_bits = 0
  while val < 0.1:
      val *= 10
      exp_bits += 1

  bs_bits = max(0, 6 - exp_bits)
  exp_bits = min(7, exp_bits)

  if exp_bits == 0:
      quantized_val = bisection_quantization(val, 7)
  elif exp_bits >= 6:
      quantized_val = val
  else:
      quantized_val = bisection_quantization(val, bs_bits)

  quantized_val = quantized_val if sign_bit == 0 else -quantized_val
  quantized_val *= 10**(-exp_bits)
  return quantized_val


def quantize_rowwise(x: torch.Tensor, dt = False):
  abso = torch.abs(x)
  output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
  output = x  / output_maxs[0,None]
  if not dt:
      output.apply_(round_fp8)
  else:
      output.apply_(round_dt8)
  return output, output_maxs

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
  output = x * state_x
  return output

def init_weights(m, in_std = 1.0):
  if isinstance(m, nn.Linear):
      init.normal_(m.weight, mean=0.0, std=in_std)
      if m.bias is not None:
          init.zeros_(m.bias)

def measure_quantization_error(original_tensor, dequantized_tensor):
  abs_error = torch.abs(original_tensor - dequantized_tensor)
  return torch.mean(abs_error), abs_error


sigmas = [1,10,100]
for sig in sigmas:
  print('===================')
  print('current sigma: ', sig)
  fp16_model = nn.Sequential(
      nn.Linear(64, 64),
      nn.Linear(64, 64)
  )

  fp16_model.apply(lambda m: init_weights(m, sig))

  print("   unquantized")
  print(fp16_model.state_dict()['0.weight'][0])
  #print(bnb.triton.quantize_rowwise.quantize_rowwise)
  testing, testmax = quantize_rowwise(fp16_model.state_dict()['0.weight'])

  print("quantized")
  print(testing[0])

  print("dequantized")
  print(dequantize_rowwise(testing,testmax)[0])

  testing_dt, dt_max = quantize_rowwise(fp16_model.state_dict()['0.weight'], dt = True)

  print("   unquantized")
  print(fp16_model.state_dict()['0.weight'][0])
  print("quantized_dt")
  print(testing_dt[0])
  print("dequantized")
  print(dequantize_rowwise(testing_dt,dt_max)[0])

  fp8_mae, fp8_err = measure_quantization_error(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing,testmax)[0])
  print("fp8 mean abs error: ", fp8_mae)

  dt8_mae, dt8_err = measure_quantization_error(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing_dt,testmax)[0])
  print("dt8 mean abs error: ", dt8_mae)
