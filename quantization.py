
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
      pass
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
  '''Takes in a (2d) tensor and returns quantized array
  DT = if you use the dynamic tree quantization. False is using fp8
  tensor.view( -1,shape[1]) can reshape a 3d tensor to 2d. that should work'''
  abso = torch.abs(x)
  output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
  output = x  / output_maxs[None,:]
  if not dt:
      output.apply_(round_fp8)
  else:
      output.apply_(round_dt8)
  return torch.squeeze(output), output_maxs

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
  '''Dequantizes the tensor given the maxes'''
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


def quantize_stable_embedding(x, batch_size, dt = False):
  '''Qunatizes the given array
  Batch size must be a divisor of the array size
  Returns the quantized array, the maximums, and the indexes
  DT  = false, means using fp8
  Dt = true, means using the dynamic tree
  ''''
  if (x.numel() % batch_size != 0):
    print("Invalid batch size. Batch size should be a divisor of " + str(x.numel()))
    return

  flatarg = torch.argsort(x.flatten())
  indexing = flatarg.reshape((x.numel()//batch_size,batch_size))

  reshapedx = x.flatten()[indexing]

  output, maxes = quantize_rowwise(reshapedx,dt)

  return output.reshape(x.shape), maxes, indexing

def dequantize_stable_embedding(input, maxes, indexing):
  '''Takes the quantized matrices and dequantizes them by multiplying the normalized and maxes
  Then uses the indexes to place the dequantized values back into their original spots
  returns dequantized values in the right positions
  Takes in the quantized array, the array maximums, and the indexes'''
  outreshape = input.reshape(indexing.shape)

  dequant = dequantize_rowwise(outreshape, maxes).flatten()
  dequant[indexing.flatten()] = dequant.clone()
  return dequant.reshape(input.shape)
