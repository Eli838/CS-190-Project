import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
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

      x_fp8 = x.clone().to(torch.float32)

      for i in range(len(x_fp8)):
        for j in range(len(x_fp8[i])):
          result = 1.0
          bin_str = binary(x[i][j])

          bin_mantissa = bin_str[9:32]
          res_mantissa = bin_mantissa[:7-exp]    
          result += calc_mantissa(res_mantissa)

          bin_exp = bin_str[1:9]
          exp_int = calc_exp(bin_exp, exp)
          result *= 2**exp_int

          if bin_str[0] == '1':
            result *= -1

          x_fp8[i][j] = result    

      return x_fp8.to(torch.float32)

    #the input is normalized tensor x,
    def round_dt8(x, exp = 4):
        x_dt8 = x.clone().to(torch.float32)
        output = torch.zeros_like(x_dt8)
        num_levels = 2 ** (7)
        
        for i, row in enumerate(x_dt8):
            for j, val in enumerate(row):
                sign_bit = 0 if val >= 0 else 1
                inversed_bits = []
                val = abs(val)
                
                # Bisection tree quantization
                range_min, range_max = 0, 1
                for _ in range(7):
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
                        
                quantized_val = quantized_val if sign_bit == 0 else -quantized_val
                output[i, j] = quantized_val

        return output

    def quantize_rowwise(x: torch.Tensor, dt = False):
        abso = torch.abs(x)
        output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
        output = x[0]  / output_maxs[0,None]
        if not dt:
            output = round_fp8(output)
        else:
            output = round_dt8(output)
        return output, output_maxs

    def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
        output = x * state_x
        return output

    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                init.zeros_(m.bias)
    return (
        binary,
        calc_exp,
        calc_mantissa,
        copy,
        dequantize_rowwise,
        init,
        init_weights,
        math,
        nn,
        np,
        quantize_rowwise,
        random,
        round_dt8,
        round_fp8,
        struct,
        torch,
    )


@app.cell
def __(dequantize_rowwise, init_weights, nn, quantize_rowwise):
    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    fp16_model.apply(init_weights)

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
    return fp16_model, testing, testmax


@app.cell
def __(dequantize_rowwise, fp16_model, quantize_rowwise):
    testing_dt, dt_max = quantize_rowwise(fp16_model.state_dict()['0.weight'], dt = True)

    print("   unquantized")
    print(fp16_model.state_dict()['0.weight'][0])
    print("quantized_dt")
    print(testing_dt[0])
    print("dequantized")
    print(dequantize_rowwise(testing_dt,dt_max)[0])
    return dt_max, testing_dt


@app.cell
def __(torch):
    def measure_quantization_error(original_tensor, dequantized_tensor):
        """
        Measures the quantization error in terms of absolute errors.
        """
        abs_error = torch.abs(original_tensor - dequantized_tensor)

        return torch.mean(abs_error), abs_error
    return measure_quantization_error,


@app.cell
def __(
    dequantize_rowwise,
    fp16_model,
    measure_quantization_error,
    testing,
    testmax,
):
    fp8_mae, fp8_err = measure_quantization_error(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing,testmax)[0])
    print("fp8 mean abs error: ", fp8_mae)
    return fp8_err, fp8_mae


@app.cell
def __(
    dequantize_rowwise,
    fp16_model,
    measure_quantization_error,
    testing_dt,
    testmax,
):
    dt8_mae, dt8_err = measure_quantization_error(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing_dt,testmax)[0])
    print("dt8 mean abs error: ", dt8_mae)
    return dt8_err, dt8_mae


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
