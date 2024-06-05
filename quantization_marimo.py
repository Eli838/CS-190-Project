import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    EXP_COUNT = {0:0,
                 1:0,
                 2:0,
                 3:0,
                 4:0,
                 5:0,
                 6:0,
                 7:0,
                }
    return EXP_COUNT, mo


@app.cell
def __(EXP_COUNT):
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
    def bisection_quantization(num, bits = 7):
        if bits == 0:
            return 0.1
        val = abs(num)
        inversed_bits = []
        # Bisection tree quantization
        range_min, range_max = 0, 1
        for p in range(bits):
            p += 1
            bit_val = 2**(-p)
            if val >= bit_val:
                inversed_bits.append(1)
                val -= bit_val
            else:
                inversed_bits.append(0)

        quantized_val = 0
        for k, bit in enumerate(inversed_bits):
            if bit:
                quantized_val += 2**-(k + 1)
        return quantized_val


    #unsigned bisection quantization
    #in-progress
    def bisection_quantization_unsigned(num, bits=7):
        val = num     #no absolute value since unsigned
        inversed_bits = []
        #bisection tree quantization
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
                quantized_val += 2**-(k+1)
        return quantized_val


    #the input is normalized tensor x,
    def round_dt8(x, exp = 4, num_bits = 8):
        x_dt8 = x.clone().to(torch.float32)
        output = torch.zeros_like(x_dt8)

        for i, row in enumerate(x_dt8):
            for j, val in enumerate(row):
                sign_bit = 0 if val >= 0 else 1
                val = abs(val)
                exp_bits = 0
                while val < 0.1:
                    val *= 10
                    exp_bits += 1

                bs_bits = max(0, num_bits - 2 - exp_bits)
                exp_bits = min(7, exp_bits)
                EXP_COUNT[exp_bits] += 1

                if exp_bits == 0:
                    quantized_val = bisection_quantization(val, num_bits - 1)
                elif exp_bits >= num_bits - 2:
                    quantized_val = 0.1
                else:
                    quantized_val = bisection_quantization(val, bs_bits)

                quantized_val = quantized_val if sign_bit == 0 else -quantized_val
                quantized_val *= 10**(-exp_bits)
                output[i, j] = quantized_val

        return output


    #round_dt8_unsigned
    #the input is normalized tensor x
    #in-progress
    def round_dt8_unsigned(x, exp=4):
        val = x.clone().to(torch.float32)
        for i in range(len(val)):
            for j in range(len(val[i])):
                abs_val = val[i][j]
                exp_bits = 0
                while abs_val < 0.1:
                    abs_val *= 10
                    exp_bits += 1
                bs_bits = max(0, 6 - exp_bits)
                exp_bits = min(7, exp_bits)
                if exp_bits == 0:
                    quantized_val = bisection_quantization_unsigned(abs_val, 7)
                elif exp_bits >= 6:
                    quantized_val = abs_val
                else:
                    quantized_val = bisection_quantization_unsigned(abs_val, bs_bits)
                quantized_val *= 10**(-exp_bits)
                val[i][j] = quantized_val
        return val


    def quantize_rowwise(x: torch.Tensor, dt = False, num_bits =8):
        abso = torch.abs(x)
        output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
        output = x[0]  / output_maxs[0,None]
        if not dt:
            output = round_fp8(output)
        else:
            output = round_dt8(output, num_bits = num_bits)
        return output, output_maxs

    def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
        output = x * state_x
        return output

    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                init.zeros_(m.bias)

    def measure_quantization_error(original_tensor, dequantized_tensor):
        abs_error = torch.abs(original_tensor - dequantized_tensor)
        return torch.mean(abs_error), abs_error
    return (
        binary,
        bisection_quantization,
        bisection_quantization_unsigned,
        calc_exp,
        calc_mantissa,
        copy,
        dequantize_rowwise,
        init,
        init_weights,
        math,
        measure_quantization_error,
        nn,
        np,
        quantize_rowwise,
        random,
        round_dt8,
        round_dt8_unsigned,
        round_fp8,
        struct,
        torch,
    )


@app.cell
def __(EXP_COUNT, dequantize_rowwise, init_weights, nn, quantize_rowwise):
    ## usage example
    MAE = nn.L1Loss() 
    num_bits = range(3,9)
    for num in num_bits:
        print("#####################", num)
        fp16_model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64)
        )

        fp16_model.apply(init_weights)

        # print("   unquantized")
        # print(fp16_model.state_dict()['0.weight'][0])
        # #print(bnb.triton.quantize_rowwise.quantize_rowwise)
        testing, testmax = quantize_rowwise(fp16_model.state_dict()['0.weight'])
        # print("quantized")
        # print(testing[0])
        # print("dequantized")
        # print(dequantize_rowwise(testing,testmax)[0])

        testing_dt, dt_max = quantize_rowwise(fp16_model.state_dict()['0.weight'], dt = True, num_bits = num)
        print("   unquantized")
        print(fp16_model.state_dict()['0.weight'][0])
        print("quantized_dt")
        print(testing_dt[0])
        print("dequantized")
        print(dequantize_rowwise(testing_dt,dt_max)[0])

        fp8_mae = MAE(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing,testmax)[0])
        print("fp8 mean abs error: ", fp8_mae)

        dt8_mae = MAE(fp16_model.state_dict()['0.weight'], dequantize_rowwise(testing_dt,dt_max)[0])
        print("dt",num," mean abs error: ", dt8_mae)

        print("exp_count: ", EXP_COUNT)
        print("#####################")

        for k in EXP_COUNT.keys():
            EXP_COUNT[k] = 0
    return (
        MAE,
        dt8_mae,
        dt_max,
        fp16_model,
        fp8_mae,
        k,
        num,
        num_bits,
        testing,
        testing_dt,
        testmax,
    )


@app.cell
def __(MAE, fp16_model, np, num_bits, quantize_rowwise):
    for n in num_bits:
        maes = []
        for r_cnt, row in enumerate(fp16_model.state_dict()['0.weight']):
            o, mx = quantize_rowwise(row.view(1,64), dt = True, num_bits=n)
            row_mae = MAE(o, row.view(1,64))
            maes.append(row_mae)
        print('bitsize: ', n, 'MAE: ', np.mean(maes), 'row MAEs: ', maes)
        
    return maes, mx, n, o, r_cnt, row, row_mae


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
