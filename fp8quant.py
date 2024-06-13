import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import datetime
# from matplotlib import pyplot as plt
import numpy as np
# from tqdm import tqdm
import torchvision.models as models
import numpy as np
import struct
import torch.nn.init as init

output_path = 'fp8'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model():
    vgg16 = models.vgg16(pretrained=False)
    model_path = 'finetuned_vgg16_9'
    vgg16.classifier[4] = nn.Linear(4096,1024)
    vgg16.classifier[6] = nn.Linear(1024,10)
    vgg16.load_state_dict(torch.load(model_path, map_location=device))
    vgg16.to(device)
    vgg16.eval()
    return vgg16

            
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

#the input is normalized tensor x,
def round_dt8(x, exp = 4, num_bits = 8):
  val = copy.deepcopy(x)

  sign_bit = 0 if val >= 0 else 1
  val = abs(val)
  exp_bits = 0
  while val < 0.1:
      val *= 10
      exp_bits += 1

  bs_bits = max(0, num_bits - 2 - exp_bits)
  exp_bits = min(num_bits -1, exp_bits)
  EXP_COUNT[exp_bits] += 1

  if exp_bits == 0:
      quantized_val = bisection_quantization(val, 7)
  elif exp_bits >= num_bits - 2:
      quantized_val = 0.0
  else:
      quantized_val = bisection_quantization(val, bs_bits)

  quantized_val = quantized_val if sign_bit == 0 else -quantized_val
  quantized_val *= 10**(-exp_bits)
  return quantized_val



def quantize_rowwise(x: torch.Tensor, dt = False):
  '''Takes in a (2d) tensor and returns quantized array
  DT = if you use the dynamic tree quantization. False is using fp8
  tensor.view( shape[0],-1) can reshape a 3d tensor to 2d. that should work'''
  abso = torch.abs(x)
  output_maxs  = torch.max(abso,1)[0].unsqueeze(-1)
  output = x  / output_maxs[None,:]
  if not dt:
      output.apply_(round_fp8)
  else:
      def dt_wrapper(x: torch.Tensor):
        if bits < 4:
            exp_bits = bits - 1
        else:
            exp_bits = 8
        return round_dt8(x, num_bits = bits)
      output.apply_(dt_wrapper)
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
  '''
  if (x.numel() % batch_size != 0):
    print("Invalid batch size. Batch size should be a divisor of " + str(x.numel()))
    return

  flatarg = torch.argsort(torch.abs(x.flatten()))
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
def quantize_dequantize_fp8(mat):
    testing_dt, dt_max = quantize_rowwise(mat, dt = False)
    return dequantize_rowwise(testing_dt,dt_max)


vgg16 = create_model()
count = 0
curr_path = output_path  
# curr_path.mkdir(parents=True, exist_ok=True) 

for layer in [*vgg16.features,*vgg16.classifier]:
    count += 1 
    curr_layer_path = curr_path 
    try:
        if len(layer.weight.shape) == 4:
            weights = layer.weight.detach()
            print(f'Layer {count}')# weights shape pre-quantization: {weights.shape}\nWeights: {weights}')
            for filter in range(0, weights.shape[0]):
                # print(f'Filter num {filter}')
                for channel in range(0, weights.shape[1]):
                    # print(f'Channel num {channel}')
                    # print(layer.weight[filter,channel])
                    weights[filter,channel] = quantize_dequantize_fp8(weights[filter,channel])
                    # for row in range(0,weights.shape[2]):
                    #     weights[filter,channel, row] = quantize_dequantize_dt(weights[filter,channel,row])
                    # print(f'Finish window')
            # print(f'Layer {count} weights shape post-quantization: {weights.shape}\nWeights: {weights}')
            # layer.weight = nn.parameter.Parameter(weights)
            print(f'Layer {count} weights shape post-quantization: {weights.shape}\nWeights: {weights}')
            layer.weight = nn.parameter.Parameter(weights)
        else:
            weights = layer.weight.detach()
            print(f'Layer {count}')# weights shape pre-quantization: {layer.weight.shape}\nWeights: {weights}')
            weights = quantize_dequantize_fp8(weights)
            # for row in tqdm(range(0,weights.shape[0])):
            #     weights[row] = quantize_dequantize_dt(weights[row])
            layer.weight = nn.parameter.Parameter(weights)
            # print(f'Layer {count} weights shape post-quantization: {layer.weight.shape}\nWeights: {weights}')
            # print(layer.weight)
    except (TypeError, AttributeError):
        pass

model_path = 'fp8_quantized_model'
# count_path = output_path / f'dt_counts_bl_{bit_length}.pkl'
torch.save(vgg16.state_dict(), model_path)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    