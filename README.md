# Models
## AlexNet
For these notebooks, they should each have their own folder. They will expect a data folder to be present in the parent directory. If it is not available, one will be created with the CIFAR-10 data. This allows for the data to be shared. It will also typically create output directories within their current folder which is why it is important to give them each their own directory.

### Fine Tuning
File: `Frozen-Alexnet.ipynb`

This file loads AlexNet, replaces the classifier, freezes the feature extraction layers, and fine-tunes the model. It also evaluates the model on test data and exports its state dictionary in a file called `model_[CURRENT TIMESTAMP]_final_frozen_alexnet`. Elijah has the fine-tuned AlexNet stored locally, along with many of the other quantized AlexNets.
### Linear Quantization
File: `LinearQuantTest.ipynb`


This file loads a fine-tuned AlexNet from state dictionary. You need to either paste the state dictionary file into its current directory and change the file name for the updated timestamp or give it an entirely different path. 

#### Quantization Portion
This applies the linear scale quantization detailed in our report by quantizing a weight tensor, dequantizing it, and injecting it back into the model.

The quantization function is `quantize_requantize` which takes an N-D tensor, torch datatype corresponding to the values stored in the tensor, and the torch datatype you want to quantize to. This uses pytorch's functions to find the min and max of these.


The notebook contains two approaches for applying the quantization: Per Tensor linear scale quantization and Row-wise linear scale quantization.
We **only** used per tensor linear scale quantization. The notebook has code to apply the quantization, make the model use a GPU, and then test the model. Side note: the cell that applies the quantization has a try and except statement to skip over layers that do not have weights.


This file also has the beginnings of block-wise linear quantization commented out which achieved low accuracy. It could possibly be resolved by significantly increasing the accuracy of the scale factor (a challenge we had with the row-wise linear quantization) if someone wanted to use it for future work. It should take an N-D tensor as input and will quantize 2D slices over the last 2 axes of the shape.

#### Weight Computation Portion
We estimated the size of AlexNet's weights if they were quantized to a specific datatype by determining the number of elements in AlexNet's weights and multiplying them by the number of bits said datatype uses to get the size of the weights in bits. There is also a function (`bits_to_mb`) to convert these bits to megabytes.

### FP8 Quantization
File: `FP8Quant.ipynb`

This notebook is similar to the previous one, the main difference is that the quantization cell is modified so it applies the row-wise FP8 to only 2D sub-tensors of the passed in weight tensor. the 2D tensors are from the last 2 dimensions in the tensor's shape where these are used as the length and width. The size of the AlexNet weights were not calculated here since both int8 and fp8 are 8 bits.


### Dynamic Tree Quantization

#### Quantization Portion
File: `AutomatedQuantization.ipynb`

This notebook applies dynamic tree quantization to bit sizes [8,7,6,5,4,3]. Each of these models is saved to an output folder in the current directory as a state dictionary. Their naming convention is `dt_quantized_model_bl_[BIT_LENGTH]`. It also saves the counts for dynamic tree exponents used per model in the same folder. Their naming convention is `dt_counts_bl_[BIT_LENGTH].pkl` If this notebook has already been run, running it again may overwrite previous saved models or counts. The dynamic tree quantization function needed a wrapper function (`dt_wrapper`) so it could be passed into `apply_` in `quantized_rowwise`.

File: `AutoQuantTest.ipynb`

This file loads models from `AutomatedQuantization.ipynb` that correspond the the bit\_lengths in the bit\_lengths list at the top of the notebook. It will evaluate each of these models on the CIFAR-10 test set and output its accuracy. It also produces these results as a markdown table.

This file also includes code to plot the accuracy of AlexNet vs the number of dynamic tree bits. Futhermore, this notebook loads the exponent count files saved by the previous notebook for every bit\_length in the bit\_lengths list. This data is also output as a markdown table along with a markdown table that shows the percentages of weights for each dynamic tree exponent on AlexNet.

#### Weight Comparison Portion
File: `AutoQuantTest.ipynb`

The latter part of the `AutoQuantTest.ipynb` notebook estimates the size of these loaded AlexNet models. When passing a model into `compute_model_size`, you pass in both an AlexNet model and the number of bits you want to compute the size for. This counts the number of elements in AlexNet, keeping a count of weights and count of maximums needed for dynamic tree quantization. We then multiply the number of weights by the number of bits passed in to get the size of those weights. We then multiply the number of maximums required and multiply that by 32 since the maximums are stored as FP32. Both of these values are returned in MB along with the sum of these values in MB. This code also produces a graph for weight sizes and maximums size vs number of dynamic tree bits. Furthermore, this notebook computes MSE and MAE between loaded models and the original, fine-tuned alexnet. There is also code to generate a markdown table of sizes and MSEs.

### Stable Embedding
File: `StableEmbeddingRun.ipynb`

This notebook loads a fine-tuned AlexNet supplied by the user, applies stable embedding quantization, and evaluates its accuracy on CIFAR-10.
