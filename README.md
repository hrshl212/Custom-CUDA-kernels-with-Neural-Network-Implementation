# Custom-CUDA-kernels-with-Neural-Network-Implementation
The repository contains custom CUDA kernels for linear layer, softmax and relu which are integrated with python to develop a Neural Network

simpleNN_wt_CUDA.py contains the neural network implementation which call for custom CUDA kernels for linear layer, relu and softmax function. Custom autograd needs to be implemented for these layers for accurate gradient calculation during backpropagation (implemented in custom_autograd.py). This again makes use of custom kernels for its implementation wrapped in PyTorch `autograd.function`. 

simplenn.py includes the neural network with pytorch implementation which uses the pytorch's in-built cuda kernal implementation.
