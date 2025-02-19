# Custom-CUDA-kernels-with-Neural-Network-Implementation
The repository contains custom CUDA kernels for linear layer, softmax and relu which are integrated with python to develop a Neural Network

`simpleNN_wt_CUDA.py` contains the neural network implementation which call for custom CUDA kernels for linear layer, relu and softmax function. Custom autograd needs to be implemented for these layers for accurate gradient calculation during backpropagation (implemented in custom_autograd.py). This again makes use of custom kernels for its implementation wrapped in PyTorch `autograd.function`. This method ensures the custom CUDA Softmax function is fully differentiable and can be used in any PyTorch model while benefiting from GPU acceleration. 

Considering two linear layers are involved and we want to track the weights and bias for each layer, I have implemented the custom linear layer as PyTorch `nn.Module`. Because of this, each CustomLinear instance maintains separate weights and biases and the PyTorch autograd engine correctly tracks gradients for each layer.

`simplenn.py` includes the neural network with pytorch implementation which uses the pytorch's in-built cuda kernal implementation. This is included in the repository to compare PyTorch in-built implementation with my custom cuda kernels implementation.

The PyTorch implementation takes a time of 0.35 sec whereas my implementation takes a time of 0.37 sec which is comparable in performance
