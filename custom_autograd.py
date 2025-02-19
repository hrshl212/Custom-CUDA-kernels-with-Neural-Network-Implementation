import torch
import custom_kernels
from torch.autograd import Function

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Save input for backward computation
        return custom_kernels.relu(input)  # Call your CUDA kernel

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors  # Retrieve saved input
        grad_input = grad_output.clone()  
        grad_input[input < 0] = 0  # ReLU derivative: 1 for x > 0, 0 for x <= 0
        return grad_input

# Define a wrapper function for easy use
def custom_relu(x):
    return CustomReLU.apply(x)

class CustomSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, input):
        softmax_output = custom_kernels.softmax(input)  # Use CUDA softmax kernel
        ctx.save_for_backward(softmax_output)  # Save for backward pass
        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        grad_input = custom_kernels.softmax_backward(grad_output, softmax_output)  # Use CUDA backward kernel
        return grad_input

# Wrapper function to use in models
def custom_softmax(input):
    return CustomSoftmaxFunction.apply(input)


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        output = custom_kernels.linear_forward(input, weights, bias)
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        grad_input, grad_weights, grad_bias = custom_kernels.linear_backward(grad_output, input, weights)
        return grad_input, grad_weights, grad_bias

class CustomLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32, device="cuda") * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32, device="cuda"))

    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weights, self.bias)
