#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void linear_forward_kernel(float* input, float* weights, float* bias, float* output, int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weights[col * in_features + k];
        }
        output[row * out_features + col] = sum + bias[col];
    }
}

// Backward pass kernel for gradients
__global__ void linear_backward_kernel(float* grad_output, float* input, float* weights, 
                                       float* grad_input, float* grad_weights, float* grad_bias, 
                                       int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradient w.r.t. input (grad_input = grad_output * W)
    if (row < batch_size && col < in_features) {
        float sum = 0.0f;
        for (int j = 0; j < out_features; j++) {
            sum += grad_output[row * out_features + j] * weights[j * in_features + col];
        }
        grad_input[row * in_features + col] = sum;
    }

    // Compute gradient w.r.t. weights (grad_weights = grad_output^T * input)
    if (row < out_features && col < in_features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += grad_output[i * out_features + row] * input[i * in_features + col];
        }
        grad_weights[row * in_features + col] = sum;
    }

    // Compute gradient w.r.t. bias (grad_bias = sum over grad_output)
    if (col < out_features && row == 0) {  // Only 1 thread per column accumulates
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += grad_output[i * out_features + col];
        }
        grad_bias[col] = sum;
    }
}


__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float max_val = -1e20f;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, input[row * cols + i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum_exp += expf(input[row * cols + i] - max_val);
    }

    for (int i = 0; i < cols; i++) {
        output[row * cols + i] = expf(input[row * cols + i] - max_val) / sum_exp;
    }
}

__global__ void softmax_backward_kernel(float* grad_output, float* softmax_output, float* grad_input, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    for (int i = 0; i < cols; i++) {
        float si = softmax_output[row * cols + i];
        float grad_sum = 0.0f;

        for (int j = 0; j < cols; j++) {
            float sj = softmax_output[row * cols + j];
            float delta_ij = (i == j) ? 1.0f : 0.0f;
            grad_sum += (delta_ij - sj) * grad_output[row * cols + j];
        }

        grad_input[row * cols + i] = si * grad_sum;
    }
}

torch::Tensor linear_forward_cuda(torch::Tensor input, torch::Tensor weights, torch::Tensor bias) {
    auto output = torch::zeros({input.size(0), weights.size(0)}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((weights.size(0) + TILE_SIZE - 1) / TILE_SIZE, (input.size(0) + TILE_SIZE - 1) / TILE_SIZE);

    linear_forward_kernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), input.size(0), input.size(1), weights.size(0)
    );

    return output;
}

std::vector<torch::Tensor> linear_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weights) {
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_bias = torch::zeros({weights.size(0)}, torch::TensorOptions().dtype(weights.dtype()).device(weights.device()));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocksInput((input.size(1) + TILE_SIZE - 1) / TILE_SIZE, (input.size(0) + TILE_SIZE - 1) / TILE_SIZE);
    dim3 numBlocksWeight((weights.size(1) + TILE_SIZE - 1) / TILE_SIZE, (weights.size(0) + TILE_SIZE - 1) / TILE_SIZE);

    linear_backward_kernel<<<numBlocksInput, threadsPerBlock>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(), weights.data_ptr<float>(), 
        grad_input.data_ptr<float>(), grad_weights.data_ptr<float>(), grad_bias.data_ptr<float>(), 
        input.size(0), input.size(1), weights.size(0)
    );

    return {grad_input, grad_weights, grad_bias};
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input, torch::device(input.device()));
    int threads = 1024;
    int blocks = (input.numel() + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}

torch::Tensor softmax_backward_cuda(torch::Tensor grad_output, torch::Tensor softmax_output) {
    auto grad_input = torch::zeros_like(grad_output);
    int rows = grad_output.size(0);
    int cols = grad_output.size(1);
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_backward_kernel<<<blocks, threads>>>(grad_output.data_ptr<float>(), softmax_output.data_ptr<float>(), grad_input.data_ptr<float>(), rows, cols);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu_cuda, "Custom ReLU function");
    m.def("softmax", &softmax_cuda, "Custom Softmax function");
    m.def("softmax_backward", &softmax_backward_cuda, "Custom Softmax backward function");
    m.def("linear_forward", &linear_forward_cuda, "Custom Linear Forward");
    m.def("linear_backward", &linear_backward_cuda, "Custom Linear Backward");
}