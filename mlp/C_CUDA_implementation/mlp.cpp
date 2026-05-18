#include "mlp.h"
#include "kernels.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

MLP::MLP(int input, int hidden, int output, int layers)
    : input_dim(input), hidden_dim(hidden), output_dim(output),
      num_hidden_layers(layers) {

  layer_dims.push_back(input);
  for (int i = 0; i < layers; i++) {
    layer_dims.push_back(hidden);
  }
  layer_dims.push_back(output);

  // Allocate pointers
  int num_weight_matrices = layer_dims.size() - 1;
  d_W.resize(num_weight_matrices);
  d_b.resize(num_weight_matrices);
  d_dW.resize(num_weight_matrices);
  d_db.resize(num_weight_matrices);
  d_A.resize(layer_dims.size());
  d_Z.resize(num_weight_matrices);
}

MLP::~MLP() {
  for (auto p : d_W)
    cudaFree(p);
  for (auto p : d_b)
    cudaFree(p);
  for (auto p : d_dW)
    cudaFree(p);
  for (auto p : d_db)
    cudaFree(p);
  for (auto p : d_A)
    cudaFree(p);
  for (auto p : d_Z)
    cudaFree(p);
}

void MLP::init_parameters() {
  for (size_t i = 0; i < d_W.size(); i++) {
    int in = layer_dims[i];
    int out = layer_dims[i + 1];

    // He Initialization
    float std_dev = sqrt(2.0f / in);
    int size_W = out * in;
    int size_b = out;

    float *h_W = (float *)malloc(size_W * sizeof(float));
    float *h_b = (float *)calloc(size_b, sizeof(float)); // Bias 0

    for (int j = 0; j < size_W; j++) {
      h_W[j] =
          ((float)rand() / RAND_MAX * 2.0f - 1.0f) * std_dev; // Approx Gaussian
    }

    cudaMalloc(&d_W[i], size_W * sizeof(float));
    cudaMemcpy(d_W[i], h_W, size_W * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_b[i], size_b * sizeof(float));
    cudaMemcpy(d_b[i], h_b, size_b * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate Gradients
    cudaMalloc(&d_dW[i], size_W * sizeof(float));
    cudaMalloc(&d_db[i], size_b * sizeof(float));

    free(h_W);
    free(h_b);
  }
}

void MLP::forward(const float *d_X, int batch_size) {

  int in_size = input_dim * batch_size;
  if (d_A[0] == nullptr)
    cudaMalloc(&d_A[0], in_size * sizeof(float));
  cudaMemcpy(d_A[0], d_X, in_size * sizeof(float), cudaMemcpyDeviceToDevice);

  for (size_t i = 0; i < d_W.size(); i++) {
    int n_in = layer_dims[i];
    int n_out = layer_dims[i + 1];

    // NOTE: In real app, realloc will be needed it batch size changes. just
    // assuming constant size for now
    if (d_Z[i] == nullptr)
      cudaMalloc(&d_Z[i], n_out * batch_size * sizeof(float));
    if (d_A[i + 1] == nullptr)
      cudaMalloc(&d_A[i + 1], n_out * batch_size * sizeof(float));

    // Z = W * A + b
    // W: (n_out, n_in), A: (n_in, batch) -> Z: (n_out, batch)
    gpu_matmul(d_W[i], d_A[i], d_Z[i], n_out, batch_size, n_in);
    gpu_add_bias(d_Z[i], d_b[i], n_out, batch_size);

    // activation
    if (i < d_W.size() - 1) {
      // ReLU
      gpu_relu(d_Z[i], d_A[i + 1], n_out * batch_size);
    } else {
      // Softmax
      gpu_softmax(d_Z[i], d_A[i + 1], n_out, batch_size);
    }
  }
}

void MLP::backward(const float *d_Y, int batch_size) {
  // d_Y is one-hot encoded labels (n_classes, batch_size)
  // Output layer error: dZ = A - Y
  int L = d_W.size() - 1;
  int n_out = layer_dims[L + 1];
  int size = n_out * batch_size;

  float *d_dZ;
  cudaMalloc(&d_dZ, size * sizeof(float));
  cudaMemcpy(d_dZ, d_A[L + 1], size * sizeof(float), cudaMemcpyDeviceToDevice);

  gpu_update_params(d_dZ, d_Y, 1.0f, size);

  // Backprop Loop
  for (int i = L; i >= 0; i--) {
    int n_in = layer_dims[i];
    int n_out = layer_dims[i + 1];

    // dW = (1/m) * dZ * A_prev^T
    // dZ: (n_out, batch), A_prev: (n_in, batch)
    // dZ * A_prev^T -> (n_out, n_in)

    float *d_A_prev_T;
    cudaMalloc(&d_A_prev_T, n_in * batch_size * sizeof(float));
    gpu_transpose(d_A[i], d_A_prev_T, n_in, batch_size);

    gpu_matmul(d_dZ, d_A_prev_T, d_dW[i], n_out, n_in, batch_size);

    // db = (1/m) * sum(dZ, axis=1)
    gpu_sum_cols(d_dZ, d_db[i], n_out, batch_size);

    cudaFree(d_A_prev_T);

    if (i > 0) {
      // dZ_prev = W^T * dZ * derv(Z_prev)
      // W: (n_out, n_in) -> W^T: (n_in, n_out)
      // dZ: (n_out, batch)
      // Result: (n_in, batch)

      float *d_W_T;
      cudaMalloc(&d_W_T, n_out * n_in * sizeof(float));
      gpu_transpose(d_W[i], d_W_T, n_out, n_in);

      float *d_dZ_prev;
      cudaMalloc(&d_dZ_prev, n_in * batch_size * sizeof(float));

      gpu_matmul(d_W_T, d_dZ, d_dZ_prev, n_in, batch_size, n_out);

      gpu_apply_relu_derv(d_dZ_prev, d_Z[i - 1], n_in * batch_size);

      cudaFree(d_W_T);
      cudaFree(d_dZ);
      d_dZ = d_dZ_prev;
    }
  }
  cudaFree(d_dZ);
}

void MLP::update(float lr) {
  for (size_t i = 0; i < d_W.size(); i++) {
    int size_W = layer_dims[i] * layer_dims[i + 1];
    int size_b = layer_dims[i + 1];
    gpu_update_params(d_W[i], d_dW[i], lr, size_W);
    gpu_update_params(d_b[i], d_db[i], lr, size_b);
  }
}

float MLP::get_loss(const float *d_Y, int batch_size) {
  int L = d_W.size() - 1;
  int n_out = layer_dims[L + 1];
  int size = n_out * batch_size;
  
  // d_A[L+1] contains the output probabilities after forward pass
  return gpu_cross_entropy_loss(d_Y, d_A[L + 1], size);
}
