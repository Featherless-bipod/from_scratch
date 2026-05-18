#ifndef MLP_H
#define MLP_H

#include <string>
#include <vector>

class MLP {
public:
  // Parameters
  int input_dim;
  int hidden_dim;
  int output_dim;
  int num_hidden_layers;

  // Weights and Biases
  std::vector<float *> d_W;
  std::vector<float *> d_b;

  // Gradients
  std::vector<float *> d_dW;
  std::vector<float *> d_db;

  // Cache for Forward Pass
  // A[0] = Input, A[1] = H1_out ...
  // Z[0] = H1_pre, Z[1] = H2_pre ...
  std::vector<float *> d_A;
  std::vector<float *> d_Z;

  // Dimensions of each layer
  std::vector<int> layer_dims;

  MLP(int input, int hidden, int output, int layers);
  ~MLP();

  void init_parameters();
  void forward(const float *d_X, int batch_size);
  void backward(const float *d_Y, int batch_size);
  void update(float lr);

  float get_loss(const float *d_Y, int batch_size);

  int get_max_dim(); // for temporary buffers if needed
};

#endif
