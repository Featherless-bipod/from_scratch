#include "mlp.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void load_csv(const char *filename, std::vector<float> &X,
              std::vector<float> &Y, int &rows, int &cols, int max_rows = -1) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    exit(1);
  }

  std::string line;
  rows = 0;
  cols = 0;

  // skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (max_rows > 0 && rows >= max_rows)
      break;
    if (line.empty()) continue;

    // Handle quoted font names containing commas
    size_t data_start = 0;
    if (line.front() == '"') {
        size_t quote_end = line.find('"', 1);
        if (quote_end != std::string::npos) {
            data_start = quote_end + 1; 
            if (data_start < line.size() && line[data_start] == ',') {
                data_start++; 
            }
        }
    } else {
        size_t first_comma = line.find(',');
        if (first_comma != std::string::npos) {
            data_start = first_comma + 1;
        }
    }

    std::string rest = line.substr(data_start);
    std::stringstream ss(rest);
    std::string val;

    // read label
    std::getline(ss, val, ',');
    Y.push_back(std::stof(val));

    // read pixels
    int c = 0;
    while (std::getline(ss, val, ',')) {
      X.push_back(std::stof(val) / 255.0f); // normalize
      c++;
    }
    if (rows == 0)
      cols = c;
    rows++;
  }
}

// one one one *echo* hot hot hot *echo* ONEEEEEE HOTTTTTTTTTTTT
void to_one_hot(const std::vector<float> &Y, std::vector<float> &Y_one_hot,
                int num_classes) {
  Y_one_hot.resize(Y.size() * num_classes, 0.0f);
  for (size_t i = 0; i < Y.size(); i++) {
    int label = (int)Y[i];
    if (label >= 0 && label < num_classes) {
      Y_one_hot[label * Y.size() + i] = 1.0f;
    }
  }
}

int main() {
  srand(time(NULL));

  // 1. Load Data
  std::cout << "Loading data..." << std::endl; // funky little print statement
  std::vector<float> h_X_vec, h_Y_vec;
  int samples, features;

  const char *data_path = "./TMNIST_Data.csv";
  int cols = 0;
  load_csv(data_path, h_X_vec, h_Y_vec, samples, cols);

  features = cols;
  int classes = 10;

  std::cout << "Loaded " << samples << " samples with " << features
            << " features." << std::endl;

  std::vector<float> h_Y_one_hot;
  to_one_hot(h_Y_vec, h_Y_one_hot, classes);

  // 2. Prepare GPU Data
  float *d_X, *d_Y;
  cudaMalloc(&d_X, h_X_vec.size() * sizeof(float));
  cudaMalloc(&d_Y, h_Y_one_hot.size() * sizeof(float));

  cudaMemcpy(d_X, h_X_vec.data(), h_X_vec.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, h_Y_one_hot.data(), h_Y_one_hot.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // 3. Init MLP
  std::cout << "Initializing MLP..." << std::endl;
  MLP mlp(features, 10, classes, 2); 
  mlp.init_parameters();

  // 4. Training Loop
  int epochs = 100;
  float lr = 0.1f;
  int batch_size = samples;

  std::cout << "Starting training..." << std::endl;
  for (int i = 0; i < epochs; i++) {
    // Forward
    mlp.forward(d_X, batch_size);

    // Backward
    mlp.backward(d_Y, batch_size);

    // Update
    // Note: We need to scale lr by 1/m here because we didn't do it in backward
    mlp.update(lr / batch_size);

    if (i % 10 == 0) {
      float loss = mlp.get_loss(d_Y, batch_size);
      std::cout << "Epoch " << i << " complete. Loss: " << loss / batch_size << std::endl;
    }
  }

  std::cout << "Training complete." << std::endl;

  cudaFree(d_X);
  cudaFree(d_Y);

  return 0;
}
