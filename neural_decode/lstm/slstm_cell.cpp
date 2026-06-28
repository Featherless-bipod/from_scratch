#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> slstm_forward_pointwise_cpp(
    torch::Tensor Wx,
    torch::Tensor Ry,
    torch::Tensor b,
    torch::Tensor states
) {
    auto raw = Wx + Ry + b; 
    auto B = states.size(1);
    auto H = states.size(2);
    
    auto states_unbound = states.view({4, B, H}).unbind(0);
    auto y = states_unbound[0];
    auto c = states_unbound[1];
    auto n = states_unbound[2];
    auto m = states_unbound[3];
    
    auto raw_unbound = raw.view({B, 4, H}).unbind(1);
    auto iraw = raw_unbound[0];
    auto fraw = raw_unbound[1];
    auto zraw = raw_unbound[2];
    auto oraw = raw_unbound[3];
    
    auto logfplusm = m + fraw;
    
    auto n_all_zero = torch::all(n == 0.0).item<bool>();
    torch::Tensor mnew;
    if (n_all_zero) {
        mnew = iraw;
    } else {
        mnew = torch::max(iraw, logfplusm);
    }
    
    auto ogate = torch::sigmoid(oraw);
    auto igate = torch::minimum(torch::exp(iraw - mnew), torch::ones_like(iraw));
    auto fgate = torch::minimum(torch::exp(logfplusm - mnew), torch::ones_like(iraw));
    
    auto cnew = fgate * c + igate * torch::tanh(zraw);
    auto nnew = fgate * n + igate;
    auto ynew = ogate * cnew / nnew;
    
    auto new_states = torch::stack({ynew, cnew, nnew, mnew}, 0);
    auto gates = torch::stack({igate, fgate, zraw, ogate}, 0);
    
    return {new_states, gates};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("slstm_forward_pointwise", &slstm_forward_pointwise_cpp, "sLSTM forward pointwise (C++)");
}
