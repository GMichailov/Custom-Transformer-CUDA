#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <math.h>
#include <torch/torch.h>

torch::Tensor absolute_encoding_unfused_cuda(torch::Tensor in) {
    int batch_size = in.size(0);
    int seq_len = in.size(1);
    int d_model = in.size(2);

    // Create duplicate on same device.
    auto options = in.options();
    auto pe = in.clone(); // Clone, not in place because I need autograd to work.

    auto start = thrust::make_counting_iterator<int>(0);
    auto end = thrust::make_counting_iterator<int>(pe.numel());

    thrust::device_ptr<float> out(pe.data_ptr<float>());

    thrust::transform(start, end, out,
        [=] __host__ __global__ (int idx) {
            int dim = idx % d_model;
            int pos = (idx / d_model) % seq_len;
            int batch = idx / (seq_len * d_model);

            float angle = pos / powf(10000.0f, (2.0f * (dim/2)) / (float)d_model);
            float val = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
            return out[idx] + val;
        }
    );

    return pe;
}

torch::Tensor rope_encoding_unfused_cuda(torch::Tensor in) {
    int batch_size = in.size(0);
    int seq_len = in.size(1);
    int n_heads = in.size(2);
    int head_dim = in.size(3);

    auto options = in.options();
    auto pe = in.clone();
    thrust::device_ptr<float> out(pe.data_ptr<float>());

    auto start = thrust::make_counting_iterator<int>(0);
    auto end = thrust::make_counting_iterator<int>(pe.numel());

    thrust::for_each(start, end,
        [=] __host__ __global__ (int idx) {
            int dim   = idx % head_dim;
            int head  = (idx / head_dim) % n_heads;
            int pos   = (idx / (head_dim * n_heads)) % seq_len;
            int batch = idx / (seq_len * n_heads * head_dim);

            if (dim % 2 == 0 && dim + 1 < head_dim) {
                int base = ((batch * seq_len + pos) * n_heads + head) * head_dim;
                int i = base + dim; 
                int j = base + dim + 1;

                float theta = pos / powf(10000.0f, (2.0f * (dim/2)) / (float)head_dim);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);

                float x0 = out[i];
                float x1 = out[j];

                out[i] = x0 * cos_t - x1 * sin_t;
                out[j] = x0 * sin_t + x1 * cos_t;
            }
        }    
    );

    return pe;
}