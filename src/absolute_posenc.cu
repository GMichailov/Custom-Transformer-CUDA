#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <math.h>
#include <torch/torch.h>

torch::Tensor absolute_encoding_cuda(torch::Tensor in, int64_t max_seq_len) {
    int batch_size = in.size(0);
    int seq_len = in.size(1);
    int d_model = in.size(2);

    // Create duplicate on same device.
    auto options = in.options();
    auto pe = in.clone(); // Clone, not in place because I need autograd to work.

    auto start = thrust::make_counting_iterator<int>(0);
    auto end = thrust::make_counting_iterator<int>(batch_size* seq_len * d_model);

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
