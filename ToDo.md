- Create Transformer Block struct
- Create Model struct
- Figure out cublas handle situation

- Quick small training run to purposely overfit model to make sure it works.
- Create training run on MNIST with libtorch

- switch to cublaslt to support mixed precision

- Fuse Transformer Layers into one large GEMM

Standing Out: Focus on smaller nets rather than massive ones
- Computation Speed Up Mode (For Inference)
    - Fuse entire attention layers into a single kernel (multihead attention + MLP + norm).
    - FlashAttention
    - Specialized kernel for real-time models that assumes 1 item and never batches
    - Multihead Latent Attention Compression.
    - Switch to CublasLt and make this work with mixed precision.
- Memory Aware Mode (For Training)
    - Aggressive Recomputation in backward pass to avoid gradient overhead.
    - Reuse buffers.
    - Create super memory aware model that can handle super low levels of RAM by pulling in parts of the model at different times (ie layer N is active, layer N+5 just got the signal to load in, layer N-5 has been sent back to disk from RAM) for devices like raspeberry pis with 2 GB RAM.

- Make QKV one large matrix and pass pointers accordingly to avoid memory movements.