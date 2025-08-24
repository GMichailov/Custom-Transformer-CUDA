Standing Out: Focus on smaller nets rather than massive ones
- Computation Speed Up Mode (For Inference)
    - Fuse entire attention layers into a single kernel (multihead attention + MLP + norm) (Flash attention)
    - Specialized kernel for real-time models that assumes 1 item and never batches
    - Multihead Latent Attention Compression.
    - Switch to CublasLt and make this work with mixed precision.
- Memory Aware Mode (For Training)
    - Aggressive Recomputation in backward pass to avoid gradient overhead.
    - Reuse buffers.
    - Create super memory aware model that can handle super low levels of RAM by pulling in parts of the model at different times (ie layer N is active, layer N+5 just got the signal to load in, layer N-5 has been sent back to disk from RAM) for devices like raspeberry pis with 2 GB RAM.

- Make QKV one large matrix and pass pointers accordingly to avoid memory movements.
- Make repetetive cpu computations multiplying model dims together constexpr


- Operations
    - Needs
        - QkvFusedProjection: X * [Wq | Wk | Wv]
        - QkvPackedGradients: dX = dQ * Wq^T + dK * Wk^T + dV * Wv^T -> [dQ | dK | dV] (wide) * [Wq^T | Wk^T | Wv^T] (Tall)
        - dWqkv = X^T * [dQ | dK | dV]
        - dbias = row - sum([dQm dK, dV])
        - QKMatmul: QK^T / root(d_head)
        - Softmax but later on, use torch for now
        - AttentionOutput: Softmax_Result * V
        - Backward for attention output
        - OutputProjection: attention * Wo + bias
        - Backward: similar to QKV
        - FeedForward: Matmul + fused bias + activation
    - Functions
        - Matmul
        - Matmul fused bias
        - matmul fused scaling