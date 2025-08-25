#include "Gemm.hpp"
#include "Precision.hpp"

using namespace ember::Gemm;

template <typename scalar_t>
inline void matmul(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
)
{
    auto compute_precision = ember::Precision::CublasLtType<scalar_t>::compute;
    auto datatype = ember::Precision::CublasLtType<scalar_t>::type;

    cublasOperation_t transposeMatrixA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transposeMatrixB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(
        &desc, 
        compute_precision,
        datatype
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transposeMatrixA,
        sizeof(&transposeMatrixA)
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transposeMatrixB,
        sizeof(&transposeMatrixB)
    );

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(
        &Adesc, datatype, 
        transposeA ? shared_dim : rows,
        transposeA ? rows : shared_dim,
        shared_dim
    );
    cublasLtMatrixLayoutCreate(
        &Bdesc, datatype, 
        transposeB ? cols : shared_dim,
        transposeB ? shared_dim : cols,
        cols
    );
    cublasLtMatrixLayoutCreate(&Cdesc, datatype, rows, cols, cols);

    scalar_t alpha = scaling_factor;
    scalar_t beta_val = beta;

    cublasLtMatmul(
        handle, desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta_val,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, 0
    );

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}



template <typename scalar_t>
inline void matmul_bias(
    const scalar_t* A, const scalar_t* B, const scalar_t* bias, scalar_t* C,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
)
{
    auto compute_precision = ember::Precision::CublasLtType<scalar_t>::compute;
    auto datatype = ember::Precision::CublasLtType<scalar_t>::type;

    cublasOperation_t transposeMatrixA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transposeMatrixB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(
        &desc, 
        compute_precision,
        datatype
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transposeMatrixA,
        sizeof(transposeMatrixA)
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transposeMatrixB,
        sizeof(transposeMatrixB)
    );

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, 
        &epilogue, sizeof(epilogue)
    );
    cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, 
        &bias, sizeof(bias)
    );

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(
        &Adesc, datatype, 
        transposeA ? shared_dim : rows,
        transposeA ? rows : shared_dim,
        shared_dim
    );
    cublasLtMatrixLayoutCreate(
        &Bdesc, datatype, 
        transposeB ? cols : shared_dim,
        transposeB ? shared_dim : cols,
        cols
    );
    cublasLtMatrixLayoutCreate(&Cdesc, datatype, rows, cols, cols);

    scalar_t alpha = scaling_factor;
    scalar_t beta_val = beta;
    cublasLtMatmul(
        handle, desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta_val,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, 0
    );

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}



template <typename scalar_t>
inline void matmul_strided_batch(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int batch_count,
    long long int strideA, long long int strideB, long long int strideC,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
)
{
    auto compute_precision = ember::Precision::CublasLtType<scalar_t>::compute;
    auto datatype = ember::Precision::CublasLtType<scalar_t>::type;

    cublasOperation_t transposeMatrixA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transposeMatrixB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(
        &desc, 
        compute_precision,
        datatype
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transposeMatrixA,
        sizeof(transposeMatrixA)
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transposeMatrixB,
        sizeof(transposeMatrixB)
    );

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(
        &Adesc, datatype, 
        transposeA ? shared_dim : rows,
        transposeA ? rows : shared_dim,
        shared_dim
    );
    cublasLtMatrixLayoutCreate(
        &Bdesc, datatype, 
        transposeB ? cols : shared_dim,
        transposeB ? shared_dim : cols,
        cols
    );
    cublasLtMatrixLayoutCreate(&Cdesc, datatype, rows, cols, cols);

    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
        &batch_count, sizeof(batch_count)
    );
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
        &batch_count, sizeof(batch_count)
    );
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
        &batch_count, sizeof(batch_count)
    );
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &strideA, sizeof(strideA)
    );
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &strideB, sizeof(strideB)
    );
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &strideC, sizeof(strideC)
    );

    scalar_t alpha = scaling_factor;
    scalar_t beta_value = beta;
    cublasLtMatmul(
        handle, desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta_value,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, 0
    );

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}





template <typename scalar_t>
inline void matmul_bias_activation(
    const scalar_t* A, const scalar_t* B, const scalar_t* bias, scalar_t* C,
    int rows, int cols, int shared_dim,
    cublasLtEpilogue_t epilogue
)
{
    auto compute_precision = ember::Precision::CublasLtType<scalar_t>::compute;
    auto datatype = ember::Precision::CublasLtType<scalar_t>::type;

    cublasOperation_t transposeMatrixA = CUBLAS_OP_N;
    cublasOperation_t transposeMatrixB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(
        &desc, 
        compute_precision,
        datatype
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transposeMatrixA,
        sizeof(transposeMatrixA)
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transposeMatrixB,
        sizeof(transposeMatrixB)
    );

    cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, 
        &epilogue, sizeof(epilogue)
    );
    cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, 
        &bias, sizeof(bias)
    );

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, datatype, rows, shared_dim, shared_dim);
    cublasLtMatrixLayoutCreate(&Bdesc, datatype, shared_dim, cols, cols);
    cublasLtMatrixLayoutCreate(&Cdesc, datatype, rows, cols, cols);

    scalar_t alpha = static_cast<scalar_t>(1.0f);
    scalar_t beta  = static_cast<scalar_t>(0.0f);
    cublasLtMatmul(
        handle, desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, 0
    );

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}





template <typename scalar_t>
inline void matmul_activation(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int rows, int cols, int shared_dim,
    cublasLtEpilogue_t epilogue
)
{
    auto compute_precision = ember::Precision::CublasLtType<scalar_t>::compute;
    auto datatype = ember::Precision::CublasLtType<scalar_t>::type;

    cublasOperation_t transposeMatrixA = CUBLAS_OP_N;
    cublasOperation_t transposeMatrixB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(
        &desc, 
        compute_precision,
        datatype
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transposeMatrixA,
        sizeof(transposeMatrixA)
    );
    cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transposeMatrixB,
        sizeof(transposeMatrixB)
    );

    cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, 
        &epilogue, sizeof(epilogue)
    );

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, datatype, rows, shared_dim, shared_dim);
    cublasLtMatrixLayoutCreate(&Bdesc, datatype, shared_dim, cols, cols);
    cublasLtMatrixLayoutCreate(&Cdesc, datatype, rows, cols, cols);

    scalar_t alpha = static_cast<scalar_t>(1.0f);
    scalar_t beta  = static_cast<scalar_t>(0.0f);
    cublasLtMatmul(
        handle, desc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, 0
    );

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}