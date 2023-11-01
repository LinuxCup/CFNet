#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "atomics.cuh"

#define ceil_div(N, M) ((N + M - 1) / M)
typedef unsigned long long int ULONG;

int const threadsPerBlock = sizeof(ULONG) * 8;

template<typename scalar_t>
inline scalar_t* DATA_PTR(at::Tensor mat){
    return at::cuda::detail::getTensorInfo<scalar_t, int64_t>(mat).data;
}

// vote nms
namespace vote_nms{
    // pcds_fg (N, 4), 4 -> (x, y, z, center_score)
    // pcds_center (K, 3), 3 -> (cx, cy, cz)
    // dist_thresh, distant smaller than dist_thresh is regarded as same object
    __global__ void compute_matching_cuda_kernel(const float* pcds_fg, ULONG* matching_mat, ULONG* matching_mat_vote,
        float dist_thresh, float vote_thresh, int64_t N, int64_t C, int64_t col_blocks)
    {
        for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (N * N); i = i + blockDim.x * gridDim.x){
            int64_t b0 = i / N;
            int64_t b1 = i - b0 * N;
            if(b1 > b0){
                int64_t diff_0 = b0 * C;
                int64_t diff_1 = b1 * C;

                float dist = 0;
                float v_tmp=0;
                for(int kn=0; kn < 3; kn++){
                    v_tmp = pcds_fg[diff_0 + kn] - pcds_fg[diff_1 + kn];
                    dist = dist + v_tmp * v_tmp;
                }
                
                dist = sqrt(dist);
                
                int nblock = b1 / threadsPerBlock;
                int inblock = b1 % threadsPerBlock;
                ULONG t_v = 1ULL << inblock;
                if(dist < dist_thresh){
                    atomicOr(&matching_mat[b0 * col_blocks + nblock], t_v);
                }

                if(dist < vote_thresh){
                    atomicOr(&matching_mat_vote[b0 * col_blocks + nblock], t_v);
                }
            }
        }
    }

    __global__ void nms_kernel_post(const ULONG* matching_mat, ULONG* remv, int64_t* keep, int64_t N, int64_t C, int64_t col_blocks, int64_t K)
    {
        // init keep
        for(int i = 0; i < K; i++){
            keep[i] = -1;
        }

        // postprocess
        int num_to_keep = 0;
        for(int i = 0; (i < N) && (num_to_keep < K); i++){
            int nblock = i / threadsPerBlock;
            int inblock = i % threadsPerBlock;
            if(!(remv[nblock] & (1ULL << inblock))){
                keep[num_to_keep] = i;
                const ULONG* p = matching_mat + i * col_blocks;
                for(int j = nblock; j < col_blocks; j++){
                    remv[j] |= p[j];
                }

                num_to_keep += 1;
            }
        }
    }

    __global__ void vote_merge_kernel(const float* pcds_fg, const ULONG* matching_mat_vote, const int64_t* keep, float* pcds_center,
        int64_t N, int64_t C, int64_t col_blocks, int64_t K)
    {
        for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i = i + blockDim.x * gridDim.x){
            int64_t keep_idx = keep[i];
            if(keep_idx >= 0){
                float weight_sum = pcds_fg[keep_idx * C + 3];
                for(int kn=0; kn < 3; kn++){
                    pcds_center[i * 3 + kn] = pcds_fg[keep_idx * C + kn] * pcds_fg[keep_idx * C + 3];
                }

                const ULONG* cur_matching_mat_vote = matching_mat_vote + keep_idx * col_blocks;
                for(int j=(keep_idx+1); j < N; j++){
                    int nblock = j / threadsPerBlock;
                    int inblock = j % threadsPerBlock;
                    if((cur_matching_mat_vote[nblock] & (1ULL << inblock)))
                    {
                        weight_sum += pcds_fg[j * C + 3];
                        for(int kn=0; kn < 3; kn++){
                            pcds_center[i * 3 + kn] += pcds_fg[j * C + kn] * pcds_fg[j * C + 3];
                        }
                    }
                }

                // weight mean
                for(int kn=0; kn < 3; kn++){
                    pcds_center[i * 3 + kn] = pcds_center[i * 3 + kn] / weight_sum;
                }
            }
        }
    }
}


void vote_nms_cuda(at::Tensor pcds_fg, at::Tensor pcds_center, float dist_thresh, float vote_thresh)
{
    cudaSetDevice(pcds_fg.get_device());
    int64_t N = pcds_fg.size(0);
    int64_t C = pcds_fg.size(1);

    int64_t K = pcds_center.size(0);

    const int col_blocks = ceil_div(N, threadsPerBlock);
    
    // malloc GPU memory
    ULONG* matching_mat_data;
    cudaMalloc((void **)&matching_mat_data, N * col_blocks * sizeof(ULONG));
    cudaMemset(matching_mat_data, 0, N * col_blocks * sizeof(ULONG));

    ULONG* matching_mat_vote_data;
    cudaMalloc((void **)&matching_mat_vote_data, N * col_blocks * sizeof(ULONG));
    cudaMemset(matching_mat_vote_data, 0, N * col_blocks * sizeof(ULONG));

    // nms kernel
    float* pcds_fg_data = DATA_PTR<float>(pcds_fg);
    vote_nms::compute_matching_cuda_kernel<<<BLOCKS(N * N), THREADS>>>(pcds_fg_data, matching_mat_data, matching_mat_vote_data, dist_thresh, vote_thresh, N, C, col_blocks);

    // nms postprocess
    ULONG* remv_data;
    cudaMalloc((void **)&remv_data, col_blocks * sizeof(ULONG));
    cudaMemset(remv_data, 0, col_blocks * sizeof(ULONG));

    int64_t* keep_data;
    cudaMalloc((void **)&keep_data, K * sizeof(int64_t));
    cudaMemset(keep_data, 0, K * sizeof(int64_t));

    vote_nms::nms_kernel_post<<<1, 1>>>(matching_mat_data, remv_data, keep_data, N, C, col_blocks, K);

    // vote nms
    float* pcds_center_data = DATA_PTR<float>(pcds_center);
    vote_nms::vote_merge_kernel<<<BLOCKS(K), THREADS>>>(pcds_fg_data, matching_mat_vote_data, keep_data, pcds_center_data, N, C, col_blocks, K);

    // free GPU memoty
    cudaFree(matching_mat_data);
    cudaFree(matching_mat_vote_data);
    cudaFree(remv_data);
    cudaFree(keep_data);
}


void vote_nms_fast_cuda(at::Tensor pcds_fg, at::Tensor pcds_center, at::Tensor matching_mat, at::Tensor matching_mat_vote, at::Tensor remv, at::Tensor keep, float dist_thresh, float vote_thresh)
{
    cudaSetDevice(pcds_fg.get_device());
    int64_t N = pcds_fg.size(0);
    int64_t C = pcds_fg.size(1);

    int64_t K = pcds_center.size(0);

    const int col_blocks = ceil_div(N, threadsPerBlock);
    
    // malloc GPU memory
    ULONG* matching_mat_data = reinterpret_cast<ULONG*>(DATA_PTR<int64_t>(matching_mat));
    ULONG* matching_mat_vote_data = reinterpret_cast<ULONG*>(DATA_PTR<int64_t>(matching_mat_vote));

    // nms kernel
    float* pcds_fg_data = DATA_PTR<float>(pcds_fg);
    vote_nms::compute_matching_cuda_kernel<<<BLOCKS(N * N), THREADS>>>(pcds_fg_data, matching_mat_data, matching_mat_vote_data, dist_thresh, vote_thresh, N, C, col_blocks);

    // nms postprocess
    ULONG* remv_data = reinterpret_cast<ULONG*>(DATA_PTR<int64_t>(remv));
    int64_t* keep_data = DATA_PTR<int64_t>(keep);

    vote_nms::nms_kernel_post<<<1, 1>>>(matching_mat_data, remv_data, keep_data, N, C, col_blocks, K);

    // vote nms
    float* pcds_center_data = DATA_PTR<float>(pcds_center);
    vote_nms::vote_merge_kernel<<<BLOCKS(K), THREADS>>>(pcds_fg_data, matching_mat_vote_data, keep_data, pcds_center_data, N, C, col_blocks, K);
}


// voxel_sum
namespace voxel_sum{
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, D1, D2, ..., Dn)
    template<typename real>
    __global__ void VoxelSumCudaKernel1(real* pcds_ind_data, real* voxel_out_data,
                                        int64_t BS, int64_t N, int64_t D, int64_t loop,
                                        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, n;

            bs = i / N;
            n = i - bs * N;
            
            int64_t index_ind = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0];
            int flag = 1;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = static_cast<int64_t>(pcds_ind_data[index_ind + d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[1 + d];
                }
                else{
                    flag = 0;
                }
            }

            if(flag == 1){
                atomAdd(&voxel_out_data[index_voxel], 1);
            }
        }
    }
}

void voxel_sum_cuda(at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size)
{
    cudaSetDevice(pcds_ind.get_device());
    int64_t BS = pcds_ind.size(0);
    int64_t N = pcds_ind.size(1);
    int64_t D = pcds_ind.size(2);

    int64_t loop1 = BS * N;

    int64_t *pcds_ind_data = DATA_PTR<int64_t>(pcds_ind);
    int64_t *voxel_out_data = DATA_PTR<int64_t>(voxel_out);

    voxel_sum::VoxelSumCudaKernel1<int64_t><<<BLOCKS(loop1), THREADS>>>(pcds_ind_data, voxel_out_data, BS, N, D, loop1,
    DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));
}


// voxel_query
namespace voxel_query{
    // voxel_in, (BS, C, D1, D2, ..., Dn)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // pcds_feat, (BS, C, N, 1)
    template<typename real>
    __global__ void VoxelQueryCudaKernel1(real* pcds_feat_data, int64_t* pcds_ind_data, real* voxel_in_data,
                                        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                        int64_t* voxel_in_size, int64_t* voxel_in_stride)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, c, n;
            int64_t index_pcds, index_ind, index_voxel;
            int64_t index_res;

            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_ind = bs * N * D + n * D;
            index_voxel = bs * voxel_in_stride[0] + c * voxel_in_stride[1]; // bs and c

            int flag = 1;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = pcds_ind_data[index_ind + d];
                if((ind_tmp >=0) && (ind_tmp < voxel_in_size[2 + d])){
                    index_voxel = index_voxel + ind_tmp * voxel_in_stride[2 + d];
                }
                else{
                    flag = 0;
                }
            }
            if(flag == 1){
                pcds_feat_data[index_pcds] = voxel_in_data[index_voxel];
            }
        }
    }
}

void voxel_query_cuda(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_in, at::Tensor voxel_in_size, at::Tensor voxel_in_stride)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C = pcds_feat.size(1);
    int64_t N = pcds_feat.size(2);
    
    int64_t D = pcds_ind.size(2);
    int64_t loop1 = BS * C * N;
    AT_DISPATCH_ALL_TYPES(pcds_feat.scalar_type(), "VoxelQueryCudaKernel1", [&] {
        scalar_t *pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        int64_t *pcds_ind_data = DATA_PTR<int64_t>(pcds_ind);
        scalar_t *voxel_in_data = DATA_PTR<scalar_t>(voxel_in);

        voxel_query::VoxelQueryCudaKernel1<scalar_t><<<BLOCKS(loop1), THREADS>>>(pcds_feat_data, pcds_ind_data, voxel_in_data, BS, C, N, D, loop1,
        DATA_PTR<int64_t>(voxel_in_size), DATA_PTR<int64_t>(voxel_in_stride));
    });
}