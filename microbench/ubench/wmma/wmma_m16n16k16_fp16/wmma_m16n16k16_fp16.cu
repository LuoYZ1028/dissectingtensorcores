#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include "../../../hw_def/hw_def.h"

// #define SHARED_MEM_SIZE (32 * 102 4) // 32 KB
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
//#define THREADS_NUM 32
// iterate over the array ITERS times

#ifndef ILPconfig
#define ILPconfig 1
#endif

#ifndef ITERS
#define ITERS  (1024)
#endif

using namespace nvcuda;
#define M 16
#define N 16
#define K 16
#define BLOCKS_NUM 1
#define WARP_SIZE 32
#define A_LAYOUT wmma::row_major
#define B_LAYOUT wmma::col_major
#define D_LAYOUT wmma::mem_row_major
#define N_GLOBAL N
#define K_GLOBAL K
#define factor ((N_GLOBAL / N) * (K_GLOBAL / K))
#define a_num (ILPconfig * factor)
#define b_num (ILPconfig * factor)

template <class T, class R>
__global__ void tensor161616_flops(uint64_t *startClk, uint64_t *stopClk, 
	  T *mat_a, T *mat_b, R *res, int M_GLOBAL) {
	int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	// int factor = (N_GLOBAL / N) * (K_GLOBAL / K);
	// int a_frag_num = ILPconfig * factor;
	// int b_frag_num = a_frag_num;
	// declare fragments
	wmma::fragment<wmma::matrix_a, M, N, K, T, A_LAYOUT> a_frag[a_num];
	wmma::fragment<wmma::matrix_b, M, N, K, T, B_LAYOUT> b_frag[b_num];
	wmma::fragment<wmma::accumulator, M, N, K, R> acc_frag[ILPconfig];
	for (int i = 0; i < ILPconfig; i++)
		wmma::fill_fragment(acc_frag[i], 0.0f);
	// wmma::fragment<wmma::accumulator, M, N, K, R> acc_frag;
	// wmma::fill_fragment(acc_frag, 0.0f);

	int a_row = warpId * M;
	// loop over n & k
	for (int n = 0; n < N_GLOBAL; n += N) {
		int b_col = n;
		for (int k = 0; k < K_GLOBAL; k += K) {
			int a_col = k;
			int b_row = k;
			// load input then perform MMA 
			if (a_row < M_GLOBAL && a_col < K_GLOBAL && b_row < K_GLOBAL && b_col < N_GLOBAL) {
				wmma::load_matrix_sync(a_frag[0 * factor + b_col * N + a_col], 
									mat_a + a_row * K_GLOBAL + a_col, K_GLOBAL);
				wmma::load_matrix_sync(b_frag[0 * factor + b_col * N + b_row], 
									mat_b + b_col * K_GLOBAL + b_row, K_GLOBAL);
				// wmma::mma_sync(acc_frag[0], a_frag[0], b_frag[0], acc_frag[0]);
				#if ILPconfig >= 2
				wmma::load_matrix_sync(a_frag[1 * factor + b_col * N + a_col], 
									mat_a + a_row * K_GLOBAL + a_col, K_GLOBAL);
				wmma::load_matrix_sync(b_frag[1 * factor + b_col * N + b_row], 
									mat_b + b_col * K_GLOBAL + b_row, K_GLOBAL);
				// wmma::mma_sync(acc_frag[1], a_frag[1], b_frag[1], acc_frag[1]);
				#endif
				#if ILPconfig >= 3
				wmma::load_matrix_sync(a_frag[2 * factor + b_col * N + a_col], 
									mat_a + a_row * K_GLOBAL + a_col, K_GLOBAL);
				wmma::load_matrix_sync(b_frag[2 * factor + b_col * N + b_row],
									mat_b + b_col * K_GLOBAL + b_row, K_GLOBAL);
				// wmma::mma_sync(acc_frag[2], a_frag[2], b_frag[2], acc_frag[2]);
				#endif
				#if ILPconfig >= 4
				wmma::load_matrix_sync(a_frag[3 * factor + b_col * N + a_col], 
									mat_a + a_row * K_GLOBAL + a_col, K_GLOBAL);
				wmma::load_matrix_sync(b_frag[3 * factor + b_col * N + b_row], 
									mat_b + b_col * K_GLOBAL + b_row, K_GLOBAL);
				// wmma::mma_sync(acc_frag[3], a_frag[3], b_frag[3], acc_frag[3]);
				#endif
				#if ILPconfig >= 5
				wmma::load_matrix_sync(a_frag[4 * factor + b_col * N + a_col], 
									mat_a + a_row * K_GLOBAL + a_col, K_GLOBAL);
				wmma::load_matrix_sync(b_frag[4 * factor + b_col * N + b_row], 
									mat_b + b_col * K_GLOBAL + b_row, K_GLOBAL);
				// wmma::mma_sync(acc_frag[4], a_frag[4], b_frag[4], acc_frag[4]);
				#endif
			}
		}
		// wmma::store_matrix_sync(res + a_row * N_GLOBAL + b_col, acc_frag[0], N_GLOBAL, D_LAYOUT);
	}
	

	uint64_t start = 0;
	uint64_t stop = 0;

	asm volatile("bar.sync 0;");
	asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

	__syncwarp();
	// loop over n & k
	for (int n = 0; n < N_GLOBAL; n += N) {
		int b_col = n;
		for (int k = 0; k < K_GLOBAL; k += K) {
			int a_col = k;
			int b_row = k;
			// load input then perform MMA 
			if (a_row < M_GLOBAL && a_col < K_GLOBAL && b_row < K_GLOBAL && b_col < N_GLOBAL) {
				wmma::mma_sync(acc_frag[0], a_frag[0 * factor + b_col * N + a_col], 
							b_frag[0 * factor + b_col * N + b_row], acc_frag[0]);
				#if ILPconfig >= 2
				wmma::mma_sync(acc_frag[1], a_frag[1 * factor + b_col * N + a_col], 
							b_frag[1 * factor + b_col * N + b_row], acc_frag[1]);
				#endif
				#if ILPconfig >= 3
				wmma::mma_sync(acc_frag[2], a_frag[2 * factor + b_col * N + a_col], 
							b_frag[2 * factor + b_col * N + b_row], acc_frag[2]);
				#endif
				#if ILPconfig >= 4
				wmma::mma_sync(acc_frag[3], a_frag[3 * factor + b_col * N + a_col], 
							b_frag[3 * factor + b_col * N + b_row], acc_frag[3]);
				#endif
				#if ILPconfig >= 5
				wmma::mma_sync(acc_frag[4], a_frag[4 * factor + b_col * N + a_col],
							b_frag[4 * factor + b_col * N + b_row], acc_frag[4]);
				#endif
			}
		}
	}
	__syncwarp();
	asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

	int res_col = 0;
	wmma::store_matrix_sync(res + a_row * N_GLOBAL + res_col, acc_frag[0], N_GLOBAL, D_LAYOUT);
	startClk[warpId] = start;
	stopClk[warpId] = stop;
}

// template <class T, class R> 
__host__ void mma_on_host(half *A, half *B, float *D, int M_GLOBAL) {
	for (int m = 0; m < M_GLOBAL; m++) {
		for (int n = 0; n < N_GLOBAL; n++) {
			float temp = 0.0f;
			for (int k = 0; k < K_GLOBAL; k++) {
				temp += __half2float(A[m * K_GLOBAL + k]) * __half2float(B[n * K_GLOBAL + k]);
			}
			D[m * N_GLOBAL + n] = temp;
		}
	}
}

template <class T, class R> 
float tensor161616_max_flops(int warp_num, bool report_fma_bw = false) {
	intilizeDeviceProp(0);
	int total_threads = warp_num * WARP_SIZE;

	uint32_t M_GLOBAL = M * warp_num;
	// uint32_t N_GLOBAL = N;
	// uint32_t K_GLOBAL = K;
	uint32_t A_GLOBAL = M_GLOBAL * K_GLOBAL;
	uint32_t B_GLOBAL = K_GLOBAL * N_GLOBAL;
	uint32_t D_GLOBAL = M_GLOBAL * N_GLOBAL;

	uint64_t *startClk = (uint64_t *)malloc(total_threads * sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t *)malloc(total_threads * sizeof(uint64_t));
	T *data_a = (T *)malloc(A_GLOBAL * sizeof(T));
	T *data_b = (T *)malloc(B_GLOBAL * sizeof(T));
	R *cuda_res = (R *)malloc(D_GLOBAL * sizeof(R));
	R *cpu_res = (R *)malloc(D_GLOBAL * sizeof(R));

	uint64_t *startClk_ptr;
	uint64_t *stopClk_ptr;
	T *data_a_ptr;
	T *data_b_ptr;
	R *cuda_res_ptr;

	// 矩阵AB内，元素都是固定的数
	for (int i = 0; i < A_GLOBAL; i++) { data_a[i] = T(i / 16); }
	for (int i = 0; i < B_GLOBAL; i++) { data_b[i] = T(i / 16); }
	
	// 使用 cudaMalloc 在 GPU 内分配空间，地址赋予 ptr
	gpuErrchk(cudaMalloc(&startClk_ptr, total_threads * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&stopClk_ptr, total_threads * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&data_a_ptr, A_GLOBAL * sizeof(T)));
	gpuErrchk(cudaMalloc(&data_b_ptr, B_GLOBAL * sizeof(T)));
	gpuErrchk(cudaMalloc(&cuda_res_ptr, D_GLOBAL * sizeof(R)));
	// 将数据搬到上述 GPU 分配的空间
	gpuErrchk(cudaMemcpy(data_a_ptr, data_a, 
		A_GLOBAL * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data_b_ptr, data_b, 
		B_GLOBAL * sizeof(T), cudaMemcpyHostToDevice));
	// 给 mma 操作计时
	tensor161616_flops<T, R><<<BLOCKS_NUM, WARP_SIZE * warp_num>>>(
		startClk_ptr, stopClk_ptr, data_a_ptr, data_b_ptr, cuda_res_ptr, 
		M_GLOBAL);
	mma_on_host(data_a, data_b, cpu_res, M_GLOBAL);
	gpuErrchk(cudaPeekAtLastError());
	// 没有发生错误才将 时间数据 和 乘法结果 放入 GPU 内部
	gpuErrchk(cudaMemcpy(startClk, startClk_ptr, 
		total_threads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(stopClk, stopClk_ptr, 
		total_threads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(cuda_res, cuda_res_ptr, 
		D_GLOBAL * sizeof(R), cudaMemcpyDeviceToHost));
  
	for (int i = 0 ; i < D_GLOBAL; i++) {
		if (fabs(cuda_res[i] - cpu_res[i]) > 0.01 * cpu_res[i])
       		printf("mismatch i=%d result_cuda=%f result_cpu=%f\n", i, cuda_res[i],
              cpu_res[i]);
	}

	// 总耗时 = 最晚结束时间 - 最早开始时间
	uint64_t total_time =
		*std::max_element(&stopClk[0], &stopClk[total_threads]) -
		*std::min_element(&startClk[0], &startClk[total_threads]);

	float fma_bw = ((float)(ITERS * M * N * K * ILPconfig * warp_num)) 
		/ (float)total_time;

	std::cout << "wmma-m" << M << "n" << N << "k" << K << \
		".row.row.fp16  latency " << (float)total_time/(float)ITERS << " cycles\n";
	std::cout << "FMA tensor bandwidth = " << fma_bw << "(FMA/clk/SM)\n";
	std::cout << "Total Clk number = " << total_time << std::endl;
	return 0.0;
}

int main() {
	std::vector<int> warps = {1, 2, 6, 8, 12, 16};
	intilizeDeviceProp(0);
	std::cout << "***********************************" << std::endl;
	std::cout << "wmma-m" << M << "n" << N << "k" << K << \
		".row.row.fp16 microbenchmark with ILP = " << ILPconfig << std::endl;

	for (auto& e:warps) {
		std::cout << "Number of warps = " << e << std::endl;
		tensor161616_max_flops<half, float>(e);
		std::cout << std::endl;
	}
	return 0;
}
  
