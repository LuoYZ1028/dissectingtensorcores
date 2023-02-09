#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "../../../hw_def/hw_def.h"

// #define SHARED_MEM_SIZE (32 * 1024 / 4) // 32 KB
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

template <class T, class R>
__global__ void tensor161616_flops(uint64_t *startClk, uint64_t *stopClk, 
	  half *mat_a, half *mat_b, float *res, uint32_t stride) {
  	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x * blockDim.x + tid;
	uint32_t wid = bid / warpSize;
	// 取当前线程所在 warp 对应矩阵的首地址
	mat_a = mat_a + wid * M * K;
	mat_b = mat_b + wid * K * N;
	res = res + wid * M * N;

	/* 1. 建立各矩阵的 fragment 寄存器，存放矩阵的一行（一列） */
	wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> frag_A;
	wmma::fragment<wmma::matrix_b, M, N, K, T, wmma::row_major> frag_B;
	wmma::fragment<wmma::accumulator, M, N, K, R> frag_D;
	
	/* 2. 将对应的切片放入 fragment 寄存器内，结果寄存器清零 */
	// load_matrix_sync 的参数3：ldm 描述行主序矩阵的行间元素跨度
	wmma::load_matrix_sync(frag_A, mat_a, K);
	wmma::load_matrix_sync(frag_B, mat_b, N);
	wmma::fill_fragment(frag_D, 0.0f);

	uint64_t start = 0;
	uint64_t stop = 0;
	/* 3. 同步所有线程，开始计时 */
	asm volatile("bar.sync 0;");
	asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
	// 似乎这个循环只是为了拖延时间以免计时值太小
	for (int j = 0; j < ITERS; ++j) {
		wmma::mma_sync(frag_D, frag_A, frag_B, frag_D);
		
		// ILP > 1时需要再次同步 ?
		#if (ILPconfig > 1)
		__syncwarp();
		#endif
	}
	/* 4. 同步所有线程，结束计时，并将结果写入 res 中 */
	__syncwarp();
	asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
	wmma::store_matrix_sync(res, frag_D, 16, wmma::mem_row_major);

	startClk[bid] = start;
	stopClk[bid] = stop;
}


template <class T, class R> 
float tensor161616_max_flops(int threads_per_blk, bool report_fma_bw = false) {
	intilizeDeviceProp(0);
	int total_threads = threads_per_blk * BLOCKS_NUM;
	int warp_num = total_threads / WARP_SIZE;
	// 根据假设，A 和 B 矩阵的大小都为16*16
	unsigned A_GLOBAL = M * K * warp_num * ILPconfig;
	unsigned B_GLOBAL = K * N * warp_num * ILPconfig;
	unsigned D_GLOBAL = M * N * warp_num * ILPconfig;

	uint64_t *startClk = (uint64_t *)malloc(total_threads * sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t *)malloc(total_threads * sizeof(uint64_t));
	T *data_a = (T *)malloc(A_GLOBAL * sizeof(T));
	T *data_b = (T *)malloc(B_GLOBAL * sizeof(T));
	R *res = (R *)malloc(D_GLOBAL * sizeof(R));
  
	uint64_t *startClk_ptr;
	uint64_t *stopClk_ptr;
	T *data_a_ptr;
	T *data_b_ptr;
	R *res_ptr;
  
	// 矩阵AB内，元素都是固定的从0~255的顺序数
	for (uint32_t i = 0; i < M * K; i++) { data_a[i] = (T)i; }
	for (uint32_t i = 0; i < K * N; i++) { data_b[i] = (T)i; }
	
	// 使用 cudaMalloc 在 GPU 内分配空间，地址赋予 ptr
	gpuErrchk(cudaMalloc(&startClk_ptr, total_threads * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&stopClk_ptr, total_threads * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&data_a_ptr, A_GLOBAL * sizeof(T)));
	gpuErrchk(cudaMalloc(&data_b_ptr, B_GLOBAL * sizeof(T)));
	gpuErrchk(cudaMalloc(&res_ptr, D_GLOBAL * sizeof(R)));
	// 将数据搬到上述 GPU 分配的空间
	gpuErrchk(cudaMemcpy(data_a_ptr, data_a, 
		A_GLOBAL * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data_b_ptr, data_b, 
		B_GLOBAL * sizeof(T), cudaMemcpyHostToDevice));
	// 给 mma 操作计时
	tensor161616_flops<T, R><<<BLOCKS_NUM, threads_per_blk>>>(
		startClk_ptr, stopClk_ptr, data_a_ptr, data_b_ptr, res_ptr, 0);
	gpuErrchk(cudaPeekAtLastError());
	// 没有发生错误才将 时间数据 和 乘法结果 放入 GPU 内部
	gpuErrchk(cudaMemcpy(startClk, startClk_ptr, 
		total_threads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(stopClk, stopClk_ptr, 
		total_threads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res, res_ptr, 
		D_GLOBAL * sizeof(R), cudaMemcpyDeviceToHost));
  
	float mma_bw, fma_bw;
	// 总耗时 = 最晚结束时间 - 最早开始时间
	uint64_t total_time =
		*std::max_element(&stopClk[0], &stopClk[total_threads]) -
		*std::min_element(&startClk[0], &startClk[total_threads]);

	// ? 不清楚此处是什么意思
	float fpuFMA = (float)(ITERS * total_threads * 1 * 1 * 1 * 0 ) /
		  ((float)total_time);  // max 64FMA/clk/SM on RTX3070Ti

	mma_bw = ((float)(ITERS * total_threads)) / (float)total_time;
	fma_bw = ((float)(ITERS * M * N * K * ILPconfig * warp_num)) 
		/ (float)total_time;
  
	std::cout << "wmma-m" << M << "n" << N << "k" << K << \
		".row.row.fp16  latency " << (float)total_time/(float)ITERS << " cycles\n";
	std::cout << "FMA tensor bandwidth = " << fma_bw + fpuFMA << "(FMA/clk/SM)\n";
	std::cout << "Total Clk number = " << total_time << std::endl;
  
	if (report_fma_bw)
	  return fma_bw;
	else
	  return mma_bw;
}

int main() {
	std::vector<int> warps = {1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
	intilizeDeviceProp(0);
	std::cout << "***********************************" << std::endl;
	std::cout << "wmma-m" << M << "n" << N << "k" << K << \
		".row.row.fp16 microbenchmark with ILP = " << ILPconfig << std::endl;

	for (auto& e:warps) {
		std::cout << "Number of warps = " << e << std::endl;
		tensor161616_max_flops<__half, float>(WARP_SIZE * e);
		std::cout << std::endl;
	}
	return 0;
}
  