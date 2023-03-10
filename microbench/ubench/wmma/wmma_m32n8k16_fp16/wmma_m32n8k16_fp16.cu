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
#define M 32
#define N 8
#define K 16

template <class T, class R>
__global__ void tensor32816_flops(uint64_t *startClk, uint64_t *stopClk, 
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
		// #if (ILPconfig >= 2) 
		// asm volatile(
		//   "wmma.mma.sync.aligned.m16n16k16.row.row.f32.f16.f16.f32 "
		//   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//   : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
		//   : "r"(A[2]), "r"(A[3]), 
		//     "r"(B[1]), 
		//     "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
		//   ); 
		// #endif
		// #if (ILPconfig >= 3)
		// asm volatile(
		//   "wmma.mma.sync.aligned.m16n16k16.row.row.f32.f16.f16.f32 "
		//   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//   : "=f"(D[8]), "=f"(D[9]), "=f"(D[10]), "=f"(D[11])
		//   : "r"(A[4]), "r"(A[5]), 
		//     "r"(B[2]), 
		//     "f"(C[8]), "f"(C[9]), "f"(C[10]), "f"(C[11])
		//   ); 
		// #endif
		// #if (ILPconfig >= 4)
		// asm volatile(
		//     "wmma.mma.sync.aligned.m16n16k16.row.row.f32.f16.f16.f32 "
		//     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//     : "=f"(D[12]), "=f"(D[13]), "=f"(D[14]), "=f"(D[15])
		//     : "r"(A[6]), "r"(A[7]), 
		//       "r"(B[3]), 
		//       "f"(C[12]), "f"(C[13]), "f"(C[14]), "f"(C[15])
		// ); 
		// #endif
		// #if (ILPconfig >= 5)
		// asm volatile(
		//   "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
		//   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//   : "=f"(D[16]), "=f"(D[17]), "=f"(D[18]), "=f"(D[19])
		//   : "r"(A[8]), "r"(A[9]), 
		//     "r"(B[4]), 
		//     "f"(C[16]), "f"(C[17]), "f"(C[18]), "f"(C[19])
		// );
		// #endif
		// #if (ILPconfig >= 6)
		// asm volatile(
		//     "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
		//     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//     : "=f"(D[20]), "=f"(D[21]), "=f"(D[22]), "=f"(D[23])
		//     : "r"(A[10]), "r"(A[11]), 
		//       "r"(B[5]), 
		//       "f"(C[20]), "f"(C[21]), "f"(C[22]), "f"(C[23])
		// ); 
		// #endif
		// #if (ILPconfig >= 7)
		// asm volatile(
		//   "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
		//   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//   : "=f"(D[24]), "=f"(D[25]), "=f"(D[26]), "=f"(D[27])
		//   : "r"(A[12]), "r"(A[13]), 
		//     "r"(B[6]), 
		//     "f"(C[24]), "f"(C[25]), "f"(C[26]), "f"(C[27])
		// ); 
		// #endif
		// #if (ILPconfig >= 8)
		// asm volatile(
		//     "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
		//     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
		//     : "=f"(D[28]), "=f"(D[29]), "=f"(D[30]), "=f"(D[31])
		//     : "r"(A[14]), "r"(A[15]), 
		//       "r"(B[7]), 
		//       "f"(C[28]), "f"(C[29]), "f"(C[30]), "f"(C[31])
		// ); 
		// #endif
		
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
float tensor32816_max_flops(int THREADS_PER_BLOCK, bool report_fma_bw = false) {
	intilizeDeviceProp(0);
	int BLOCKS_NUM = 1;
	int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;
	int WARP_SIZE = 32;
  
	// 根据假设，A 和 B 矩阵的大小都为16*16
	unsigned total_A_SIZE = M * K * (TOTAL_THREADS / WARP_SIZE) * ILPconfig;
	unsigned total_B_SIZE = K * N * (TOTAL_THREADS / WARP_SIZE) * ILPconfig;
	unsigned total_R_SIZE = M * N * (TOTAL_THREADS / WARP_SIZE) * ILPconfig;
  
	uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
	T *data_a = (T *)malloc(total_A_SIZE * sizeof(T));
	T *data_b = (T *)malloc(total_B_SIZE * sizeof(T));
	R *res = (R *)malloc(total_R_SIZE * sizeof(R));
  
	uint64_t *startClk_ptr;
	uint64_t *stopClk_ptr;
	T *data_a_ptr;
	T *data_b_ptr;
	R *res_ptr;
  
	// 矩阵AB内，元素都是固定的从0~255的顺序数
	for (uint32_t i = 0; i < M * K; i++) { data_a[i] = (T)i; }
	for (uint32_t i = 0; i < K * N; i++) { data_b[i] = (T)i; }
	
	// 使用 cudaMalloc 在 GPU 内分配空间，地址赋予 ptr
	gpuErrchk(cudaMalloc(&startClk_ptr, TOTAL_THREADS * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&stopClk_ptr, TOTAL_THREADS * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&data_a_ptr, total_A_SIZE * sizeof(T)));
	gpuErrchk(cudaMalloc(&data_b_ptr, total_B_SIZE * sizeof(T)));
	gpuErrchk(cudaMalloc(&res_ptr, total_R_SIZE * sizeof(R)));
	// 将数据搬到上述 GPU 分配的空间
	gpuErrchk(cudaMemcpy(data_a_ptr, data_a, 
		total_A_SIZE * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data_b_ptr, data_b, 
		total_B_SIZE * sizeof(T), cudaMemcpyHostToDevice));
	// 给 mma 操作计时
	tensor32816_flops<T, R><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
		startClk_ptr, stopClk_ptr, data_a_ptr, data_b_ptr, res_ptr, 0);
	gpuErrchk(cudaPeekAtLastError());
	// 没有发生错误才将 时间数据 和 乘法结果 放入 GPU 内部
	gpuErrchk(cudaMemcpy(startClk, startClk_ptr, 
		TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(stopClk, stopClk_ptr, 
		TOTAL_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res, res_ptr, 
		total_R_SIZE * sizeof(R), cudaMemcpyDeviceToHost));
  
	float mma_bw, fma_bw;
	// 总耗时 = 最晚结束时间 - 最早开始时间
	uint64_t total_time =
		*std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
		*std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);

	// ? 不清楚此处是什么意思
	float fpuFMA = (float)(ITERS * TOTAL_THREADS * 1 * 1 * 1 * 0 ) /
		  ((float)total_time);  // max 64FMA/clk/SM on RTX3070Ti

	mma_bw = ((float)(ITERS * TOTAL_THREADS)) / (float)total_time;
	fma_bw = ((float)(ITERS * M * N * K * ILPconfig * (TOTAL_THREADS / WARP_SIZE))) 
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
		tensor32816_max_flops<__half, float>(32 * e); // 每个 warp 有32个线程
		std::cout << std::endl;
	}
	return 0;
}
  