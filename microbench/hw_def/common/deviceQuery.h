#ifndef DEVICE_QUERY_H
#define DEVICE_QUERY_H

#include <cuda_runtime.h>

/*
 * 简而言之，该头文件定义了各种 device 的参数数据
 * 并根据实际机器的属性，对各“常量”进行了赋值
*/
unsigned SM_NUMBER;           // number of SMs
unsigned WARP_SIZE;           // max threads per warp
unsigned MAX_THREADS_PER_SM;  // max threads / sm
unsigned MAX_SHARED_MEM_SIZE; // Max configerable shared memory size in bytes
unsigned MAX_WARPS_PER_SM;    // max warps / sm
unsigned MAX_REG_PER_SM;      // max warps / sm

unsigned MAX_THREAD_BLOCK_SIZE;         // max threads per threadblock
unsigned MAX_SHARED_MEM_SIZE_PER_BLOCK; // Max configerable shared memory size
                                        // per block in bytes
unsigned
    MAX_REG_PER_BLOCK; // Max configerable shared memory size per block in bytes

size_t L2_SIZE; // L2 size in bytes

size_t MEM_SIZE;            // Memory size in bytes
unsigned MEM_CLK_FREQUENCY; // Memory clock freq in MHZ
unsigned MEM_BITWIDTH;      // Memory bit width

// launched threadblocks
unsigned THREADS_PER_BLOCK;
unsigned BLOCKS_PER_SM;
unsigned THREADS_PER_SM;
unsigned BLOCKS_NUM;
unsigned TOTAL_THREADS;

// cudaDeviceProp 是一个结构体，其中定义了
// 支持CUDA版本号、内存大小、最大thread数目、执行单元的频率等基本信息
cudaDeviceProp deviceProp;

unsigned intilizeDeviceProp(unsigned deviceID) {
  cudaSetDevice(deviceID);
  // 该函数对应 cudaDeviceProp 结构体
  // 传入 deviceID，对 deviceProp 内部各成员进行赋值
  cudaGetDeviceProperties(&deviceProp, deviceID);

  // core stats
  SM_NUMBER = deviceProp.multiProcessorCount;
  MAX_THREADS_PER_SM = deviceProp.maxThreadsPerMultiProcessor;
  MAX_SHARED_MEM_SIZE = deviceProp.sharedMemPerMultiprocessor;
  WARP_SIZE = deviceProp.warpSize;
  MAX_WARPS_PER_SM =
      deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
  MAX_REG_PER_SM = deviceProp.regsPerMultiprocessor;

  // threadblock stats
  MAX_THREAD_BLOCK_SIZE = deviceProp.maxThreadsPerBlock;
  MAX_SHARED_MEM_SIZE_PER_BLOCK = deviceProp.sharedMemPerBlock;
  MAX_REG_PER_BLOCK = deviceProp.regsPerBlock;

  // launched thread blocks to ensure GPU is fully occupied as much as possible
  THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
  BLOCKS_PER_SM =
      deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
  THREADS_PER_SM = BLOCKS_PER_SM * THREADS_PER_BLOCK;
  BLOCKS_NUM = BLOCKS_PER_SM * SM_NUMBER;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  // L2 cache
  L2_SIZE = deviceProp.l2CacheSize;

  // memory
  MEM_SIZE = deviceProp.totalGlobalMem;
  MEM_CLK_FREQUENCY = deviceProp.memoryClockRate * 1e-3f;
  MEM_BITWIDTH = deviceProp.memoryBusWidth;

  return 1;
}

#endif