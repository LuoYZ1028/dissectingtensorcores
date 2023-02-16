rm *.app

nvcc -g -G -Xptxas -dlcm=cv -Xptxas -dscm=wt -arch=sm_70 \
    --define-macro ILPconfig=1,ITERS=999,MEAN=0.0,STDDEV=1.0 \
    -gencode=arch=compute_70,code=\"sm_70,compute_70\"  wmma_m16n16k16_fp16.cu \
    -o wmma_m16n16k16_fp16.app -I /usr/local/cuda-11.0/samples/common/inc/ -L  -lcudart;
    
cuda-gdb