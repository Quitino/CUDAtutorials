#include <stdio.h>
#include <cuda_runtime.h>



__global__ void histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int idx = threadIdx.x;
    //int iterms = sizeof(d_in);
    volatile __shared__ int sdata[128];
    __shared__ int tmpbin[24];// 8个线程的 bin

    sdata[idx + 0] = d_in[idx + 0]; ++tmpbin[sdata[idx + 0]%BIN_COUNT + 0] ;
    sdata[idx + 8] = d_in[idx + 8]; ++tmpbin[sdata[idx + 8]%BIN_COUNT + 0] ;
    sdata[idx + 16] = d_in[idx + 16]; ++tmpbin[sdata[idx + 16]%BIN_COUNT + 3] ;
    sdata[idx + 24] = d_in[idx + 24]; ++tmpbin[sdata[idx + 24]%BIN_COUNT + 3] ;
    sdata[idx + 32] = d_in[idx + 32]; ++tmpbin[sdata[idx + 32]%BIN_COUNT + 6] ;
    sdata[idx + 40] = d_in[idx + 40]; ++tmpbin[sdata[idx + 40]%BIN_COUNT + 6] ;
    sdata[idx + 48] = d_in[idx + 48]; ++tmpbin[sdata[idx + 48]%BIN_COUNT + 9] ;
    sdata[idx + 56] = d_in[idx + 56]; ++tmpbin[sdata[idx + 56]%BIN_COUNT + 9] ;
    sdata[idx + 64] = d_in[idx + 64]; ++tmpbin[sdata[idx + 64]%BIN_COUNT + 12] ;
    sdata[idx + 72] = d_in[idx + 72]; ++tmpbin[sdata[idx + 72]%BIN_COUNT + 12] ;
    sdata[idx + 80] = d_in[idx + 80]; ++tmpbin[sdata[idx + 80]%BIN_COUNT + 15] ;
    sdata[idx + 88] = d_in[idx + 88]; ++tmpbin[sdata[idx + 88]%BIN_COUNT + 15] ;
    sdata[idx + 96] = d_in[idx + 96]; ++tmpbin[sdata[idx + 96]%BIN_COUNT + 18] ;
    sdata[idx + 104] = d_in[idx + 104]; ++tmpbin[sdata[idx + 104]%BIN_COUNT + 18] ;
    sdata[idx + 112] = d_in[idx + 112]; ++tmpbin[sdata[idx + 112]%BIN_COUNT + 21] ;
    sdata[idx + 120] = d_in[idx + 120]; ++tmpbin[sdata[idx + 120]%BIN_COUNT + 21] ;

    __syncthreads();
/*
    if(idx > 1) 
    {
        d_bins[0] += tmpbin[idx * 3 - 0];
        d_bins[1] += tmpbin[idx * 3 - 2];
        d_bins[2] += tmpbin[idx * 3 - 1]; 
    } 
    if(idx == 0) d_bins[0] + tmpbin[0];
*/
    for(int i = 0;i < 24; ++i)
    {
        d_bins[0] += tmpbin[i%3];
        d_bins[1] += tmpbin[i%3];
        d_bins[2] += tmpbin[i%3];
    }
}


__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);//原子锁
}



int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 128;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = 3;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);

    // generate the input array on the host   0~127
    int h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = i;
    }
    int h_bins[BIN_COUNT];
    for(int i = 0; i < BIN_COUNT; i++) {
        h_bins[i] = 0;
    }

    // declare GPU memory pointers
    int * d_in;
    int * d_bins;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_bins, BIN_BYTES);

    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice); 

    histo<<<1, ARRAY_SIZE/16>>>(d_bins, d_in, BIN_COUNT);
    //simple_histo<<<1, 128>>>(d_bins, d_in, BIN_COUNT);
    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < BIN_COUNT; i++) {
        printf("bin %d: count %d\n", i, h_bins[i]);
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_bins);
        
    return 0;
}
