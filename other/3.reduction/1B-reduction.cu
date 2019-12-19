/* asum: sum of all entries of a vector.
 * This code only calculates one block to show the usage of shared memory and synchronization */

#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;

/* sum all entries in x and asign to y */
__global__ void reduction_1(const FLOAT *x, FLOAT *y)
{
    __shared__ FLOAT sdata[256];
    int tid = threadIdx.x;

    /* load data to shared mem 
    共享内存是块内线程可见的,所以就有竞争问题的存在,也可以通过共享内存进行通信. 
    为了避免内存竞争,可以使用同步语句:void __syncthreads();语句相当于在线程
    块执行时各个线程的一个障碍点,当块内所有线程都执行到本障碍点的时候才能进行下
    一步的计算；但是,__syncthreads(); 频繁使用会影响内核执行效率。*/
    sdata[tid] = x[tid];//这个x是 FLOAT *x;
    __syncthreads();

    /* reduction using shared mem 把for循环展开*/
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0) {
        *y = sdata[0] + sdata[1];
    }
}


//课件中第五个算法
__global__ void reduction_2(const FLOAT *x, FLOAT *y)
{
    __shared__  FLOAT sdata[256];//加volatile关键字，避免编译器自己进行优化.
    int tid = threadIdx.x;

    sdata[tid] = x[tid];
    __syncthreads();

    if(tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if(tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    
    if(tid < 32) 
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    if(tid == 0)  y[0] =sdata[0];

}




//__device__ 只能在GPU上被调用
__device__ void warpReduce(volatile FLOAT *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduction_3(const FLOAT *x, FLOAT *y)
{
    __shared__ FLOAT sdata[256];
    int tid = threadIdx.x;

    /* load data to shared mem */
    sdata[tid] = x[tid];
    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) y[0] = sdata[0];
}

int main()
{
    int N = 256;   /* must be 256 */
    int nbytes = N * sizeof(FLOAT);

    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL;
    int i;
    FLOAT as = 0;

    /************** allocate GPU mem ***************/
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, sizeof(FLOAT));
    if (dx == NULL || dy == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }
    printf("allocated %e MB on GPU\n", nbytes / (1024.f * 1024.f));



    /**************** alllocate CPU mem ************/
    hx = (FLOAT *) malloc(nbytes);
    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %e MB on CPU\n", nbytes / (1024.f * 1024.f));



    /****************** init *********************/
    for (i = 0; i < N; i++) {
        hx[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    /* call GPU */
    reduction_1<<<1, N>>>(dx, dy);
    /* let GPU finish */
    cudaThreadSynchronize();
    /* copy data from GPU */
    cudaMemcpy(&as, dy, sizeof(FLOAT), cudaMemcpyDeviceToHost);
    printf("reduction_1, answer: 256, calculated by GPU:%g\n", as);


    /* call GPU */
    reduction_2<<<1, N>>>(dx, dy);
    /* let GPU finish */
    cudaThreadSynchronize();
    /* copy data from GPU */
    cudaMemcpy(&as, dy, sizeof(FLOAT), cudaMemcpyDeviceToHost);
    printf("reduction_2, answer: 256, calculated by GPU:%g\n", as);


    /* call GPU */
    reduction_3<<<1, N>>>(dx, dy);
    /* let GPU finish */
    cudaThreadSynchronize();
    /* copy data from GPU */
    cudaMemcpy(&as, dy, sizeof(FLOAT), cudaMemcpyDeviceToHost);
    printf("reduction_3, answer: 256, calculated by GPU:%g\n", as);


    cudaFree(dx);
    cudaFree(dy);
    free(hx);

    return 0;
}
