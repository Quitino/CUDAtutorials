#include <stdio.h>
#include <cuda.h>  //头文件

typedef float FLOAT;
#define USE_UNIX 1

/* get thread id: 1D block and 2D grid ，blockDim.x 内置变量*/ 
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* warm up, start GPU, optional 先将GPU启动起来，因为实际运行的时候遇到GPU代码再启动会耽搁一点点时间，所以可以先将GPU启动起来 */
void warmup();

/* get time stamp */
double get_time(void);

/* host, add  CPU中运行的代码*/
void vec_add_host(FLOAT *x, FLOAT *y, FLOAT *z, int N);

/* device function GPU中运行的代码，__global__标志为GPU核函数，返回类型必须是void  */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 1D block */
    int idx = get_tid();

    if (idx < N) z[idx] = z[idx] + y[idx] + x[idx];
}

void vec_add_host(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    int i;

    for (i = 0; i < N; i++) z[i] = z[i] + y[i] + x[i];
}

/* a little system programming */
//系统兼容，不同系统求取时间戳
#if USE_UNIX
#include <sys/time.h>
#include <time.h>

double get_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}


#else
#include <windows.h>

double get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }

    QueryPerformanceCounter(&timer);

    t = timer.QuadPart * 1. / fre.QuadPart;

    return t;
}
#endif

/* warm up GPU */
__global__ void warmup_knl()
{
    int i, j;

    i = 1;
    j = 2;
    i = i + j;
}

void warmup()
{
    int i;

    for (i = 0; i < 8; i++) {
        warmup_knl<<<1, 256>>>();
    }
}

int main()
{
    int N = 20000000;//这么多个浮点数
    int nbytes = N * sizeof(FLOAT);

    /* 1D block */
    int bs = 256;

    /* 2D grid */
	//总共的线程数：N + bs - 1  ，再除以块数就是需要的网格数
	//ceil函数的作用是求不小于给定实数的最小整数，网格数只能多不能少
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);


	//申请内存，一个是GPU上的内存d(device),一个是CPU上的内存h(host)
    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL, *hy = NULL;
    FLOAT *dz = NULL, *hz = NULL;

    int itr = 30;
    int i;
    double th, td;

    /* warm up GPU */
    warmup();

    /* allocate GPU mem GPU上申请内存的方式*/
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    if (dx == NULL || dy == NULL || dz == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem CPU上申请内存*/
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    if (hx == NULL || hy == NULL || hz == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %.2f MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
        hz[i] = 1;
    }

    /* copy data to GPU ；把CPU上的数据拷贝到GPU*/
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

    /* warm up */
    warmup();

    /* call GPU */
    cudaThreadSynchronize();//这个函数是使GPU算完了再计算CPU上的，因为GPU和CPU是分开的，激活后就单独运行，CPU也继续运行
    td = get_time();
    
    for (i = 0; i < itr; i++) vec_add<<<grid, bs>>>(dx, dy, dz, N);

    cudaThreadSynchronize();
    td = get_time() - td;

    /* CPU */
    th = get_time();
    for (i = 0; i < itr; i++) vec_add_host(hx, hy, hz, N);
    th = get_time() - th;

    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

	//释放内存
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);

    return 0;
}
