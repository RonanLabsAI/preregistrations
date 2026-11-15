// Isolated diagnostic ceiling for mma.sync.m16n8k16 f16 inputs and f32 accumulation.
// This changes no TensorFold code or model data and cannot establish a served speedup.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e));std::exit(2);} }
__global__ void probe(float* out, int iterations) {
    const unsigned a=0x38003800u;
    float d[4][4] = {};
    for (int n=0;n<iterations;++n) {
#pragma unroll
        for(int j=0;j<4;++j) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(d[j][0]),"+f"(d[j][1]),"+f"(d[j][2]),"+f"(d[j][3])
              : "r"(a),"r"(a),"r"(a),"r"(a),"r"(a),"r"(a));
        }
    }
    float s=0;
#pragma unroll
    for(int j=0;j<4;++j)
#pragma unroll
        for(int k=0;k<4;++k) s+=d[j][k];
    out[blockIdx.x*blockDim.x+threadIdx.x]=s;
}
__global__ void init_stream(float* a,float* b,float* c,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){a[i]=1;b[i]=2;c[i]=0;}
}
__global__ void triad(const float* a,const float* b,float* c,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)c[i]=a[i]+2*b[i];
}
int main() {
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));
    if(p.major!=10||p.minor!=3||p.multiProcessorCount!=152){std::fprintf(stderr,"wrong diagnostic device\n");return 3;}
    constexpr int threads=256,iterations=8192,repeats=10;
    int blocks=p.multiProcessorCount*8;size_t count=size_t(blocks)*threads;
    float* out=nullptr;ck(cudaMalloc(&out,count*sizeof(float)));
    probe<<<blocks,threads>>>(out,iterations);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<float> host(count);ck(cudaMemcpy(host.data(),out,count*sizeof(float),cudaMemcpyDeviceToHost));
    for(float v:host) if(v!=float(64*iterations)){std::fprintf(stderr,"MMA checksum failure\n");return 4;}
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    std::printf("{\"checksum_pass\":true,\"sms\":%d,\"threads\":%d,\"blocks\":%d,\"iterations\":%d,\"repeats\":%d,\"samples\":[",p.multiProcessorCount,threads,blocks,iterations,repeats);
    for(int trial=0;trial<5;++trial){
        ck(cudaEventRecord(a));
        for(int r=0;r<repeats;++r)probe<<<blocks,threads>>>(out,iterations);
        ck(cudaGetLastError());ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));
        float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
        double ops=double(blocks)*(threads/32)*iterations*4*4096*repeats;
        std::printf("%s{\"milliseconds\":%.9g,\"fp16_tflops\":%.9g}",trial?",":"",ms,ops/(double(ms)*1e9));
    }
    std::printf("],\"stream_triad_samples\":[");
    constexpr size_t n=size_t(1)<<29;
    float *x=nullptr,*y=nullptr,*z=nullptr;
    ck(cudaMalloc(&x,n*sizeof(float)));ck(cudaMalloc(&y,n*sizeof(float)));ck(cudaMalloc(&z,n*sizeof(float)));
    init_stream<<<blocks,threads>>>(x,y,z,n);triad<<<blocks,threads>>>(x,y,z,n);ck(cudaDeviceSynchronize());
    float first=0,last=0;ck(cudaMemcpy(&first,z,sizeof(float),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(&last,z+n-1,sizeof(float),cudaMemcpyDeviceToHost));
    if(first!=5||last!=5){std::fprintf(stderr,"triad checksum failure\n");return 5;}
    for(int trial=0;trial<5;++trial){
        ck(cudaEventRecord(a));for(int r=0;r<repeats;++r)triad<<<blocks,threads>>>(x,y,z,n);
        ck(cudaGetLastError());ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
        double bytes=double(n)*sizeof(float)*3*repeats;
        std::printf("%s{\"milliseconds\":%.9g,\"decimal_GB_s\":%.9g}",trial?",":"",ms,bytes/(double(ms)*1e6));
    }
    std::puts("]}");ck(cudaFree(x));ck(cudaFree(y));ck(cudaFree(z));
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaFree(out));
}
