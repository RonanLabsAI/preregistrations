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
    std::puts("]}");ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaFree(out));
}
