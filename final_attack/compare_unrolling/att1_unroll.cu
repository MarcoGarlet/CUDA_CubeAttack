#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>
#include "../trivium_unroll.cu"

#include "../common.h"

#define TRIES	5
#define INSTANCES 10

#define M_d 5

#define N 1000 // N tests BLR 

__constant__ uint8_t d_k[K_dim];



__global__ void cuda_encrypt(uint8_t *I_iv,uint8_t *out){
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	out[tid] = encrypt(&I_iv[tid*IV_dim],d_k);
}



int main(){
	uint8_t *dev_IV,*dev_out,*out;
	unsigned int len=1024;
	uint8_t k[K_dim]={0,1,2,3,4,5,6,7,8,9};
	out = (uint8_t *)malloc(len*sizeof(int8_t));; 
	CHECK(cudaMalloc((void **)&dev_IV,len*IV_dim*sizeof(uint8_t)));
	CHECK(cudaMalloc((void **)&dev_out,len*sizeof(uint8_t)));
	CHECK(cudaMemset(dev_IV,0,len*IV_dim*sizeof(uint8_t)));
	CHECK(cudaMemcpyToSymbol(d_k,k, sizeof(uint8_t)*K_dim));
	int blocksize=32;
	dim3 block(blocksize, 1);
	dim3 grid((len+ block.x - 1) / block.x, 1);
	cuda_encrypt<<<grid,block>>>(dev_IV,dev_out);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(out, dev_out, len  * sizeof(uint8_t),cudaMemcpyDeviceToHost));
	for(int i=0;i<len;i++){
		printf("\nout[%d] = %d\n",i,out[i]);
		break;
	}




	
	return 0;
}

