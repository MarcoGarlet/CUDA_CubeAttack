#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>

//#include "cipher.cu"


/*
__device__ void flush_iv(uint8_t iv[]){
	for(int i=0;i<IV_dim;i++){
		iv[i]=0;
	}
}

void flush_iv_host(uint8_t iv[]){
	for(int i=0;i<IV_dim;i++){
		iv[i]=0;
	}
}
*/


/*

void IV_gen_host(uint64_t id, uint8_t len, uint8_t cube_index[],uint8_t *iv){
	uint8_t c_i;
	flush_iv_host(iv);
	for(unsigned int i=0; i<len;i++){
		c_i = ((id/(1ull<<i)))%2;
		//iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB_host(c_i * (1<<(cube_index[i]%8))); //Grain
		//iv[(cube_index[i]/8)] += to_MSB_host(c_i * (1<<(cube_index[i]%8))); //Trivium
		iv[(cube_index[i]/8)] += (c_i * (1<<((cube_index[i]%8)))); // Jambu
	}
	
}


__device__ void IV_gen(uint64_t tid, uint8_t len, uint8_t cube_index[],uint8_t *iv){
	uint8_t c_i;
	flush_iv(iv);
	for(unsigned int i=0; i<len;i++){
		c_i = ((tid/(1ull<<i)))%2;
		//iv[(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8)));
		//iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8)));

		iv[(cube_index[i]/8)] += (c_i * (1<<((cube_index[i]%8))));


	}
	
}
*/

__device__ bool unsigned8_element_in_array_gpu(uint8_t el, uint8_t *arr, uint64_t len){
	bool found = false;
	for(int i=0 ;i<len;i++){
		if(arr[i] == el){
			found = true;
			break;
		}
	}
	return found;
}


