#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>
//#include "mygrain_lib.cuh"


#include "../../../cipher.cu"
#include "../../../common.h"
//#include "../../../mygrain_lib.cuh"
#include "../../../utils.cu"

#define TRIES	1
#define INSTANCES 1
#define CONSTANT 1
#define M_d 10
#define min_M_d 4
#define N 100 // N tests BLR 

__constant__ uint8_t d_k[K_dim];
__constant__ uint8_t k_curr_dev;





/*
  Kernel for selecting different random k
*/

__global__ void setup_kernel(curandState* state,uint64_t seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void random_k(uint8_t *out_k,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]); // The sequence number is another seed-like value. It is used so that, if all cores have the same seed, but different sequence numbers, then they will get different random values.
  
  out_k[tid] = curand_uniform(&states[tid])*(M_d-min_M_d)+min_M_d;
}

__global__ void random_I(uint8_t *out_I,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]);
  out_I[tid] = curand_uniform(&states[tid])*(BIT_I);
}

__global__ void random_I_unique(uint8_t *out_I,uint8_t *out_k,curandState *states){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t start = 0;
	uint8_t el;
	curand_init(clock64(), tid, 0, &states[tid]);

	for(uint64_t i=0;i<tid;i++)
		start+=out_k[i];
	for(uint64_t i=0;i<out_k[tid];i++){
		el = curand_uniform(&states[tid])*(BIT_I);
		while(unsigned8_element_in_array_gpu(el,&out_I[start],out_k[tid])){
			//curand_init(clock64(), tid, 0, &states[tid]);
			el =curand_uniform(&states[tid])*(BIT_I);
		}
		out_I[start+i] = el;
	}	
}

__global__ void generate_IV(uint8_t *cube,uint8_t *I_iv,uint8_t len){
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // each tid map I_i
  IV_gen(tid,len,cube,&I_iv[tid*IV_dim]); // uint64_t tid, uint8_t len, uint8_t cube_index[],uint8_t *iv
}




__global__ void init_random_key(curandState *states){
	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
 	curand_init(clock64(), idx, 0, &states[idx]);
}

__global__ void generate_key_set(uint8_t *k,curandState *states){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    k[tid] = curand_uniform(&states[tid])*0xFF;
}

__global__ void generate_key_set_xor(uint8_t *k1_set, uint8_t *k2_set, uint8_t *k_xor_set){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    k_xor_set[tid] = k1_set[tid]^k2_set[tid];
}

__global__ void cuda_encrypt(uint8_t *I,uint8_t *out){
	uint8_t IV[IV_dim];
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	IV_gen(tid,k_curr_dev,I,IV); 
	//out[tid] = encrypt_unroll(IV,d_k);
	out[tid] = encrypt(IV,d_k);
}

/// Interleaved Pair Implementation with less divergence
__global__ void sumCubeReduceInterleaved (uint8_t *g_idata, uint8_t *g_odata, unsigned int n) {
	// set thread ID
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local pointer of this block 
	uint8_t *idata = g_idata + blockIdx.x * blockDim.x;
	// boundary check if(idx >= n) return;
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			idata[tid] ^= idata[tid + stride];
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}


__global__ void sumCubeCompleteUnroll (uint8_t *g_idata, uint8_t *g_odata,
	unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

// convert global data pointer to the local pointer of this block
uint8_t *idata = g_idata + blockIdx.x * blockDim.x * 8;

// unrolling 8
if (idx + 7 * blockDim.x < n)
{
	uint8_t a1 = g_idata[idx];
	uint8_t a2 = g_idata[idx + blockDim.x];
	uint8_t a3 = g_idata[idx + 2 * blockDim.x];
	uint8_t a4 = g_idata[idx + 3 * blockDim.x];
	uint8_t b1 = g_idata[idx + 4 * blockDim.x];
	uint8_t b2 = g_idata[idx + 5 * blockDim.x];
	uint8_t b3 = g_idata[idx + 6 * blockDim.x];
	uint8_t b4 = g_idata[idx + 7 * blockDim.x];
	g_idata[idx] = a1 ^ a2 ^ a3 ^ a4 ^ b1 ^ b2 ^ b3 ^ b4;
}

__syncthreads();

// in-place reduction and complete unroll
if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

__syncthreads();

if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

__syncthreads();

if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

__syncthreads();

if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

__syncthreads();

// unrolling warp
if (tid < 32)
{
	volatile uint8_t *vsmem = idata;
	vsmem[tid] ^= vsmem[tid + 32];
	vsmem[tid] ^= vsmem[tid + 16];
	vsmem[tid] ^= vsmem[tid +  8];
	vsmem[tid] ^= vsmem[tid +  4];
	vsmem[tid] ^= vsmem[tid +  2];
	vsmem[tid] ^= vsmem[tid +  1];
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}



void flush_arr(uint8_t *arr, uint64_t len){
	for(uint64_t i=0; i<len; i++)
		arr[i]=0;
}




void print_arr_host(const char *s, uint8_t *k, unsigned int dim){
  printf("\n%s = [",s);
  for(int i=0; i<dim;i++){
	printf(" %u, ",k[i]);
  }
  printf("]\n");
}

void print_arr_host_u64(const char *s, uint64_t *k, unsigned int dim){
    printf("\n%s = [",s);
    for(int i=0; i<dim;i++){
      printf(" %lu, ",k[i]);
    }
    printf("]\n");
  }

void print_arr_IV_host(const char *s, uint8_t *k, unsigned int dim){
	for(int i=0; i<dim;i++){
	  if(i%12==0){
		  if(i!=0) 
		    printf(" ]");
		  printf("\n%s = [",s);
	  }
	  printf(" %u, ",k[i]);
	}
	printf("]\n");
  }

uint64_t sum_k(uint8_t *k, unsigned int dim){
	uint64_t sum=0;
	for(int i=0;i<dim;i++)
		sum+=k[i];
	return sum;
}

uint64_t get_nbytes_IV(uint8_t *k, unsigned int dim){
	uint64_t sum = 0;
	for(int i=0;i<dim;i++)
		sum+=1<<k[i];
	return sum;
} 

void copy_data_I(uint8_t *src, uint8_t *dst, unsigned int dim){
	for(int i=0;i<dim;i++)
		dst[i] = src[i];
}

void random_key(uint8_t k[]){
	unsigned int h;
	for(int i=0;i<K_dim;i++){
		h = 8;
		if (i==K_dim-1)
			h = BIT_K - i*8;
		k[i]=rand()%(1<<h);
	}
}

void xor_key_host(uint8_t k_0[],uint8_t k_1[],uint8_t k_xor[]){
	for(int i=0;i<K_dim;i++)
		k_xor[i] = k_0[i]^k_1[i];
}




void print_IV_host(uint8_t iv[]){
	printf("\n IV vector = [ ");
	for(int i=0;i<IV_dim;i++){
		printf("%u ",iv[i]);
	}
	printf("]\n");
}

void print_key_host(uint8_t k[]){
	printf("\n K vector = [ ");
	for(int i=0;i<K_dim;i++){
		printf("%u ",k[i]);
	}
	printf("]\n");
}

void copy_IV_host(uint8_t* dst, uint8_t* src){
	for(int i=0; i<IV_dim; i++)
		dst[i]=src[i];
}

bool equal_IV(uint8_t *iv,uint8_t *iv1){

	bool r=true;
	for(int i=0; i<IV_dim; i++){
		if(iv[i]!=iv1[i]){
			r = false;
			break;
		}
	}
	return r; 
}


bool signed64_element_in_array(int64_t el, int64_t *arr, uint64_t len){
	bool found = false;
	for(int i=0 ;i<len;i++){
		if(arr[i] == el){
			found = true;
			break;
		}
	}
	return found;
}

bool unsigned64_element_in_array(uint64_t el, uint64_t *arr, uint64_t len){
	bool found = false;
	for(int i=0 ;i<len;i++){
		if(arr[i] == el){
			found = true;
			break;
		}
	}
	return found;
}


bool unsigned8_element_in_array(uint8_t el, uint8_t *arr, uint64_t len){
	bool found = false;
	for(int i=0 ;i<len;i++){
		if(arr[i] == el){
			found = true;
			break;
		}
	}
	return found;
}

uint64_t get_IV_line(uint8_t *iv, uint8_t *h_IV_host,uint64_t dim,int64_t *excluded,int64_t c){
	int64_t r = -1;
	for(uint64_t i=0;i<dim;i++){
		if(!signed64_element_in_array(i,excluded,c) && equal_IV(iv, &h_IV_host[i*IV_dim])){
			r = i;
			break;
		}
	}
	return r;
}




bool check_kernel3_results(uint8_t *h_IV, uint8_t *h_IV_host, uint64_t dim){
	int64_t j,c=0;
	bool r = true;
	int64_t *excluded = (int64_t *)malloc(dim*sizeof(int64_t));

	for(int i=0;i<dim;i++)
		excluded[i]=-1;
	for(int i=0;i<dim;i++){
		j = get_IV_line(&h_IV[i*IV_dim],h_IV_host,dim,excluded,c);
		if(j == -1){
			printf("\nNOT FOUND\n");
			print_IV_host(&h_IV[i*IV_dim]);
			r = false;
			break;
		}

		excluded[c] = j;
		c+=1;
	}
	return r;

}




uint8_t compute_cube(uint8_t *k,uint8_t *dev_I,uint8_t *out_cube, uint8_t *reduce_out,cudaStream_t si,uint64_t length){

	uint8_t *hreduce_out,gpu_sum;
	uint64_t blocksize;

	// check blocksize for given length, the length is 2**k

	if(length <  1073741824){  // 2 **30 limit for grid block
		if(length/33554431 > 0) // 2*25
			blocksize = 1024;
		else	
			blocksize=32;
	}
	else{
		printf("\ncube len not supported for 1d grid and 1 d block\n");
		exit(1);
	}
	blocksize = (length<blocksize)? length:blocksize;
	dim3 block(blocksize, 1);
	dim3 grid((length+ block.x - 1) / block.x, 1);
	hreduce_out =(uint8_t *)malloc(sizeof(uint8_t)*grid.x);

	cudaMemcpyToSymbolAsync(d_k,k, sizeof(uint8_t)*K_dim,0,cudaMemcpyHostToDevice,si);
	CHECK(cudaStreamSynchronize(si));


	cuda_encrypt<<<grid,block,0,si>>>(dev_I,out_cube);
	CHECK(cudaStreamSynchronize(si));

	sumCubeReduceInterleaved<<<grid,block,0,si>>>(out_cube,reduce_out,length);

	CHECK(cudaStreamSynchronize(si));

	CHECK(cudaMemcpyAsync(hreduce_out, reduce_out, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost,si));
	CHECK(cudaStreamSynchronize(si));

	gpu_sum = 0;
	

	for (int i = 0; i < grid.x; i++){
		gpu_sum ^= hreduce_out[i];

	} 

	
	free(hreduce_out);
	return gpu_sum;
}


uint8_t compute_cube_host(uint8_t *k,uint8_t *h_I,unsigned int length, uint8_t k_len){


	uint8_t gpu_sum=0,c;
	uint8_t IV[IV_dim];

	for(uint64_t i=0;i<length;i++){
		
	
		IV_gen_host(i,IV_dim,h_I,IV); 
		//print_arr_host("IV ",IV,IV_dim);
		//print_arr_host("k ",k,K_dim);
		c = encrypt_host(IV,k);
		//printf("\nc = %lu\n",c);
		gpu_sum = (gpu_sum+c)%2;
	}

	return gpu_sum;
}

uint64_t update_length_cubes( uint8_t *out_k_host,uint64_t *del1,uint64_t *add1,uint64_t c,uint64_t del_1_counter,uint64_t add_1_counter){
	uint64_t dim=0;
	for(uint64_t i=0;i<c;i++){
		
		// check if we have to add element in cube
		if(unsigned64_element_in_array(i,add1,add_1_counter))
			dim+=out_k_host[i]+1;

		// check if we have to remove element in cube
		if(unsigned64_element_in_array(i,del1,del_1_counter))
			dim+=out_k_host[i]-1;
	}
	return dim;
}

uint8_t random_I_element(uint8_t *excluded,uint64_t dim){
	uint8_t element = rand()%BIT_I;
	while(unsigned8_element_in_array(element,excluded,dim))
		element = rand()%BIT_I;
	return element;
}

void random_I_unique_host(uint8_t *out_I,uint8_t *out_k,uint64_t k){
	uint64_t start = 0;
	uint8_t el;
	for(uint64_t i=0;i<k;i++){
	
		for(uint64_t i1=0;i1<out_k[i];i1++){
			el = rand()%BIT_I;
			while(unsigned8_element_in_array(el,&out_I[start],i1)){
				el = rand()%BIT_I;
				//printf("\nel = %lu\n",el);
			}
			out_I[start+i1] = el;
		}
    	//print_arr_host("IV ",&out_I[start], out_k[i]); 
		start+=out_k[i];
	}

}


void expand_cube(uint8_t *out_I_host_new,uint8_t *out_I_host,uint64_t dim){
	for(int i=0;i<dim;i++)
		out_I_host_new[i]=out_I_host[i];
	out_I_host_new[dim] = random_I_element(out_I_host,dim);
}


void reduce_cube(uint8_t *out_I_host_new,uint8_t *out_I_host,uint64_t dim){
	uint64_t pos = rand()%dim;
	uint8_t t;
	t = out_I_host[dim-1];
	out_I_host[dim-1] = out_I_host[pos];
	out_I_host[pos] = t;
	for(int i=0;i<dim-1;i++)
		out_I_host_new[i]=out_I_host[i];
}

void print_cube(uint8_t *out_I_host,uint64_t dim){
    printf("\n cube [ ");
    for(int index=0;index<dim;index++){
        printf("%u, ",out_I_host[index]);
    }
    printf("]\n");
}

//update_out_k_host(out_k_host,del1,add1,del_1_counter,add_1_counter);
uint8_t **update_out_k_host(uint8_t *out_I_host, uint8_t *out_k_host,uint64_t *del1,uint64_t *add1,uint64_t del_1_counter,uint64_t add_1_counter, uint64_t c){
  
	uint64_t new_cube_d=0,start=0,start_new=0,k_dim_new = update_length_cubes(out_k_host,del1,add1,c,del_1_counter,add_1_counter);
	printf("\nk_dim_new = %ld\n",k_dim_new);
	uint8_t *out_I_host_new = (uint8_t *)malloc(k_dim_new*sizeof(uint8_t)), *out_k_host_new = (uint8_t *)malloc((add_1_counter+del_1_counter)*sizeof(uint8_t));
	uint8_t **res = (uint8_t **)malloc(sizeof(uint8_t *)*2);

	for(uint64_t d=0; d<c; d++){
		
		if(unsigned64_element_in_array(d,add1,add_1_counter)){
			// check if we have to add element in cube


            if(out_k_host[d]<IV_dim*8){
				out_k_host_new[new_cube_d] = out_k_host[d]+1;
            	printf("\nBefore add element in cube\n");
            	print_cube(&out_I_host[start],out_k_host[d]);
				expand_cube(&out_I_host_new[start_new],&out_I_host[start],out_k_host[d]);
            	printf("\nAfter add element in cube\n");
            	print_cube(&out_I_host_new[start_new],out_k_host_new[new_cube_d]);
				start_new+=out_k_host_new[new_cube_d];
				new_cube_d+=1;
			}
			else	
				printf("\nCube to huge to expand\n");
          
		}
		if(unsigned64_element_in_array(d,del1,del_1_counter)){
			// check if we have to remove element in cube


			out_k_host_new[new_cube_d] = out_k_host[d]-1;
            printf("\nBefore remove element in cube\n");
            print_cube(&out_I_host[start],out_k_host[d]);
			reduce_cube(&out_I_host_new[start_new],&out_I_host[start],out_k_host[d]);
            printf("\nAfter remove element in cube\n");
            print_cube(&out_I_host_new[start_new],out_k_host_new[new_cube_d]);
			start_new+=out_k_host_new[new_cube_d];
			new_cube_d+=1;
		}
		//update new out_k_host and out_I_host
		start+=out_k_host[d];
  	}
	free(out_k_host);
	free(out_I_host);


	res[0] = out_k_host_new;
	res[1] = out_I_host_new;
	return res;                                                               
}



void fprint_cube(FILE *fp, uint8_t *out_I_host,uint64_t dim){
    char pre[3]= ", ";
    for(int index=0;index<dim;index++){
        if(index==dim-1){
            pre[0]='\n';
            pre[1]='\0';
        }
        fprintf(fp,"%u%s",out_I_host[index],pre);
    }

}

int main(){
	


	// INIT seed for random function
	time_t t;
	srand((unsigned) time(&t));

	//INSTANCES
    unsigned int blocksize = 1;
	dim3 block(blocksize, 1);
	dim3 grid((INSTANCES+ block.x - 1) / block.x, 1);
	printf("\nExecution Configure (block %d grid %d) for k sizes I Kernel:\n", block.x, grid.x);

	/*finding maxterms exploiting parallelism*/
	curandState *devStates;
    //setup_kernel <<< block,grid >>> (devStates,time(NULL));
	//CHECK(cudaDeviceSynchronize());


 	uint8_t *out_k,*out_I,*out_k_host,*out_I_host;

	//INSTANCES
    out_k_host = (uint8_t *)malloc(INSTANCES*sizeof(uint8_t));
	CHECK(cudaMalloc((uint8_t **)&out_k,INSTANCES*sizeof(uint8_t)));
	CHECK(cudaMalloc((void **)&devStates, INSTANCES*sizeof(curandState))); 
	printf("\n FIRST KERNEL CALL - k-sets \n");
  	random_k<<<block,grid>>>(out_k,devStates);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_k_host,out_k,INSTANCES*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    print_arr_host("k dim",out_k_host, INSTANCES); 
	CHECK(cudaFree(out_k));
	CHECK(cudaFree(devStates));	
	
	//out_k_host[0]=6;
	/*Kernel 2*/

	// INSTANCES vectors of M_d length
	uint64_t k_sum = sum_k(out_k_host,INSTANCES);

	uint64_t data_len = k_sum;
	printf("\n SECOND KERNEL CALL - I GEN\ndata_len= %lu\n",data_len);
	blocksize=1;
	block.x = blocksize;
	grid.x = (INSTANCES+ block.x - 1) / block.x;
	printf("\nblock=%u, grid=%u\n",block.x,grid.x);
	
	printf("\nTotal amount of bit positions = %lu\n",k_sum);

	out_I_host = (uint8_t *)malloc(k_sum*(sizeof(uint8_t)));

	//CHECK(cudaMalloc((void **)&devStates, k_sum*sizeof(curandState))); 
	//CHECK(cudaMalloc((uint8_t **)&out_k,INSTANCES*sizeof(uint8_t)));
	//CHECK(cudaMalloc((uint8_t **)&out_I,k_sum*sizeof(uint8_t)));

    //CHECK(cudaMemcpy(out_k,out_k_host,INSTANCES*sizeof(uint8_t),cudaMemcpyHostToDevice));


	//random_I_unique<<<block,grid>>>(out_I,out_k,devStates);
	//CHECK(cudaDeviceSynchronize());
	printf("\nBefore random I\n");
	random_I_unique_host(out_I_host,out_k_host,INSTANCES);
    //CHECK(cudaMemcpy(out_I_host,out_I,k_sum*sizeof(uint8_t),cudaMemcpyDeviceToHost));
	//CHECK(cudaFree(out_k));
	//CHECK(cudaFree(out_I));
	//CHECK(cudaFree(devStates));	

	//out_I_host = (uint8_t *)malloc(k_sum*(sizeof(uint8_t)));

    print_arr_host("I sets",out_I_host, k_sum); 


	uint64_t n_bytes_IV = get_nbytes_IV(out_k_host,INSTANCES);


	/*Kernel 4*/
   
	/*When using stream, you got to allocate large chunk of pinned memory and then play with offset inside each stream*/




	


	/*kernel 4: random key generator*/
	uint8_t p0,p1,p2,p1_2,p0_host,p1_host,p2_host,p1_2_host;
	uint64_t del_1[INSTANCES],add_1[INSTANCES],ACTIVE_INSTANCES=INSTANCES,start=0,start_I=0,n0,n1;
	uint64_t del_1_counter=0,add_1_counter=0,sol_dim=0,sol_count=0,*sol_dim_arr=NULL;
	uint8_t k0[K_dim*sizeof(uint8_t)]={0x0};
	uint8_t k1[K_dim],k2[K_dim],kx[K_dim];
	uint8_t *out_cube, *reduce_out,**updated_cubes,*maxterms=NULL;
	
	cudaStream_t stream[N*CONSTANT];
	
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();



	for(int t=0; t<TRIES; t++){
		n0=0,n1=0,start=0,start_I=0,del_1_counter=0,add_1_counter=0;
		k_sum = sum_k(out_k_host,ACTIVE_INSTANCES);
		data_len = 0;
		/*
			Move inside T loop and recompute all cubes in IV
		*/

		//CHECK(cudaMalloc((void **)&dev_IV,n_bytes_IV*(sizeof(uint8_t)*IV_dim)));
		CHECK(cudaMalloc((void **)&out_cube,n_bytes_IV*(sizeof(uint8_t))));
		CHECK(cudaMalloc((void **)&reduce_out,n_bytes_IV*(sizeof(uint8_t))));
		CHECK(cudaMalloc((void **)&out_I,k_sum*(sizeof(uint8_t))));
		CHECK(cudaMemcpy(out_I,out_I_host,k_sum*sizeof(uint8_t),cudaMemcpyHostToDevice));

		/*
			Update ACTIVE_INSTANCES with del_1,add_1
		*/

		for(int c=0; c<ACTIVE_INSTANCES; c++){

			printf("\n##### NEW CUBE %d #####\n",c);
			
			cudaMemcpyToSymbol(*(&k_curr_dev),&out_k_host[c], sizeof(uint8_t));

			data_len = (uint64_t)1<<(uint64_t)out_k_host[c];
			n0=0,n1=0,p0=0,p1=0,p2=0,p1_2=0;
			// memcopy for each kernel call is the same

			//CHECK(cudaMemcpyAsync(&dev_IV[start],&h_IV[start],data_len*sizeof(uint8_t)*IV_dim,cudaMemcpyHostToDevice));


			print_cube(&out_I_host[start_I],out_k_host[c]);


			for (int i = 0; i < N*CONSTANT; ++i) 
    			CHECK(cudaStreamCreate(&stream[i]));

			for(int i=0;i<N*CONSTANT;i++){

				printf("\rStream %d",i);
				fflush(stdout);
				

				random_key(k1);
				random_key(k2);
				//print_arr_host("K1: ",k1,K_dim);
				//print_arr_host("K2: ",k2,K_dim);

				xor_key_host(k1,k2,kx);
				//print_arr_host("Kx: ",kx,K_dim);

		
				

				
				if(i==0){
					p0 = compute_cube(k0,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);
					p0_host = compute_cube_host(k0,&out_I_host[start_I],data_len,out_k_host[c]);
					n1+=(p0==1);
					n0+=(p0==0);

				}

				p1 = compute_cube(k1,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);
				p2 = compute_cube(k2,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);
				p1_2 = compute_cube(kx,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);


				n1+=(p1==1)+(p2==1);
				n0+=(p1==0)+(p2==0);

				
				//	Introduce host check
				
				
				p1_host = compute_cube_host(k1,&out_I_host[start_I],data_len,out_k_host[c]);
				p2_host = compute_cube_host(k2,&out_I_host[start_I],data_len,out_k_host[c]);
				p1_2_host = compute_cube_host(kx,&out_I_host[start_I],data_len,out_k_host[c]);
				
				if (!(p0_host == p0 && p1_host == p1 && p2_host == p2 && p1_2_host == p1_2)){
					printf("\nKernel 4: The output is NOT correct\n");
					printf("\nDEVICE: p0 = %d, p1 = %d, p2 = %d, p1_2 = %d\nHOST: p0 = %d, p1 = %d, p2 = %d, p1_2=%d\n",p0,p1,p2,p1_2,p0_host,p1_host,p2_host,p1_2_host);
					exit(1);
				}
				
					
				
				
				// print output bit kernel 4
					
				

				/*Linearity test*/
				//if (p1 ^ p2 != p1_2 && p0 ^ p1 ^ p2 != p1_2){
				if(p0 ^ p1 ^ p2 != p1_2){
					//add to I set a random term an break 
					add_1[add_1_counter] = c; // save the cube to be reduced
					add_1_counter+=1;
					printf("\nBREAK fail BLR TESTS\n");
					break;
				}
			}
			for(int i=0; i<N*CONSTANT; ++i)
				cudaStreamSynchronize(stream[i]);
			for(int i=0; i<N*CONSTANT; ++i)
				cudaStreamDestroy(stream[i]);
			// remove term from I
			if (n0+n1>(2*N*CONSTANT)+1){
				printf("\nError BLR\n");
				exit(1);
			}
			printf("\ncube %d => n0 = %lu n1 = %lu\n",c,n0,n1);
			if(!unsigned64_element_in_array(c,add_1,add_1_counter)){
				if(n0 == (2*N*CONSTANT)+1 || n1 == (2*N*CONSTANT)+1){
					if(out_k_host[c]!=1){
						printf("\nReduce cube\n");
						del_1[del_1_counter] = c;
						del_1_counter+=1;
					}
					else
						printf("\nReduce cube but dim == 1\n");
				}
					
				else{
				
					// the term will not be added to add_1 list and del_1 list
					// add I set of the cube to sol list
					//	maxterms = (uint8_t *)realloc();
					printf("\nCube %d in solution\n",c);

					sol_dim+=out_k_host[c];
					sol_count+=1;

					sol_dim_arr = (uint64_t *)realloc(sol_dim_arr,sol_count*sizeof(uint64_t));
					sol_dim_arr[sol_count-1] = out_k_host[c];

					print_cube(&out_I_host[start_I],out_k_host[c]);

                    maxterms = (uint8_t *)realloc(maxterms,sol_dim*sizeof(uint8_t));
					copy_data_I(&out_I_host[start_I],&maxterms[sol_dim-out_k_host[c]],out_k_host[c]);
				}
			}
			start+=data_len;
			start_I+=out_k_host[c];
			/*store valid I sets for superpolys recompute I to be done*/

			
		}

		/*
			Update out_k_start
			Update out_cube

			free all old memory
		*/

		//CHECK(cudaFree(dev_IV));
		CHECK(cudaFree(out_cube));
		CHECK(cudaFree(reduce_out));
		
		/*Kernel 3*/
		/*
			For set_cubes method you must provide out_k_host, 
			which contail array of k dimension choosen randomly.
		*/
		
		if (del_1_counter+add_1_counter == 0){
			printf("\nNo active INSTANCES CUBES found\n");
			break;
		}

		updated_cubes = update_out_k_host(out_I_host,out_k_host,del_1,add_1,del_1_counter,add_1_counter,ACTIVE_INSTANCES);
		out_k_host = updated_cubes[0];
		out_I_host = updated_cubes[1];
		ACTIVE_INSTANCES = del_1_counter+add_1_counter;
       
		k_sum = sum_k(out_k_host,ACTIVE_INSTANCES);
		n_bytes_IV = get_nbytes_IV(out_k_host,ACTIVE_INSTANCES);
		
		// result in h_IV
	
		
		
		// compare h_IV_host with h_IV for validation
		
		

	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);

	printf("\nM_d = %d, time execution = %lf\n",M_d,cpu_ElapseTime);

    printf("\n############ SOLUTIONS ############\n");
    //FILE *fp = fopen("./final_attack/offline/cubes_test_window.txt","w");
	FILE *fp = fopen("./final_attack/offline/cubes_test_window.txt","w");
    uint64_t mt_start=0;
    for (int mt = 0; mt<sol_count;mt++){
        fprint_cube(fp,&maxterms[mt_start],sol_dim_arr[mt]);
        mt_start+=sol_dim_arr[mt];
    }
    print_arr_host("MAXTERMS BITS",maxterms,sol_dim);
    print_arr_host_u64("MAXTERMS LEN BITS",sol_dim_arr,sol_count);

	CHECK(cudaDeviceReset());



	return 0;
	
}
