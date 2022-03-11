#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>
//#include "mygrain_lib.cuh"
#include "../../../../cipher.cu"
//#include "../../../../mygrain_lib.cuh"

#include "../../../../utils.cu"

#include "../../../../common.h"


#define CONSTANT 1
#define min_M_d 3

unsigned int TRIES;
unsigned int INSTANCES;
unsigned int M_d;

unsigned int N;



__constant__ uint8_t d_k[K_dim];
__constant__ uint8_t d_p0_dev;
__constant__ uint8_t M_d_dev;
__constant__ uint8_t k_curr_dev;



/*
  Kernel for selecting different random k
*/

__global__ void random_k(uint8_t *out_k,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]); // The sequence number is another seed-like value. It is used so that, if all cores have the same seed, but different sequence numbers, then they will get different random values.
  
  out_k[tid] = curand_uniform(&states[tid])*(M_d_dev-min_M_d)+min_M_d;
  
}

__global__ void random_I(uint8_t *out_I,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]);
  out_I[tid] = curand_uniform(&states[tid])*(IV_dim*8);
}

__global__ void random_I_unique(uint8_t *out_I,uint8_t *out_k,curandState *states){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64(), tid, 0, &states[tid]);
	uint64_t start = 0;
	uint8_t el;
	for(uint64_t i=0;i<tid;i++)
		start+=out_k[i];
	for(uint64_t i=0;i<out_k[tid];i++){
		el = curand_uniform(&states[tid])*(IV_dim*8);
		while(unsigned8_element_in_array_gpu(el,&out_I[start],out_k[tid])){
			el = curand_uniform(&states[tid])*(IV_dim*8);
		}
		out_I[start+i] = el;
	}	
}

__global__ void generate_IV(uint8_t *cube,uint8_t *I_iv,uint8_t len){
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // each tid map I_i
  IV_gen(tid,len,cube,&I_iv[tid*IV_dim]); // uint64_t tid, uint8_t len, uint8_t cube_index[],uint8_t *iv
}


__global__ void generate_key_set(uint8_t *k,curandState *states){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64(), tid, 0, &states[tid]);
    k[tid] = curand_uniform(&states[tid])*0xFF;
}

__global__ void generate_key_set_xor(uint8_t *k1_set, uint8_t *k2_set, uint8_t *k_xor_set){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    k_xor_set[tid] = k1_set[tid]^k2_set[tid];
}

__global__ void cuda_encrypt(uint8_t *k, uint8_t *I,uint8_t *out){
	uint8_t IV[IV_dim];
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	IV_gen(tid,k_curr_dev,I,IV); 
	out[tid] = encrypt(IV,k);
}

__global__ void cuda_encrypt_constant_key(uint8_t *I,uint8_t *out){
	uint8_t IV[IV_dim];
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	IV_gen(tid,k_curr_dev,I,IV); 
	out[tid] = encrypt(IV,d_k);
}

/// Interleaved Pair Implementation with less divergence
__global__ void sumZ2CubeReduceInterleaved (uint8_t *g_idata, uint8_t *g_odata, unsigned int n) {
	// set thread ID
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n){
	unsigned int tid = threadIdx.x;
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
}

__global__ void sumCubeReduceInterleaved (uint8_t *g_idata, uint8_t *g_odata, unsigned int n) {
	// set thread ID
	unsigned int id =  blockIdx.x * blockDim.x + threadIdx.x;
	if (id<n){
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local pointer of this block 
	uint8_t *idata = g_idata + blockIdx.x * blockDim.x;
	// boundary check if(idx >= n) return;
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			idata[tid] += idata[tid + stride];
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
	}
}


__global__ void andCubeReduceInterleaved (uint8_t *g_idata, uint8_t *g_odata, uint64_t n) {
	// set thread ID
	unsigned int id =  blockIdx.x * blockDim.x + threadIdx.x;
	if (id<n){
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local pointer of this block 
	uint8_t *idata = g_idata + blockIdx.x * blockDim.x;
	// boundary check if(idx >= n) return;
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			idata[tid] &= (idata[tid + stride] ==  d_p0_dev);
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
	}
}

__global__ void orCubeReduceInterleaved (uint8_t *g_idata, uint8_t *g_odata, unsigned int n) {
	// set thread ID
	unsigned int id =  blockIdx.x * blockDim.x + threadIdx.x;
	if (id<n){
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local pointer of this block 
	uint8_t *idata = g_idata + blockIdx.x * blockDim.x;
	// boundary check if(idx >= n) return;
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			idata[tid] |= idata[tid + stride];
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
	}
}


__global__ void cudaTestBLR(uint8_t *out,uint8_t* p1_set, uint8_t *p2_set, uint8_t *p1_2_set, unsigned int n){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<n){
		//out[tid] = (uint8_t)((p1_set[tid]^p2_set[tid] != p1_2_set[tid]) && (d_p0_dev^p1_set[tid]^p2_set[tid]!=p1_2_set[tid]));
		out[tid] = (uint8_t)(d_p0_dev^p1_set[tid]^p2_set[tid]!=p1_2_set[tid]);
		//out[tid] = ((p1_set[tid]^p2_set[tid] != p1_2_set[tid]));

	}
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
		sum+=1ull<<k[i];
	return sum;
} 

void copy_data_I(uint8_t *src, uint8_t *dst, unsigned int dim){
	for(int i=0;i<dim;i++)
		dst[i] = src[i];
}

void random_key(uint8_t k[]){
	for(int i=0;i<K_dim;i++)
		k[i]=rand()%(1<<8);
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


uint8_t *set_cubes(uint8_t *out_I_host,uint8_t *out_k_host,uint64_t n_bytes,uint64_t sum,unsigned int blocksize, uint64_t ACTIVE_INSTANCES){
	uint64_t start = 0, start_I = 0;
	uint8_t *h_IV,*h_I,*dev_IV,*dev_I;
	uint64_t data_len = sum;


	dim3 block(blocksize, 1);
	dim3 grid((ACTIVE_INSTANCES+ block.x - 1) / block.x, 1);
	

	// creazione degli stream asincroni non-NULL


	// host alloc and cuda malloc in one time
	CHECK(cudaHostAlloc((void**) &h_IV,n_bytes*(sizeof(uint8_t)*IV_dim),cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void**) &h_I,sum*(sizeof(uint8_t)),cudaHostAllocDefault));
	copy_data_I(out_I_host,h_I,sum);
	// copy from host memory to pinned memory
	CHECK(cudaMalloc((void **)&dev_IV,n_bytes*(sizeof(uint8_t)*IV_dim)));
	CHECK(cudaMalloc((void **)&dev_I,sum*sizeof(uint8_t)));
	cudaStream_t stream[ACTIVE_INSTANCES];
	for (int i = 0; i < ACTIVE_INSTANCES; ++i) 
    	CHECK(cudaStreamCreate(&stream[i]));
    
	for(int i=0;i<ACTIVE_INSTANCES;i++){
		/*
			foreach INSTANCES (which means each time you consider a different k choice)
			I allocate a kernel with a number of thread equal to the number of cube rotation
			however you need cube to convert tid to the conversion convention
		*/
		data_len = 1ull<<(uint64_t)out_k_host[i];// have to do this for each I - stream
		// set optimal block size for each CUDA stream
		block.x = blocksize;
		grid.x = (data_len + block.x - 1) / block.x;
		CHECK(cudaMemcpyAsync(&dev_I[start_I],&h_I[start_I],out_k_host[i]*sizeof(uint8_t),cudaMemcpyHostToDevice,stream[i]));
		generate_IV<<<grid,block,0,stream[i]>>>(&dev_I[start_I],&dev_IV[start*IV_dim],out_k_host[i]);
		CHECK(cudaMemcpyAsync(&h_IV[start*IV_dim],&dev_IV[start*IV_dim],data_len*sizeof(uint8_t)*IV_dim,cudaMemcpyDeviceToHost,stream[i]));
		start+=data_len;
		start_I+=out_k_host[i];
	}	

	for(int i=0; i<ACTIVE_INSTANCES; ++i)
		cudaStreamDestroy(stream[i]);
	
	CHECK(cudaFreeHost(h_I));
	CHECK(cudaFree(dev_I));
	CHECK(cudaFree(dev_IV));
	

	return h_IV;
}


// set_cubes_host(out_I_host,out_k_host,n_bytes_IV,k_sum);

// uint8_t * as return type

uint8_t *set_cubes_host(uint8_t* out_I_host,uint8_t* out_k_host, uint64_t n_bytes, uint64_t ksum,uint64_t ACTIVE_INSTANCES){
	
	uint8_t *cube_index, *h_IV;
	uint8_t iv[IV_dim],c_i;
	uint64_t len,l_limit,start=0,start_IV=0;

	h_IV = (uint8_t *)malloc(n_bytes*(sizeof(uint8_t)*IV_dim));

	for (int c = 0; c<ACTIVE_INSTANCES; c++){
		len = out_k_host[c]; // I size
		cube_index = &out_I_host[start];
		flush_iv_host(iv);
		l_limit=len;
		l_limit = (1ull<<(l_limit));

		for(unsigned long l=0;l<l_limit;l++){
			for(unsigned int i=0; i<len;i++){
				c_i = ((l/(1ull<<i)))%2;
				iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB_host(c_i * (1ull<<(cube_index[i]%8)));
			}
			copy_IV_host(&h_IV[start_IV],iv);
			flush_iv_host(iv);	
			start_IV += IV_dim*sizeof(uint8_t); 
		}
		start+=len;
	}

	return h_IV;
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
	uint8_t element = rand()%(IV_dim*8);
	while(unsigned8_element_in_array(element,excluded,dim))
		element = rand()%(IV_dim*8);
	return element;
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

uint8_t check_p_coeff(uint8_t *k,uint8_t *h_I,uint64_t len,uint8_t k_len){
	uint8_t gpu_sum=0,c;
	uint8_t IV[IV_dim];

	for(int i=0;i<len;i++){
		IV_gen_host(i,k_len,h_I,IV);
		//print_IV_host(IV); 
		c = encrypt_host(IV,k);
		gpu_sum ^= c;
	}
	return gpu_sum;
}


bool hostTestBLR(uint8_t gpu_sum_0,uint8_t *p1_set,uint8_t *p2_set,uint8_t *p1_2_set){
	bool test=false;
	for(int i=0;i<N*CONSTANT;i++){
		//if(p1_set[i]^p2_set[i] != p1_2_set[i] && gpu_sum_0^p1_set[i]^p2_set[i]!=p1_2_set[i]){
		if(gpu_sum_0^p1_set[i]^p2_set[i]!=p1_2_set[i]){
			test=true;
			break;
		}
	}
	return test;
}

bool const_response(uint8_t gpu_sum_0,uint8_t *p1_set,uint8_t *p2_set){
	for(uint64_t i = 0;i<N*CONSTANT;i++){
		if(gpu_sum_0!=p1_set[i] || gpu_sum_0 !=p2_set[i])
			return false;
	}
	return true;
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

int main(int argc, char *argv[]){

	/*measure time*/
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;


	if(argc!=5){
		printf("\nUSAGE:./att1 <TRIES><INSTANCES><M_d><N>, argc = %d \n",argc);
		exit(0);
	}	
	TRIES=atoi(argv[1]);
	INSTANCES=atoi(argv[2]);
	M_d=atoi(argv[3]);
	N=atoi(argv[4]);

	if(N%32!=0){
		printf("\nUsing a filter multiple of 32, nearest valid N = %u\n",((N/32)+1) * 32);
		exit(0);
	}
	printf("\nN = %u, INSTANCES = %u, M_d = %u, N = %u\n",TRIES,INSTANCES,M_d,N);
	printf("\nOK!\n");
    
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
 	uint8_t *out_k,*out_I,*out_k_host,*out_I_host;
	//INSTANCES
    out_k_host = (uint8_t *)malloc(INSTANCES*sizeof(uint8_t));
	CHECK(cudaMalloc((void **)&devStates, INSTANCES*sizeof(curandState))); 
	CHECK(cudaMalloc((uint8_t **)&out_k,INSTANCES*sizeof(uint8_t)));
	printf("\n FIRST KERNEL CALL - k-sets \n");
	cudaMemcpyToSymbol(*(&M_d_dev),&M_d, sizeof(uint8_t));
  	random_k<<<block,grid>>>(out_k,devStates);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_k_host,out_k,INSTANCES*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    print_arr_host("k dim",out_k_host, INSTANCES); 
	CHECK(cudaFree(out_k));
	CHECK(cudaFree(devStates));	
	//out_k_host[0]=10;

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
	CHECK(cudaMalloc((void **)&devStates, k_sum*sizeof(curandState))); 
	CHECK(cudaMalloc((uint8_t **)&out_k,INSTANCES*sizeof(uint8_t)));
	CHECK(cudaMalloc((uint8_t **)&out_I,k_sum*sizeof(uint8_t)));

    CHECK(cudaMemcpy(out_k,out_k_host,INSTANCES*sizeof(uint8_t),cudaMemcpyHostToDevice));


	random_I_unique<<<block,grid>>>(out_I,out_k,devStates);
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_I_host,out_I,k_sum*sizeof(uint8_t),cudaMemcpyDeviceToHost));
	CHECK(cudaFree(out_k));
	CHECK(cudaFree(out_I));
	CHECK(cudaFree(devStates));	
    print_arr_host("I sets",out_I_host, k_sum); 

	/*Kernel 3*/
	/*
		For set_cubes method you must provide out_k_host, 
		which contail array of k dimension choosen randomly.
	*/
	uint64_t n_bytes_IV = get_nbytes_IV(out_k_host,INSTANCES);
	//uint8_t *h_IV = set_cubes(out_I_host,out_k_host,n_bytes_IV,k_sum,blocksize,INSTANCES);
	// result in h_IV
	cpu_startTime = clock();
	//uint8_t *h_IV_host = set_cubes_host(out_I_host,out_k_host,n_bytes_IV,k_sum,INSTANCES);
	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);
	printf("\nKernel 3 CPU elpased times = %lf\n",cpu_ElapseTime);
	

	// compare h_IV_host with h_IV for validation
	

	
	/*Kernel 4*/
   
	/*When using stream, you got to allocate large chunk of pinned memory and then play with offset inside each stream*/

	


	
	

	// h_IVI[0] contains all cubes edges 
	// h_IVI[1] is out_I_host in pinned memory


	/*kernel 4: random key generator*/
	uint64_t del_1[INSTANCES],add_1[INSTANCES],ACTIVE_INSTANCES=INSTANCES,start=0,start_I=0,n1;
	uint64_t del_1_counter=0,add_1_counter=0,sol_dim=0,sol_count=0,*sol_dim_arr=NULL;
	uint8_t k0[K_dim*sizeof(uint8_t)]={0x0},*k1_set_dev,*k2_set_dev,*k_xor_dev,*k1_set_host,*k2_set_host,*k_xor_host;
	uint8_t *out_cube, *dev_reduce_out_n_supp,*dev_reduce_out_n_supp1,*reduce_out,**updated_cubes,*maxterms=NULL;
	uint8_t *p1_set,*p2_set,*p1_2_set;
	uint8_t *p1_set_dev,*p2_set_dev,*p1_2_set_dev;
	uint8_t *dev_out_cube_n,*dev_out_cube_n1,*dev_out_cube_n2, *dev_reduce_out_n,*hreduce_out_n;
	uint8_t gpu_sum_0;

	

	cudaMemcpyToSymbol(d_k,k0, sizeof(uint8_t)*K_dim);
	cudaStream_t stream[N*CONSTANT];

	p1_set =(uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t));
	p2_set =(uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t));
	p1_2_set =(uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t));

	CHECK(cudaMalloc((void **)&p1_set_dev,N*CONSTANT*sizeof(uint8_t)));
	CHECK(cudaMalloc((void **)&p2_set_dev,N*CONSTANT*sizeof(uint8_t)));
	CHECK(cudaMalloc((void **)&p1_2_set_dev,N*CONSTANT*sizeof(uint8_t)));

	//CHECK(cudaMalloc((void **)&p_supp,N*CONSTANT*sizeof(uint64_t)));


	CHECK(cudaMalloc((void **)&dev_reduce_out_n_supp,N*CONSTANT*sizeof(uint8_t)));
	CHECK(cudaMalloc((void **)&dev_reduce_out_n_supp1,N*CONSTANT*sizeof(uint8_t)));
	//CHECK(cudaMalloc((void **)&dev_red_out64,N*CONSTANT*sizeof(uint64_t)));

	//CHECK(cudaMalloc((void **)&dev_reduce_out_n_supp2,N*CONSTANT*sizeof(uint8_t)));


	CHECK(cudaMalloc((void **)&k1_set_dev,N*CONSTANT*sizeof(uint8_t)*K_dim));
	CHECK(cudaMalloc((void **)&k2_set_dev,N*CONSTANT*sizeof(uint8_t)*K_dim));
	CHECK(cudaMalloc((void **)&k_xor_dev,N*CONSTANT*sizeof(uint8_t)*K_dim));

	k1_set_host = (uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t)*K_dim);
	k2_set_host = (uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t)*K_dim);
	k_xor_host = (uint8_t *)malloc(N*CONSTANT*sizeof(uint8_t)*K_dim);



	for(int t=0; t<TRIES; t++){
		start=0,start_I=0,del_1_counter=0,add_1_counter=0;
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
		blocksize=32;

		// prepare kernel for k1 keys
		// prepare kernel for k2 keys
		block.x = blocksize;
		grid.x = ((N*CONSTANT*K_dim)+ block.x - 1) / block.x;
		
		CHECK(cudaMalloc((void **)&devStates, K_dim*N*CONSTANT*sizeof(curandState))); 
		generate_key_set<<<grid,block>>>(k1_set_dev,devStates);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(k1_set_host,k1_set_dev,N*CONSTANT*sizeof(uint8_t)*K_dim,cudaMemcpyDeviceToHost));
		generate_key_set<<<grid,block>>>(k2_set_dev,devStates);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(k2_set_host,k2_set_dev,N*CONSTANT*sizeof(uint8_t)*K_dim,cudaMemcpyDeviceToHost));
		CHECK(cudaFree(devStates));

		generate_key_set_xor<<<grid,block>>>(k1_set_dev,k2_set_dev,k_xor_dev);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(k_xor_host,k_xor_dev,N*CONSTANT*sizeof(uint8_t)*K_dim,cudaMemcpyDeviceToHost));


		for(int c=0; c<ACTIVE_INSTANCES; c++){
			printf("\n##### NEW CUBE %d #####\n",c);
			data_len = (uint64_t)1ull<<(uint64_t)out_k_host[c];
			//CHECK(cudaMemcpyAsync(&dev_IV[start],&h_IV[start],data_len*sizeof(uint8_t)*IV_dim,cudaMemcpyHostToDevice));
			//CHECK(cudaDeviceSynchronize());



			if(data_len <  1073741824){  // 2 **30 limit for grid block
				if(data_len/2048 > 0) // 2*25
					blocksize = 1024;
				else	
					blocksize=32;
			}
			else{
				printf("\ncube len not supported for 1d grid and 1 d block\n");
				break;
				//exit(1);
			}

			blocksize = (data_len<blocksize)? data_len:blocksize;
			block.x = blocksize;
			grid.x = (data_len+ block.x - 1) / block.x;


			
			hreduce_out_n =(uint8_t *)malloc(N*CONSTANT*grid.x*sizeof(uint8_t));
			//hreduce_out_n1 =(uint8_t *)malloc(N*CONSTANT*grid.x*sizeof(uint8_t));
			//hreduce_out_n2 =(uint8_t *)malloc(N*CONSTANT*grid.x*sizeof(uint8_t));
			printf("\n4 allocation of %lu bytes\n",N*CONSTANT*data_len*sizeof(uint8_t));
			CHECK(cudaMalloc((void **)&dev_reduce_out_n,N*CONSTANT*data_len*sizeof(uint8_t)));
			CHECK(cudaMalloc((void **)&dev_out_cube_n,data_len*N*CONSTANT*sizeof(uint8_t)));
			CHECK(cudaMalloc((void **)&dev_out_cube_n1,data_len*N*CONSTANT*sizeof(uint8_t)));
			CHECK(cudaMalloc((void **)&dev_out_cube_n2,data_len*N*CONSTANT*sizeof(uint8_t)));

			cudaMemcpyToSymbol(*(&k_curr_dev),&out_k_host[c], sizeof(uint8_t));
			printf("launching cuda encrypt with config:<<<%i,%i>>>, kernel total = %i, data_len = %lu\n",block.x,grid.x,block.x*grid.x,data_len);
			/*COMPUTE P1 for all N*CONSTANT */
			//setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);
			cpu_startTime = clock();
			for (int i = 0; i < N*CONSTANT; ++i) 
    			CHECK(cudaStreamCreate(&stream[i]));
			for(int i=0;i<N*CONSTANT;i++){
				cuda_encrypt<<<grid,block,0,stream[i]>>>(&k1_set_dev[i*K_dim],&out_I[start_I],&dev_out_cube_n[i*data_len]);
			}
			for(int i=0;i<N*CONSTANT;i++){
				cuda_encrypt<<<grid,block,0,stream[i]>>>(&k2_set_dev[i*K_dim],&out_I[start_I],&dev_out_cube_n1[i*data_len]);
			}
			for(int i=0;i<N*CONSTANT;i++){
				cuda_encrypt<<<grid,block,0,stream[i]>>>(&k_xor_dev[i*K_dim],&out_I[start_I],&dev_out_cube_n2[i*data_len]);
			}
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamSynchronize(stream[i]));
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamDestroy(stream[i]));
			cpu_endTime = clock();
			cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);
			printf("\nCompute p coeff GPU elpased times = %lf\n",cpu_ElapseTime);
	

			for (int i = 0; i < N*CONSTANT; ++i) 
    			CHECK(cudaStreamCreate(&stream[i]));
			for(int i=0;i<N*CONSTANT;i++){
				sumZ2CubeReduceInterleaved<<<grid,block,0,stream[i]>>>(&dev_out_cube_n[i*data_len],&dev_reduce_out_n[grid.x*i],data_len);
			}
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamSynchronize(stream[i]));
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamDestroy(stream[i]));

			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x*sizeof(uint8_t)*N*CONSTANT,cudaMemcpyDeviceToHost));
			for(int i=0;i<N*CONSTANT;i++){
				p1_set[i]=0;
				for (int i1 = 0; i1 < grid.x; i1++)
					p1_set[i] ^= hreduce_out_n[(i*grid.x)+i1];
			}



			for (int i = 0; i < N*CONSTANT; ++i) 
				CHECK(cudaStreamCreate(&stream[i]));
			for(int i=0;i<N*CONSTANT;i++){
				sumZ2CubeReduceInterleaved<<<grid,block,0,stream[i]>>>(&dev_out_cube_n1[i*data_len],&dev_reduce_out_n[grid.x*i],data_len);
			}
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamSynchronize(stream[i]));
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamDestroy(stream[i]));
			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x*sizeof(uint8_t)*N*CONSTANT,cudaMemcpyDeviceToHost));

			for(int i=0;i<N*CONSTANT;i++){
				p2_set[i]=0;
				for (int i1 = 0; i1 < grid.x; i1++)
					p2_set[i] ^= hreduce_out_n[(i*grid.x)+i1];
			}			


			for (int i = 0; i < N*CONSTANT; ++i) 
				CHECK(cudaStreamCreate(&stream[i]));
			for(int i=0;i<N*CONSTANT;i++){
				sumZ2CubeReduceInterleaved<<<grid,block,0,stream[i]>>>(&dev_out_cube_n2[i*data_len],&dev_reduce_out_n[grid.x*i],data_len);
			}
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamSynchronize(stream[i]));
			for(int i=0; i<N*CONSTANT; ++i)
				CHECK(cudaStreamDestroy(stream[i]));
			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x*sizeof(uint8_t)*N*CONSTANT,cudaMemcpyDeviceToHost));
			for(int i=0;i<N*CONSTANT;i++){
				p1_2_set[i]=0;
				for (int i1 = 0; i1 < grid.x; i1++)
					p1_2_set[i] ^= hreduce_out_n[(i*grid.x)+i1];
			}






			/*
			printf("\nHost check for p1_coeff set\n");
			// check host for p_1
			cpu_startTime = clock();
			for(int i=0;i<N*CONSTANT;i++){
				if(check_p_coeff(&k1_set_host[i*K_dim],&out_I_host[start_I],data_len,out_k_host[c])==p1_set[i]){
					printf("OK => %d\r",i);
					fflush(stdout);				}
				else{
					printf("\nError while compute coefficient in %d\n",i);
					exit(1);
				}
			}
			*/
	
			// check host for p_2
			/*
			printf("\nHost check for p2_coeff set\n");
			for(int i=0;i<N*CONSTANT;i++){
				if(check_p_coeff(&k2_set_host[i*K_dim],&out_I_host[start_I],data_len,out_k_host[c])==p2_set[i]){
					printf("OK => %d\r",i);
					fflush(stdout);
				}
				else{
					printf("\nError while compute coefficient in %d\n",i);
					exit(1);
				}
			}*/

	
			// check host for p1^p_2

			/*
			printf("\nHost check for p1_coeff ^ p2_coeff set\n");
			for(int i=0;i<N*CONSTANT;i++){
				if(check_p_coeff(&k_xor_host[i*K_dim],&out_I_host[start_I],data_len,out_k_host[c])==p1_2_set[i]){
					printf("OK => %d\r",i);
					fflush(stdout);
				}
				else{
					printf("\nError while compute coefficient in %d\n",i);
					exit(1);
				}
			}
			*/
			


			/*COMPUTE P0 for all N*CONSTANT */

			
			cuda_encrypt_constant_key<<<grid,block>>>(&out_I[start_I],&dev_out_cube_n[data_len]);
			CHECK(cudaDeviceSynchronize());
			sumZ2CubeReduceInterleaved<<<grid,block>>>(&dev_out_cube_n[data_len],dev_reduce_out_n,data_len);
			CHECK(cudaDeviceSynchronize());
			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost));
		
			gpu_sum_0 = 0;
			
			for (int i = 0; i < grid.x; i++){
				gpu_sum_0 ^= hreduce_out_n[i];
			} 
			/*
			if(check_p_coeff(k0,&out_I_host[start_I],data_len,out_k_host[c])==gpu_sum_0){
				printf("\nOK p0_coeff \n");
			}
			else{
				printf("\nError while compute coefficient in p0_coeff\n");
				exit(1);
			}
			

			CHECK(cudaFree(dev_out_cube_n));
			printf("\nCoeff OK\n");
			*/

			blocksize=32;
			blocksize = (N*CONSTANT<blocksize)? N*CONSTANT:blocksize;
			block.x = blocksize;
			grid.x = ((N*CONSTANT)+ block.x - 1) / block.x;

			printf("\nBLR config:<<<%d,%d>>>, kernel total = %d, N*CONSTANT = %u\n",block.x,grid.x,block.x*grid.x,N*CONSTANT);
			free(hreduce_out_n);
			hreduce_out_n=(uint8_t *)malloc(grid.x*sizeof(uint8_t));
			for(int i=0;i<grid.x;i++)
				hreduce_out_n[i]=0;

			/*BLR tests*/
			CHECK(cudaMemcpy(p1_set_dev,p1_set,sizeof(uint8_t)*N*CONSTANT,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(p2_set_dev,p2_set,sizeof(uint8_t)*N*CONSTANT,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(p1_2_set_dev,p1_2_set,sizeof(uint8_t)*N*CONSTANT,cudaMemcpyHostToDevice));
			cudaMemcpyToSymbol(*(&d_p0_dev),&gpu_sum_0, sizeof(uint8_t));
			printf("\nHOST = %d\n",gpu_sum_0);
			CHECK(cudaMemset(dev_reduce_out_n_supp,0,sizeof(uint8_t)*N*CONSTANT));
			CHECK(cudaMemset(dev_reduce_out_n_supp1,0,sizeof(uint8_t)*N*CONSTANT));

			cudaTestBLR<<<grid,block>>>(dev_reduce_out_n_supp,p1_set_dev,p2_set_dev,p1_2_set_dev,N*CONSTANT*sizeof(uint8_t));
			CHECK(cudaDeviceSynchronize());
			orCubeReduceInterleaved<<<grid,block>>>(dev_reduce_out_n_supp,dev_reduce_out_n_supp1,N*CONSTANT*sizeof(uint8_t));
			CHECK(cudaDeviceSynchronize());

			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n_supp1, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost));
				
			bool expand=false;

			for (int i = 0; i < grid.x; i++){
				if(hreduce_out_n[i]){
					printf("\nCUBE %d to expand\n",c);
					expand=true;
					add_1[add_1_counter] = c; // save the cube to be reduced
					add_1_counter+=1;
					break;
				}
			}
			/*
			printf("\nHOST CHECKING expand cube result BLR\n");
			if(!(expand==hostTestBLR(gpu_sum_0,p1_set,p2_set,p1_2_set))){
				printf("\nTEST HOST BLR ... [FAIL]\n");
				exit(1);
			}
			printf("\nTEST HOST BLR ... [OK]\n");
			*/
			if(!expand){
				n1=1;
				for (int i = 0; i < N*CONSTANT; i++){
					n1 &= (p1_set[i]==gpu_sum_0);
				} 
				for (int i = 0; i < N*CONSTANT; i++){
					n1 &= (p2_set[i]==gpu_sum_0);
				} 
				//n1+=(uint64_t)gpu_sum_0;
				/*CHECK 1 bit in response
				printf("\nHOST CHECKING constant response\n");
				if(!(n1==const_response(gpu_sum_0,p1_set,p2_set))){
					printf("\nTEST HOST constant response ... [FAIL]\n");
					printf("\nn1_gpu = %lu, n1_host= %u \n",n1,const_response(gpu_sum_0,p1_set,p2_set));
					exit(1);
				}
				printf("\nTEST HOST constant response ... n1 = %lu [OK]\n",n1);
				*/
				if(n1){
					if(out_k_host[c]>min_M_d){
						printf("\nMaxterm constant reduce cube\n");
						del_1[del_1_counter] = c;
						del_1_counter+=1;
					}
					else
						printf("\nCannot reduce cube:\n\tc_len = %d\n\tm_d_min = %d \n",out_k_host[c],min_M_d);
				}
				
				else{
					
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
			CHECK(cudaFree(dev_reduce_out_n)); // free here reuse this global memory for sum interleave in check for linearity
			CHECK(cudaFree(dev_out_cube_n));
			CHECK(cudaFree(dev_out_cube_n1));
			CHECK(cudaFree(dev_out_cube_n2));
			free(hreduce_out_n);
			printf("\nTEST OK\n");


		
			start+=data_len;
			start_I+=out_k_host[c];
			/*store valid I sets for superpolys recompute I to be done*/

			
		}

	
		//CHECK(cudaFree(dev_IV));
		CHECK(cudaFree(out_cube));
		CHECK(cudaFree(reduce_out));
		CHECK(cudaFree(out_I));
		
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
		

	}
    printf("\n############ SOLUTIONS ############\n");
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

