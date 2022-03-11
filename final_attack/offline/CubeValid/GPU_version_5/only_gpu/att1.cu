#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>
#include "../../../../cipher.cu"
#include "../../../../utils.cu"
#include "../../../../common.h"


#define CONSTANT 1
#define N 1024 // N tests BLR 

__constant__ uint8_t d_k[K_dim];
__constant__ uint8_t k_curr_dev;

typedef struct cubes {
	uint8_t *cubes;
	uint8_t *cubes_len;
	uint64_t n_cubes;
 } cubes;




/*
  Kernel for selecting different random k
*/

__global__ void setup_kernel(curandState* state,uint64_t seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
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
		sum+=1ull<<k[i];
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

uint8_t *set_cubes(uint8_t *out_I_host,uint8_t *out_k_host,uint64_t n_bytes,uint64_t sum, uint64_t ACTIVE_INSTANCES){
	uint64_t start = 0, start_I = 0;
	uint8_t *h_IV,*h_I,*dev_IV,*dev_I;
	uint64_t data_len = sum,blocksize = 32;


	dim3 block;
	dim3 grid;
	

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
		data_len = 1ull<<out_k_host[i]; // have to do this for each I - stream
		// set optimal block size for each CUDA stream
		blocksize = (data_len<blocksize)? data_len:blocksize;
		print_arr_host("I host",&h_I[start_I],out_k_host[i]);
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

uint64_t len_cube(char *s, size_t len){
	uint64_t count = 0;
	for(int i = 0; i<len; i++){
		if(s[i]==',')
			count++;
	}
	count++;
	return count;
}


uint8_t* get_cube(char *s, size_t len){
	uint64_t lcs = len_cube(s,len);
	int temp;
	uint8_t *c = (uint8_t *)malloc(lcs*sizeof(uint8_t));
	char buf[4],b_i=0,c_i=0;
	for(int i=0;i<len+1;i++){
		if(s[i]!=',' && s[i]!='\n' && s[i]!='\0' && s[i]!=EOF && i!=len){
			buf[b_i]=s[i];
			b_i++;
		}
		else{
			buf[b_i]='\0';
			if(strlen(buf)!=0){
			temp = atoi(buf);
			c[c_i] = (uint8_t)temp;
			b_i=0;
			c_i++;}
		}
	
	}

	return c;

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

uint8_t *merge_cube(uint8_t *cubes,uint8_t *new_cube,uint64_t len_cubes,uint64_t len_new_cube){
	cubes = (uint8_t *)realloc(cubes,(len_new_cube+len_cubes)*sizeof(uint8_t));
	for(uint64_t i=len_cubes;i<len_cubes+len_new_cube;i++){
		cubes[i] = new_cube[i-len_cubes];
	}
	return cubes;

	
}

void print_cubes(cubes c){

	printf("\n######## CUBES ########\n");
	uint64_t base=0;

	for(int i=0;i<c.n_cubes;i++){
		for(int i1=0;i1<c.cubes_len[i];i1++){
			printf("%u ",c.cubes[base+i1]);
		}
		base+=c.cubes_len[i];
		printf("\n");
	}
	printf("\n#######################\n");

}


void get_cubes(cubes *c){
	c->cubes = NULL;
	c->cubes_len = NULL;
	char fname[] = "./final_attack/offline/cubes_test.txt";
	int ln = 0;
	FILE *fp = fopen(fname,"r");
	
	char * line = NULL;
	
	size_t len=0;
	ssize_t read;
	uint64_t len_cubes=0,len_new_cube=0,n_cubes=0;
	if(fp == NULL)
		exit(EXIT_FAILURE);
	while((read = getline(&line, &len, fp)) != -1){
		len_new_cube = len_cube(line, read);
		c->cubes=merge_cube(c->cubes,get_cube(line, read),len_cubes,len_new_cube);
		len_cubes+=len_new_cube;
		n_cubes++;
		c->cubes_len=(uint8_t *)realloc(c->cubes_len,n_cubes*sizeof(uint8_t));
		c->cubes_len[n_cubes-1] = len_new_cube;
		ln++;

	}
	fclose(fp);
	if(line) free(line);
	c->n_cubes = n_cubes;
}



int main(){
	
	uint8_t *out_cube,*reduce_out;
	uint64_t start=0,data_len,start_I=0;

	// INIT seed for random function
	time_t t;
	srand((unsigned) time(&t));

	//INSTANCES


	
	cubes cs;
	get_cubes(&cs);
	print_cubes(cs);




	uint64_t k_sum = sum_k(cs.cubes_len,cs.n_cubes);
	uint64_t n_bytes_IV = get_nbytes_IV(cs.cubes_len,cs.n_cubes);
	printf("\nk_sum = %lu,n bytes = %lu\n",k_sum,n_bytes_IV);
	print_arr_host("out_k ",cs.cubes_len,cs.n_cubes);
	uint8_t *h_IV = set_cubes(cs.cubes,cs.cubes_len,n_bytes_IV,k_sum,cs.n_cubes);

    
	

	/*Kernel 4*/
   
	/*When using stream, you got to allocate large chunk of pinned memory and then play with offset inside each stream*/




	


	/*kernel 4: random key generator*/
    start = 0,start_I=0;
	uint8_t p0,p1,p2,p1_2;
	uint64_t del_1[cs.n_cubes],add_1[cs.n_cubes],n0,n1;
	uint64_t del_1_counter=0,add_1_counter=0;
	uint8_t k0[K_dim*sizeof(uint8_t)]={0x0};
	uint8_t k1[K_dim],k2[K_dim],kx[K_dim];
	uint8_t *dev_IV;
	
	cudaStream_t stream[N*CONSTANT];
	
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();



	
		/*
			Move inside T loop and recompute all cubes in IV
		*/
        uint8_t *out_I;

	    CHECK(cudaMalloc((void **)&dev_IV,n_bytes_IV*(sizeof(uint8_t)*IV_dim)));
	    CHECK(cudaMalloc((void **)&out_cube,n_bytes_IV*(sizeof(uint8_t))));
	    CHECK(cudaMalloc((void **)&reduce_out,n_bytes_IV*(sizeof(uint8_t))));
	    CHECK(cudaMalloc((void **)&out_I,k_sum*(sizeof(uint8_t))));
	    CHECK(cudaMemcpy(out_I,cs.cubes,k_sum*sizeof(uint8_t),cudaMemcpyHostToDevice));

		/*
			Update ACTIVE_INSTANCES with del_1,add_1
		*/

		for(int c=0; c<cs.n_cubes; c++){

			printf("\n##### NEW CUBE %d #####\n",c);
			
			cudaMemcpyToSymbol(*(&k_curr_dev),&cs.cubes_len[c], sizeof(uint8_t));
			data_len = (uint64_t)1ull<<(uint64_t)cs.cubes_len[c];
			n0=0,n1=0,p0=0,p1=0,p2=0,p1_2=0;
			// memcopy for each kernel call is the same

			//CHECK(cudaMemcpyAsync(&dev_IV[start],&h_IV[start],data_len*sizeof(uint8_t)*IV_dim,cudaMemcpyHostToDevice));
		   
		    print_cube(&cs.cubes[start_I],cs.cubes_len[c]);


			printf("\nn0=%lu, n1 = %lu\n",n0,n1);

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
					n1+=(p0==1);
					n0+=(p0==0);
				}

				p1 = compute_cube(k1,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);
				p2 = compute_cube(k2,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);
				p1_2 = compute_cube(kx,&out_I[start_I],&out_cube[start],&reduce_out[start],stream[i],data_len);

				n1+=(p1==1)+(p2==1);
				n0+=(p1==0)+(p2==0);

				
				
			
				
					
				
				
				// print output bit kernel 4
					
				

				/*Linearity test*/
				//if (p1 ^ p2 != p1_2 && p0 ^ p1 ^ p2 != p1_2){
				if (p0 ^ p1 ^ p2 != p1_2){
					//add to I set a random term an break 
					add_1[add_1_counter] = c; // save the cube to be reduced
					add_1_counter+=1;

					printf("\nBREAK fail BLR TESTS\n");
					print_arr_host("K1: ",k1,K_dim);
				    print_arr_host("K2: ",k2,K_dim);
					print_arr_host("Kx: ",kx,K_dim);
					printf("\np0 = %u, p1 = %u, p2 = %u\n",p0,p1,p2);
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
					if(cs.cubes_len[c]!=1){
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

				}
			}
			start+=data_len;
			start_I+=cs.cubes_len[c];
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
		
	
		
		// result in h_IV
	
		
		
		// compare h_IV_host with h_IV for validation
		
		

	
	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);

	printf("\nTime execution = %lf\n",cpu_ElapseTime);

  
	CHECK(cudaDeviceReset());



	return 0;
	
}
