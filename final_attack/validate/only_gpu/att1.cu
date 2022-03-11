#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>

#include "../../common.h"
#include "../../cipher.cu"
//#include "../../mygrain_lib.cuh"
#include "../../utils.cu"


unsigned int TRIES;
unsigned int INSTANCES;
unsigned int M_d;
unsigned int N;



__device__ uint8_t d_k[K_dim]={0x0};
__constant__ uint8_t d_p0_dev;
__constant__ uint8_t M_d_dev;
__constant__ uint8_t k_curr_dev;


typedef struct cubes {
	uint8_t *cubes;
	uint8_t *cubes_len;
	uint64_t n_cubes;
 } cubes;

 typedef struct superpolys {
	uint8_t *arr=NULL;
	uint8_t *arr_lens=NULL;
	uint64_t ncubes=0;
} superpolys;
/*
  Kernel for selecting different random k
*/

__global__ void random_k(uint8_t *out_k,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]);
  out_k[tid] = curand_uniform(&states[tid])*(M_d_dev)+1;
}

__global__ void random_I(uint8_t *out_I,curandState *states){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), tid, 0, &states[tid]);
  out_I[tid] = curand_uniform(&states[tid])*96;
}

__global__ void random_I_unique(uint8_t *out_I,uint8_t *out_k,curandState *states){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64(), tid, 0, &states[tid]);
	uint64_t start = 0;
	uint8_t el;
	for(uint64_t i=0;i<tid;i++)
		start+=out_k[i];
	for(uint64_t i=0;i<out_k[tid];i++){
		el = curand_uniform(&states[tid])*96;
		while(unsigned8_element_in_array_gpu(el,&out_I[start],out_k[tid])){
			el = curand_uniform(&states[tid])*96;
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
    uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x);
	curand_init(clock64(), tid, 0, &states[tid]);
    k[tid] = curand_uniform(&states[tid])*0xFF;
}

__global__ void generate_key_set_xor(uint8_t *k1_set, uint8_t *k2_set, uint8_t *k_xor_set){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    k_xor_set[tid] = k1_set[tid]^k2_set[tid];
}


__global__ void cuda_encrypt_2_exploit(uint8_t *cube,uint8_t *out,uint64_t window){
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t IV[IV_dim],pr=0;
	for(uint64_t i=0;i<window;i++){
		IV_gen(((tid)*window)+i,k_curr_dev,cube,IV);
		pr ^= encrypt_exploit(IV);
	}
	out[tid] = pr;
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


__global__ void cudaTestBLR(uint8_t *out,uint8_t* p1_set, uint8_t *p2_set, uint8_t *p1_2_set, unsigned int n){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<n){
		out[tid] = (uint8_t)((p1_set[tid]^p2_set[tid] != p1_2_set[tid]) && (d_p0_dev^p1_set[tid]^p2_set[tid]!=p1_2_set[tid]));
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



// h_IV,h_IV_host,n_bytes_IV*IV_dim
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
				//iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB_host(c_i * (1<<(cube_index[i]%8)));
				iv[(IV_dim-1)-(cube_index[i]/8)] += (c_i * (1<<(cube_index[i]%8)));
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

uint8_t check_p_coeff_1(uint8_t *IV,uint64_t len){
	uint8_t gpu_sum=0,c;
	for(int i=0;i<len;i++){
		c = encrypt_exploit_host(&IV[i*IV_dim]);
		gpu_sum ^= c;
	}
	return gpu_sum;
}

uint8_t check_p_coeff(uint8_t *h_I,uint64_t len,uint8_t k_len){
	uint8_t gpu_sum=0,c;
	uint8_t IV[IV_dim];
	for(int i=0;i<len;i++){
		IV_gen_host(i,k_len,h_I,IV);
		c = encrypt_exploit_host(IV);
		gpu_sum ^= c;
	}
	return gpu_sum;
}


bool hostTestBLR(uint8_t gpu_sum_0,uint8_t *p1_set,uint8_t *p2_set,uint8_t *p1_2_set){
	bool test=false;
	for(int i=0;i<3*N;i++){
		if(p1_set[i]^p2_set[i] != p1_2_set[i] && gpu_sum_0^p1_set[i]^p2_set[i]!=p1_2_set[i]){
			test=true;
			break;
		}
	}
	return test;
}

uint64_t count_response(uint8_t gpu_sum_0,uint8_t *p1_set,uint8_t *p2_set){
	uint64_t count=0;
	for(int i=0;i<3*N;i++){
		if(p1_set[i]==1) count+=1;
		if(p2_set[i]==1) count+=1;
	}
	count+=gpu_sum_0;
	return count;
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


uint8_t* get_cube_uint8(uint8_t *s, size_t len){
	uint64_t lcs = K_dim;
	int temp;
	uint8_t *c = (uint8_t *)malloc(lcs*sizeof(uint8_t));
	char buf[4],b_i=0,c_i=0;
	for(int i=0;i<len+1;i++){
		if(s[i]!=',' && s[i]!='\n' && s[i]!='\0' && s[i]!=(uint64_t)EOF && i!=len){
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

//h_key_2_guess
void get_key(){

	char fname[] = "./final_attack/offline/key2guess.txt";
	FILE *fp = fopen(fname,"r");
	if(fp == NULL){
		printf("\nError fopen in get key\n");
		exit(EXIT_FAILURE);
		
	}
	char * line = NULL;
	ssize_t read;
	size_t len=0;
	uint8_t *hk;
	uint64_t len_line;
	if((read = getline(&line, &len, fp)) != -1){
		len_line = len_cube(line,read);
		hk= get_cube(line,read);
	}
	else
		printf("\nError in readline of secret key\n");

	for(int i=0;i<K_dim;i++){
		h_key_2_guess[i]=hk[i];
	}
}

void get_cubes(cubes *c){
	c->cubes = NULL;
	c->cubes_len = NULL;
	char fname[] = "./final_attack/offline/cubes_test_val.txt";
	int ln = 0;
	FILE *fp = fopen(fname,"r");
	
	char * line = NULL;
	
	size_t len=0;
	ssize_t read;
	uint64_t len_cubes=0,len_new_cube=0,n_cubes=0;
	if(fp == NULL){
		printf("\nError fopen in get cubes\n");
		exit(EXIT_FAILURE);
		
	}
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

void fprint_right_sp(FILE *fp, uint8_t *out_I_host,uint64_t dim){
    for(int index=0;index<dim;index++){
        fprintf(fp,"%u%s",out_I_host[index],"\n");
    }

}

int main(int argc, char *argv[]){

	/*measure time*/
	
	get_key();
	print_key_host(h_key_2_guess);

	cubes cs;
	get_cubes(&cs);
	print_cubes(cs);
	printf("\nOK\n");

	uint8_t *dev_reduce_out_n,*hreduce_out_n=NULL,*dev_out_cube_n;
	//uint8_t k0[K_dim*sizeof(uint8_t)]={0x0},gpu_sum_0;
	uint8_t gpu_sum_0;
	uint64_t start=0,data_len=0,start_I=0;
	unsigned int blocksize = 32;
	dim3 block(blocksize, 1);
	dim3 grid((INSTANCES + block.x - 1) / block.x, 1);

	uint8_t *dev_I;

	uint64_t k_sum = sum_k(cs.cubes_len,cs.n_cubes);
	uint64_t n_bytes_IV = get_nbytes_IV(cs.cubes_len,cs.n_cubes);
	printf("\nk_sum = %lu,n bytes = %lu\n",k_sum,n_bytes_IV);
	print_arr_host("out_k ",cs.cubes_len,cs.n_cubes);
	uint8_t *h_IV = set_cubes(cs.cubes,cs.cubes_len,n_bytes_IV,k_sum,cs.n_cubes);

	
	CHECK(cudaMemcpyToSymbol(d_key_2_guess, &h_key_2_guess, K_dim*sizeof(uint8_t)));


	//CHECK(cudaMalloc((void **)&out_cube,n_bytes_IV*(sizeof(uint8_t))));
	//CHECK(cudaMalloc((void **)&reduce_out,n_bytes_IV*(sizeof(uint8_t))));


	CHECK(cudaMalloc((void **)&dev_I,k_sum*sizeof(uint8_t)));
	
	uint8_t right_superpoly[cs.n_cubes];
	uint64_t window;

	for(int c=0;c<cs.n_cubes;c++){

		data_len = (uint64_t)1ull<<(uint64_t)cs.cubes_len[c];
		/*blocksize=32;
		blocksize = (data_len<blocksize)? data_len:blocksize;*/
		if(data_len/(1ull<<24) > 0) // 2*25
			blocksize = 1024;
		else	
			blocksize=32;

		window = ((data_len/(1ull<<30))>0)? data_len/(1ull<<30):1;
		data_len/=window;
		blocksize = (data_len<blocksize)? data_len:blocksize;


		block.x = blocksize;
		grid.x = (data_len+ block.x - 1) / block.x;

		hreduce_out_n =(uint8_t *)realloc(hreduce_out_n,grid.x*sizeof(uint8_t));
		cudaMemcpyToSymbol(*(&k_curr_dev),&cs.cubes_len[c], sizeof(uint8_t));
		printf("\ncubes len = %u, start_I = %lu\n",cs.cubes_len[c],start_I);
		print_arr_host("K_host",&cs.cubes[start_I],cs.cubes_len[c]);
		CHECK(cudaMemcpyAsync(&dev_I[start_I],&cs.cubes[start_I],cs.cubes_len[c]*sizeof(uint8_t),cudaMemcpyHostToDevice));
		CHECK(cudaDeviceSynchronize());

		// compute for k0
		CHECK(cudaMalloc((void **)&dev_out_cube_n,data_len*sizeof(uint8_t)));
		CHECK(cudaMalloc((void **)&dev_reduce_out_n,data_len*sizeof(uint8_t)));
		//CHECK(cudaMemcpy((void **)&d_k,(void **)&k0, sizeof(uint8_t)*K_dim, cudaMemcpyHostToDevice));

		cuda_encrypt_2_exploit<<<grid,block>>>(&dev_I[start_I],dev_out_cube_n,window);
		CHECK(cudaDeviceSynchronize());
		sumZ2CubeReduceInterleaved<<<grid,block>>>(dev_out_cube_n,dev_reduce_out_n,data_len);
		CHECK(cudaDeviceSynchronize());
		
		for(int i=0;i<grid.x;i++)
			hreduce_out_n[i]=0;
		
		CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost));

		
		
		gpu_sum_0 = 0;
		for (int i = 0; i < grid.x; i++)
			gpu_sum_0 ^= hreduce_out_n[i];
			 
		/*
		if(!check_p_coeff(&cs.cubes[start_I],data_len,cs.cubes_len[c])==gpu_sum_0){
			printf("\n[FAIL] secret k fail\n");
			exit(1);
		}*/
		
		printf("\n[%d] gpu sum => %u\n",c,gpu_sum_0);
		
		right_superpoly[c] = gpu_sum_0;
		CHECK(cudaFree(dev_out_cube_n));
		CHECK(cudaFree(dev_reduce_out_n));

		start+=data_len;
		start_I+=cs.cubes_len[c];
		//start_IV+=(data_len*IV_dim);
	}

	printf("\n############################\n");

	for(int i=0;i<cs.n_cubes;i++){
		printf("\n[%d] gpu sum => %u\n",i,right_superpoly[i]);

	} 
	printf("\n############################\n");

	/*
    FILE *fp = fopen("./final_attack/offline/superpolies_right_val.txt","w");

    fprint_right_sp(fp,right_superpoly,cs.n_cubes);
    */

	CHECK(cudaDeviceReset());

	return 0;

}

