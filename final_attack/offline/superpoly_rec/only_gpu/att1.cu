#include <stdio.h>
#include <curand_kernel.h>
#include <iterator>
//#include "mygrain_lib.cuh"

#include "../../../common.h"
#include "../../../cipher.cu"
//#include "../../../mygrain_lib.cuh"
#include "../../../utils.cu"


unsigned int TRIES;
unsigned int INSTANCES;
unsigned int M_d;
unsigned int N;



__constant__ uint8_t d_k[K_dim];
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
	uint8_t *constant;

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


/*
__global__ void key_gen_superpoly_reconstruction(uint8_t *k_set,unsigned int len){
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t *k_start, pos = tid;
	if(tid<len){*/
		/*
			c_i = ((tid/(1ull<<i)))%2;
			iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8)));
		*//*
		k_start=&k_set[K_dim*tid];
		k_start[(tid/8)] = (1<<(pos%8));
		//k_start[(tid/8)] = to_MSB(1<<(pos%8)); 
		//k_start[(K_dim-1) - (tid/8)] = to_MSB(1<<7-(pos%8)); 

	}
  }*/
  


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

__global__ void cuda_encrypt(uint8_t *k,uint8_t *I_iv,uint8_t *out){
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	//out[tid] = encrypt_unroll(&I_iv[tid*IV_dim],k);
	out[tid] = encrypt(&I_iv[tid*IV_dim],k);
}

__global__ void cuda_encrypt_constant_key(uint8_t *cube,uint8_t *out,uint64_t window){
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t IV[IV_dim],pr=0;
	for(uint64_t i=0;i<window;i++){
		IV_gen(((tid)*window)+i,k_curr_dev,cube,IV);
		pr ^= encrypt(IV,d_k);
	//out[tid] = encrypt_unroll(IV,d_k);
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
		sum+=1<<k[i];
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
		data_len = 1<<out_k_host[i]; // have to do this for each I - stream
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



uint8_t check_p_coeff_1(uint8_t *k,uint8_t *IV,uint64_t len){
	uint8_t gpu_sum=0,c;
	for(int i=0;i<len;i++){
		c = encrypt_host(&IV[i*IV_dim],k);
		gpu_sum ^= c;
	}
	return gpu_sum;
}

uint8_t check_p_coeff(uint8_t *k,uint8_t *h_I,uint64_t len,uint8_t k_len){
	uint8_t gpu_sum=0,c;
	uint8_t IV[IV_dim];
	for(int i=0;i<len;i++){
		IV_gen_host(i,k_len,h_I,IV);
		/*
		printf("\n--------------\n");
		print_arr_host("IV ",IV,IV_dim);
		print_arr_host("k ",k,K_dim);
		printf("\n--------------\n");*/


		c = encrypt_host(IV,k);
		gpu_sum ^= c;
		//printf("\nGPU sum in host = %lu\n",gpu_sum);
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
	char fname[] = "./final_attack/offline/cubes_test_window.txt\x00";
	int ln = 0;
	FILE *fp = fopen(fname,"r");
	
	char * line = NULL;
	
	size_t len=0;
	ssize_t read;
	uint64_t len_cubes=0,len_new_cube=0,n_cubes=0;
	if(fp == NULL){
		printf("\nError opening file %s\n",fname);
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

void fprint_cube(FILE *fp, uint8_t *out_I_host,uint64_t dim,uint8_t c){
	if(dim!=0){
    char pre[3]= ", ";
    for(int index=0;index<dim;index++){
        if(index==dim-1){
            pre[0]='\n';
            pre[1]='\0';
        }
		
		if(c == 1 && index == 0)
       		fprintf(fp,"-1, %u%s",out_I_host[index],pre);
		else
		    fprintf(fp,"%u%s",out_I_host[index],pre);
		

		
    }
	}
	else	
		fprintf(fp,"X\n");

}

int main(int argc, char *argv[]){

	/*measure time*/
	
	cubes cs;
	superpolys J;
	get_cubes(&cs);
	print_cubes(cs);

	uint8_t *k_set,*k_set_host,*dev_reduce_out_n,*hreduce_out_n=NULL,*dev_out_cube_n;
	uint8_t k0[K_dim*sizeof(uint8_t)]={0x0},gpu_sum_0,gpu_sum_j;
	uint64_t start=0,data_len,a_j=0,start_I=0;
	unsigned int blocksize = 32;
	dim3 block(blocksize, 1);
	dim3 grid((INSTANCES + block.x - 1) / block.x, 1);


	cudaEvent_t start_t, stop_t;

	float gpu_time = 0.0f;

	uint64_t k_sum = sum_k(cs.cubes_len,cs.n_cubes);
	uint64_t n_bytes_IV = get_nbytes_IV(cs.cubes_len,cs.n_cubes);
	printf("\nk_sum = %lu,n bytes = %lu\n",k_sum,n_bytes_IV);
	print_arr_host("out_k ",cs.cubes_len,cs.n_cubes);
	// = set_cubes(cs.cubes,cs.cubes_len,n_bytes_IV,k_sum,cs.n_cubes);

	

	start=0;

	//CHECK(cudaMalloc((void **)&out_cube,n_bytes_IV*(sizeof(uint8_t))));
	//CHECK(cudaMalloc((void **)&reduce_out,n_bytes_IV*(sizeof(uint8_t))));

	CHECK(cudaMalloc((void **)&k_set,K_dim*8*K_dim*(sizeof(uint8_t)))); // (k_dim*8) == num_keys
	k_set_host = (uint8_t *)malloc(K_dim*8*K_dim*(sizeof(uint8_t)));
	CHECK(cudaMemset(k_set,0,K_dim*8*K_dim*(sizeof(uint8_t))));

	data_len = 8*K_dim;

	blocksize=32;
	blocksize = (data_len<blocksize)? data_len:blocksize;
	block.x = blocksize;
	grid.x = (data_len+ block.x - 1) / block.x;

	key_gen_superpoly_reconstruction<<<grid,block>>>(k_set,K_dim*8);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(k_set_host, k_set,(K_dim*8)*K_dim*(sizeof(uint8_t)) ,cudaMemcpyDeviceToHost));

	for(int i=0;i<BIT_K;i++){
		print_key_host(&k_set_host[i*K_dim]);
	}
	data_len = 0;

	J.arr_lens = (uint8_t *)malloc(cs.n_cubes*sizeof(uint8_t));
	uint8_t *dev_I;
	CHECK(cudaMalloc((void **)&dev_I,k_sum*sizeof(uint8_t)));
	for(int i=0;i<cs.n_cubes;i++)
		J.arr_lens[i]=0;
	J.constant = (uint8_t *)malloc(sizeof(uint8_t)*cs.n_cubes);

	uint64_t window;
	for(int c=0;c<cs.n_cubes;c++){
		data_len = (uint64_t)1<<(uint64_t)cs.cubes_len[c];
		if(data_len/(1<<24) > 0) // 2*25
			blocksize = 1024;
		else	
			blocksize=32;

		window = ((data_len/(1<<30))>0)? data_len/(1<<30):1;
		//window = 2;
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
		CHECK(cudaMemcpyToSymbol(d_k,k0, sizeof(uint8_t)*K_dim));
		CHECK( cudaEventCreate(&start_t) );
		CHECK( cudaEventCreate(&stop_t) );
		cudaEventRecord(start_t, 0);

		cuda_encrypt_constant_key<<<grid,block>>>(&dev_I[start_I],dev_out_cube_n,window);
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaEventRecord(stop_t, 0));
		CHECK(cudaEventSynchronize(stop_t));
		CHECK(cudaEventElapsedTime(&gpu_time, start_t, stop_t));
		printf("\nTime spent for a single cube: %.5f\n", gpu_time);
		CHECK(cudaEventDestroy(start_t));
		CHECK(cudaEventDestroy(stop_t));
		sumZ2CubeReduceInterleaved<<<grid,block>>>(dev_out_cube_n,dev_reduce_out_n,data_len);
		CHECK(cudaDeviceSynchronize());
		
		
		CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());
		
		
		gpu_sum_0 = 0;
		for (int i = 0; i < grid.x; i++)
			gpu_sum_0 ^= hreduce_out_n[i];
			 
		/*
		if(!check_p_coeff(k0,&cs.cubes[start_I],data_len,cs.cubes_len[c])==gpu_sum_0){
			printf("\n[FAIL] k0 fail\n");
			exit(1);
		}*/
		
		if(gpu_sum_0==1){
			//metti la costante 1 in J per il cubo C
			printf("\nConstant in sol %d\n",1);
			J.constant[c]=1;
		}
		
		

		// use stream to make parallel J


		for(int j=0;j<BIT_K;j++){
			printf("\nPrivate key enum [%d] out of %d\n",j,BIT_K);
			
			CHECK(cudaMemcpyToSymbol(d_k,&k_set_host[j*K_dim], sizeof(uint8_t)*K_dim));
			cuda_encrypt_constant_key<<<grid,block>>>(&dev_I[start_I],dev_out_cube_n,window);
			CHECK(cudaDeviceSynchronize());
			sumZ2CubeReduceInterleaved<<<grid,block>>>(dev_out_cube_n,dev_reduce_out_n,data_len);
			CHECK(cudaDeviceSynchronize());
			
			for(int i=0;i<grid.x;i++)
				hreduce_out_n[i]=0;
			
			CHECK(cudaMemcpy(hreduce_out_n, dev_reduce_out_n, grid.x  * sizeof(uint8_t),cudaMemcpyDeviceToHost));
			CHECK(cudaDeviceSynchronize());
			gpu_sum_j = 0;
			for (int i = 0; i < grid.x; i++)
				gpu_sum_j ^= hreduce_out_n[i];
			/*
			if(!check_p_coeff(&k_set_host[j*K_dim],&cs.cubes[start_I],data_len,cs.cubes_len[c])==gpu_sum_j){
				printf("[FAIL] k fail cube=%d, j=%d\r",c,j);
				exit(1);
			}*/

			a_j=gpu_sum_0^gpu_sum_j;
			//printf("\ngpu_sum_0 = %d, gpu_sum_j = %d\n",gpu_sum_0,gpu_sum_j);


			if(a_j==1){
					//metti j in J per il cubo C
					printf("\nJ in sol %lu\n",a_j);
					J.arr_lens[c]+=1;
					J.ncubes+=1;
					J.arr =(uint8_t *)realloc(J.arr,J.ncubes*sizeof(uint8_t));
					J.arr[J.ncubes-1]=j;
				}


		}


		CHECK(cudaFree(dev_out_cube_n));
		CHECK(cudaFree(dev_reduce_out_n));

		start+=data_len;
		start_I+=cs.cubes_len[c];
		//start_IV+=(data_len*IV_dim);
		
	}

	 start=0;
	 for(int i=0;i<cs.n_cubes;i++){
		 printf("\n Superpoly[%d] = [",i);
		 for(int i1=0;i1<J.arr_lens[i];i1++){
			printf(" %d ",J.arr[start+i1]);
		 }
		 printf("]\n");
		 start+=J.arr_lens[i];
	 }

	 printf("\n############ WRITE SOLUTIONS ############\n");
	 FILE *fp = fopen("./final_attack/offline/superpolies_window.txt","w");
 
	 uint64_t sp_start=0;
	 for (int sp = 0; sp<cs.n_cubes;sp++){
		 fprint_cube(fp,&J.arr[sp_start],J.arr_lens[sp],J.constant[sp]);
		 sp_start+=J.arr_lens[sp];
	 }
	 
	 CHECK(cudaDeviceReset());
 
 	printf("\n############ END WRITE SOLUTIONS ############\n");
	return 0;

}

