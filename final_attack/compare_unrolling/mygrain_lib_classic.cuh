#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>



#ifndef UTILS_H
#define UTILS_H

#define STREAM_BYTES	16
#define MSG_BYTES		0
#define BYTE_POSITION	0
#define BIT_POSITION	1

#define K_dim	16
#define IV_dim	12

enum GRAIN_ROUND {INIT, FP1, NORMAL};




typedef struct {
	uint8_t lfsr[128];
	uint8_t nfsr[128];
	uint8_t auth_acc[64];
	uint8_t auth_sr[64];
} grain_state;


// TODO: add struct with output: keystream and optionally macstream and tag
typedef struct {
	uint8_t keystream[STREAM_BYTES];
	uint8_t macstream[STREAM_BYTES];
	uint8_t *message;
} grain_data;

__device__ void init_grain(grain_state *grain, uint8_t *key, uint8_t *iv);
__device__ uint8_t next_lfsr_fb(grain_state *grain);
__device__ uint8_t next_nfsr_fb(grain_state *grain);
__device__ uint8_t next_h(grain_state *grain);
__device__ uint8_t shift(uint8_t fsr[128], uint8_t fb);

__device__ uint8_t next_z(grain_state *grain, uint8_t);
__device__ uint8_t* generate_keystream(grain_state *grain, grain_data *data, uint8_t *);
__device__ void print_state(grain_state *grain);

void init_grain_host(grain_state *grain, uint8_t *key, uint8_t *iv);
uint8_t next_lfsr_fb_host(grain_state *grain);
uint8_t next_nfsr_fb_host(grain_state *grain);
uint8_t next_h_host(grain_state *grain);
uint8_t shift_host(uint8_t fsr[128], uint8_t fb);
uint8_t next_z_host(grain_state *grain, uint8_t);
void generate_keystream_host(grain_state *grain, grain_data *data, uint8_t *);
void print_state_host(grain_state *grain);


#endif
/*
 * Define "PRE" to print the pre-output instead of keystream.
 * Define "INIT" to also print the bits during the initialization part.
 * Do this either here or during compilation with -D flag.
 */

__device__ uint8_t grain_round;
__constant__ uint8_t key_g[16] = {0x50, 0x61, 0x73, 0x73, 0x77, 0x6f, 0x72, 0x64, 0x54, 0x6f, 0x47, 0x75, 0x65, 0x73, 0x73, 0x21}; // key to guess
__constant__ int T = 100; // max tries per term
uint8_t grain_round_host;

void init_grain_host(grain_state *grain, uint8_t *key, uint8_t *iv)
{
	// expand the packed bytes and place one bit per array cell (like a flip flop in HW)
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 8; j++) {
			grain->lfsr[8 * i + j] = (iv[i] & (1 << (7-j))) >> (7-j);
		}
	}

	for (int i = 96; i < 127; i++) {
		grain->lfsr[i] = 1;
	}

	grain->lfsr[127] = 0;

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 8; j++) {
			grain->nfsr[8 * i + j] = (key[i] & (1 << (7-j))) >> (7-j);
		}
	}

	for (int i = 0; i < 64; i++) {
		grain->auth_acc[i] = 0;
		grain->auth_sr[i] = 0;
	}
}

void init_data_host(grain_data *data, uint8_t *msg, uint32_t msg_len)
{
	// allocate enough space for message, including the padding byte, 0x80
	data->message = (uint8_t *) calloc(8 * STREAM_BYTES + 8, 1);
	for (uint32_t i = 0; i < msg_len; i++) {
		data->message[i] = msg[i];
	}

	// always pad data with the byte 0x80
	data->message[msg_len] = 1;
}

uint8_t next_lfsr_fb_host(grain_state *grain)
{
 	/* f(x) = 1 + x^32 + x^47 + x^58 + x^90 + x^121 + x^128 */
	return grain->lfsr[96] ^ grain->lfsr[81] ^ grain->lfsr[70] ^ grain->lfsr[38] ^ grain->lfsr[7] ^ grain->lfsr[0];
}

uint8_t next_nfsr_fb_host(grain_state *grain)
{
	return grain->nfsr[96] ^ grain->nfsr[91] ^ grain->nfsr[56] ^ grain->nfsr[26] ^ grain->nfsr[0] ^ (grain->nfsr[84] & grain->nfsr[68]) ^
			(grain->nfsr[67] & grain->nfsr[3]) ^ (grain->nfsr[65] & grain->nfsr[61]) ^ (grain->nfsr[59] & grain->nfsr[27]) ^
			(grain->nfsr[48] & grain->nfsr[40]) ^ (grain->nfsr[18] & grain->nfsr[17]) ^ (grain->nfsr[13] & grain->nfsr[11]) ^
			(grain->nfsr[82] & grain->nfsr[78] & grain->nfsr[70]) ^ (grain->nfsr[25] & grain->nfsr[24] & grain->nfsr[22]) ^
			(grain->nfsr[95] & grain->nfsr[93] & grain->nfsr[92] & grain->nfsr[88]);
}

uint8_t next_h_host(grain_state *grain)
{
	// h(x) = x0x1 + x2x3 + x4x5 + x6x7 + x0x4x8
	#define x0 grain->nfsr[12]	// bi+12
	#define x1 grain->lfsr[8]		// si+8
	#define x2 grain->lfsr[13]	// si+13
	#define x3 grain->lfsr[20]	// si+20
	#define x4 grain->nfsr[95]	// bi+95
	#define x5 grain->lfsr[42]	// si+42
	#define x6 grain->lfsr[60]	// si+60
	#define x7 grain->lfsr[79]	// si+79
	#define x8 grain->lfsr[94]	// si+94

	uint8_t h_out = (x0 & x1) ^ (x2 & x3) ^ (x4 & x5) ^ (x6 & x7) ^ (x0 & x4 & x8);
	return h_out;
}

uint8_t shift_host(uint8_t fsr[128], uint8_t fb)
{
	uint8_t out = fsr[0];
	for (int i = 0; i < 127; i++) {
		fsr[i] = fsr[i+1];
	}
	fsr[127] = fb;

	return out;
}



uint8_t next_z_host(grain_state *grain, uint8_t keybit)
{
	uint8_t lfsr_fb = next_lfsr_fb_host(grain);
	uint8_t nfsr_fb = next_nfsr_fb_host(grain);
	uint8_t h_out = next_h_host(grain);

	/* y = h + s_{i+93} + sum(b_{i+j}), j \in A */
	uint8_t A[] = {2, 15, 36, 45, 64, 73, 89};

	uint8_t nfsr_tmp = 0;
	for (int i = 0; i < 7; i++) {
		nfsr_tmp ^= grain->nfsr[A[i]];
	}

	uint8_t y = h_out ^ grain->lfsr[93] ^ nfsr_tmp;
	
	uint8_t lfsr_out;

	/* feedback y if we are in the initialization instance */
	if (grain_round_host == INIT) {
		lfsr_out = shift_host(grain->lfsr, lfsr_fb ^ y);
		shift_host(grain->nfsr, nfsr_fb ^ lfsr_out ^ y);
	} else if (grain_round_host == FP1) {
		lfsr_out = shift_host(grain->lfsr, lfsr_fb ^ keybit);
		shift_host(grain->nfsr, nfsr_fb ^ lfsr_out);
	} else if (grain_round_host == NORMAL) {
		lfsr_out = shift_host(grain->lfsr, lfsr_fb);
		shift_host(grain->nfsr, nfsr_fb ^ lfsr_out);
	}

	return y;
}


void print_state_host(grain_state *grain)
{
	printf("LFSR: ");
	for (int i = 0; i < 128; i++) {
		printf("%d", grain->lfsr[i]);
	}
	printf("\nNFSR: ");
	for (int i = 0; i < 128; i++) {
		printf("%d", grain->nfsr[i]);
	}
	printf("\n");
}

void print_stream_host(uint8_t *stream, uint8_t byte_size)
{
	for (int i = 0; i < byte_size; i++) {
		uint8_t yi = 0;
		for (int j = 0; j < 8; j++) {
			yi = (yi << 1) ^ stream[i * 8 + j];
		}
		printf("%02x", yi);
	}
	printf("\n");
}

void generate_keystream_host(grain_state *grain, grain_data *data, uint8_t *key,uint8_t *ks)
{
	grain_round_host = FP1;

	uint8_t key_idx = 0;
	/* inititalize the accumulator and shift reg. using the first 64 bits */
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			uint8_t fp1_fb = (key[key_idx] & (1 << (7-j))) >> (7-j);
			grain->auth_acc[8 * i + j] = next_z_host(grain, fp1_fb);
		}
		key_idx++;
	}

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			uint8_t fp1_fb = (key[key_idx] & (1 << (7-j))) >> (7-j);
			grain->auth_sr[8 * i + j] = next_z_host(grain, fp1_fb);
		}
		key_idx++;
	}

	grain_round_host = NORMAL;



	uint16_t ks_cnt = 0;

	/* generate keystream */
	for (int i = 0; i < STREAM_BYTES; i++) {
		/* every second bit is used for keystream, the others for MAC */
		for (int j = 0; j < 16; j++) {
			uint8_t z_next = next_z_host(grain, 0);
			if (j % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 

			
		}
	}


}

uint8_t to_MSB_host(uint8_t b){
	return 	(b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

uint8_t encrypt_host(uint8_t *iv, uint8_t *key) // you have to add out_result
{
	
	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain_host(&grain, key, iv);
	grain_round_host = INIT;

	for (int i = 0; i < 256; i++) {
#ifdef INIT
		//printf("%d", next_z(&grain));
		next_z_host(&grain);
#else
		next_z_host(&grain, 0);
#endif
	}

	grain_round_host = NORMAL;

	generate_keystream_host(&grain, &data, key, out);

	return out[BYTE_POSITION]&BIT_POSITION; // return fisrt bit position (MSB), 1 or 0
}



/*

GPU Grain

*/

__device__ uint8_t shift(uint8_t fsr[128], uint8_t fb)
{
	uint8_t out = fsr[0];
	for (int i = 0; i < 127; i++) {
		fsr[i] = fsr[i+1];
	}
	fsr[127] = fb;

	return out;
}

__device__ void init_grain(grain_state *grain, uint8_t *key, uint8_t *iv){
	// expand the packed bytes and place one bit per array cell (like a flip flop in HW)
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 8; j++) {
			grain->lfsr[8 * i + j] = (iv[i] & (1 << (7-j))) >> (7-j);
		}
	}

	for (int i = 96; i < 127; i++) {
		grain->lfsr[i] = 1;
	}

	grain->lfsr[127] = 0;

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 8; j++) {
			grain->nfsr[8 * i + j] = (key[i] & (1 << (7-j))) >> (7-j);
		}
	}

	for (int i = 0; i < 64; i++) {
		grain->auth_acc[i] = 0;
		grain->auth_sr[i] = 0;
	}
}



__device__ uint8_t next_lfsr_fb(grain_state *grain)
{
 	/* f(x) = 1 + x^32 + x^47 + x^58 + x^90 + x^121 + x^128 */
	return grain->lfsr[96] ^ grain->lfsr[81] ^ grain->lfsr[70] ^ grain->lfsr[38] ^ grain->lfsr[7] ^ grain->lfsr[0];
}

__device__ uint8_t next_nfsr_fb(grain_state *grain)
{
	return grain->nfsr[96] ^ grain->nfsr[91] ^ grain->nfsr[56] ^ grain->nfsr[26] ^ grain->nfsr[0] ^ (grain->nfsr[84] & grain->nfsr[68]) ^
			(grain->nfsr[67] & grain->nfsr[3]) ^ (grain->nfsr[65] & grain->nfsr[61]) ^ (grain->nfsr[59] & grain->nfsr[27]) ^
			(grain->nfsr[48] & grain->nfsr[40]) ^ (grain->nfsr[18] & grain->nfsr[17]) ^ (grain->nfsr[13] & grain->nfsr[11]) ^
			(grain->nfsr[82] & grain->nfsr[78] & grain->nfsr[70]) ^ (grain->nfsr[25] & grain->nfsr[24] & grain->nfsr[22]) ^
			(grain->nfsr[95] & grain->nfsr[93] & grain->nfsr[92] & grain->nfsr[88]);
}

__device__ uint8_t next_h(grain_state *grain)
{
	// h(x) = x0x1 + x2x3 + x4x5 + x6x7 + x0x4x8
	#define x0 grain->nfsr[12]	// bi+12
	#define x1 grain->lfsr[8]		// si+8
	#define x2 grain->lfsr[13]	// si+13
	#define x3 grain->lfsr[20]	// si+20
	#define x4 grain->nfsr[95]	// bi+95
	#define x5 grain->lfsr[42]	// si+42
	#define x6 grain->lfsr[60]	// si+60
	#define x7 grain->lfsr[79]	// si+79
	#define x8 grain->lfsr[94]	// si+94

	uint8_t h_out = (x0 & x1) ^ (x2 & x3) ^ (x4 & x5) ^ (x6 & x7) ^ (x0 & x4 & x8);
	return h_out;
}







__device__ void print_state(grain_state *grain)
{
	printf("LFSR: ");
	for (int i = 0; i < 128; i++) {
		printf("%d", grain->lfsr[i]);
	}
	printf("\nNFSR: ");
	for (int i = 0; i < 128; i++) {
		printf("%d", grain->nfsr[i]);
	}
	printf("\n");
}

__device__ void print_stream(uint8_t *stream, uint8_t byte_size)
{
	for (int i = 0; i < byte_size; i++) {
		uint8_t yi = 0;
		for (int j = 0; j < 8; j++) {
			yi = (yi << 1) ^ stream[i * 8 + j];
		}
		printf("%02x", yi);
	}
	printf("\n");
}

__device__ uint8_t next_z(grain_state *grain, uint8_t keybit){
	uint8_t lfsr_fb = next_lfsr_fb(grain);
	uint8_t nfsr_fb = next_nfsr_fb(grain);
	uint8_t h_out = next_h(grain);

	/* y = h + s_{i+93} + sum(b_{i+j}), j \in A */
	uint8_t A[] = {2, 15, 36, 45, 64, 73, 89};

	uint8_t nfsr_tmp = 0;
	for (int i = 0; i < 7; i++) {
		nfsr_tmp ^= grain->nfsr[A[i]];
	}

	uint8_t y = h_out ^ grain->lfsr[93] ^ nfsr_tmp;
	
	uint8_t lfsr_out;

	/* feedback y if we are in the initialization instance */
	if (grain_round == INIT) {
		lfsr_out = shift(grain->lfsr, lfsr_fb ^ y);
		shift(grain->nfsr, nfsr_fb ^ lfsr_out ^ y);
	} else if (grain_round == FP1) {
		lfsr_out = shift(grain->lfsr, lfsr_fb ^ keybit);
		shift(grain->nfsr, nfsr_fb ^ lfsr_out);
	} else if (grain_round == NORMAL) {
		lfsr_out = shift(grain->lfsr, lfsr_fb);
		shift(grain->nfsr, nfsr_fb ^ lfsr_out);
	}

	return y;
}

__device__ void generate_keystream(grain_state *grain, grain_data *data, uint8_t *key,uint8_t *ks){
	grain_round = FP1;
	uint8_t key_idx = 0;
	/* inititalize GRAIN - change it will affect even preout result*/
	
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			uint8_t fp1_fb = (key[key_idx] & (1 << (7-j))) >> (7-j);
			grain->auth_acc[8 * i + j] = next_z(grain, fp1_fb);
		}
		key_idx++;
	}

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			uint8_t fp1_fb = (key[key_idx] & (1 << (7-j))) >> (7-j);
			grain->auth_sr[8 * i + j] = next_z(grain, fp1_fb);
		}
		key_idx++;
	}
	
	

	grain_round = NORMAL;

	uint16_t ks_cnt = 0;


	/* generate keystream */
	for (int i = 0; i < STREAM_BYTES; i++) {
		/* every second bit is used for keystream, the others for MAC */
		for (int j = 0; j < 16; j++) {
			uint8_t z_next = next_z(grain, 0);
			if (j % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 
	
		}
		
	}

}

// size key 16, iv 12, msg 8
__device__ uint8_t encrypt(uint8_t *iv, uint8_t *key){
	

	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain(&grain, key, iv);


	/* initialize grain and skip output */
	grain_round = INIT;

	for (int i = 0; i < 256; i++) {
#ifdef INIT
		//printf("%d", next_z(&grain));
		next_z(&grain);
#else
		next_z(&grain, 0);
#endif
	}

	grain_round = NORMAL;

	generate_keystream(&grain, &data, key, out);

	return out[BYTE_POSITION]&BIT_POSITION; // return fisrt bit position (MSB), 1 or 0
}

__device__ void print_iv_key(uint8_t *key,uint8_t *iv){

	printf("key:\t\t");
	for (int i = 0; i < 16; i++) {
		printf("%02x", key[i]);
	}
	printf("\n");

	printf("iv:\t\t");
	for (int i = 0; i < 12; i++) {
		printf("%02x", iv[i]);
	}
	printf("\n");
}




/*
Utils Functions:
*/

__device__ int value_in_array(uint8_t val,uint8_t len,uint8_t arr[]){
    for(int i = 0; i < len; i++){
		if(arr[i] == val)
			return 1;
	}
	return 0;
}

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
__device__ void print_iv(uint8_t iv[]){
	printf("\n IV vector = [ ");
	for(int i=0;i<IV_dim;i++){
		printf("%u ",iv[i]);
	}
	printf("]\n");
}

__device__ void flush_iv(uint8_t iv[]){
	for(int i=0;i<IV_dim;i++){
		iv[i]=0;
	}
}

__device__ void print_key(uint8_t k[]){
	printf("\n K vector = [ ");
	for(int i=0;i<K_dim;i++){
		printf("%u ",k[i]);
	}
	printf("]\n");
}

__device__ void print_I(uint8_t k, uint8_t *I_i){
	printf("\n LOG I \n");
	for(int i=0;i<k;i++){
		printf("\trandom term = %d\n",I_i[i]);
	}
	printf("\n");
}

__device__ void remove_from_I(uint8_t j,uint8_t k, uint8_t *I_i){
	I_i[k-1] = I_i[j];
}

__device__ void random_key(uint8_t k[],curandState *states){
	for(int i=0;i<K_dim;i++){
		k[i]=curand_uniform(states)*((1<<8)-1);
	}
}

__device__ void xor_key(uint8_t k_0[],uint8_t k_1[],uint8_t k_xor[]){
	for(int i=0;i<K_dim;i++){
		k_xor[i] = k_0[i]^k_1[i];
	}
}

__device__ void flush_key(uint8_t k[]){
	for(int i=0;i<K_dim;i++)
		k[i]=0;
}

__device__ void flush_I(uint8_t k,uint8_t *I_i){
	for(int i=0;i<k;i++) 
		I_i[i]=0;
}


__device__ void random_term(uint8_t k, uint8_t *I_i,curandState *states){
	uint8_t t_1;
	
	for(int i=0;i<k;i++){
		do{
		 // 96 bit from 0 to 95
		 t_1 = curand_uniform(states)*(IV_dim*8-1);
		}while(value_in_array(t_1,k,I_i));
		I_i[i]=t_1;
	}

}

/*
FROM GRAIN Paper:

key example 0x01234FFFFFFFFFFFFFFFFFFFFFFFFFFF

repr

(1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1...,1).


'10000000'+'11000100' ...

hex(int('100011',2)) = 0x23


Even for IV and messages

*/

__device__ uint8_t to_MSB(uint8_t b){
	return 	(b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

__device__ void IV_gen(uint64_t tid, uint8_t len, uint8_t cube_index[],uint8_t *iv){
	uint8_t c_i;
	flush_iv(iv);
	for(unsigned int i=0; i<len;i++){
		c_i = ((tid/(1ull<<i)))%2;
		iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8)));
	}

}