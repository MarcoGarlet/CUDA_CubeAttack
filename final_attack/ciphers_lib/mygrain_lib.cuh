#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>


#ifndef UTILS_H
#define UTILS_H


#define STREAM_BYTES	16
#define MSG_BYTES		0
#define BIT_POSITION_APP 0//1//3
#define BIT_POSITION 1<<BIT_POSITION_APP//2//8
#define N_ROUND		32

#define K_dim	16
#define IV_dim	12

#define BIT_K (K_dim*8)
#define BIT_I (IV_dim*8)

__device__ uint8_t d_key_2_guess[K_dim];
uint8_t h_key_2_guess[K_dim];

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
__device__ void init_grain_unroll(grain_state *grain, uint8_t *key, uint8_t *iv);
__device__ uint8_t next_lfsr_fb(grain_state *grain);
__device__ uint8_t next_nfsr_fb(grain_state *grain);
__device__ uint8_t next_h(grain_state *grain);
__device__ uint8_t shift(uint8_t fsr[128], uint8_t fb);

__device__ uint8_t next_z(grain_state *grain, uint8_t);
__device__ uint8_t next_z_unroll(grain_state *grain, uint8_t);
__device__ uint8_t* generate_keystream(grain_state *grain, grain_data *data, uint8_t *);
__device__ uint8_t* generate_keystream_unroll(grain_state *grain, grain_data *data, uint8_t *);
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



uint8_t encrypt_exploit_host(uint8_t *iv){
	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain_host(&grain, h_key_2_guess, iv);
	grain_round_host = INIT;

	for (int i = 0; i < N_ROUND; i++) {
#ifdef INIT
		//printf("%d", next_z(&grain));
		next_z_host(&grain);
#else
		next_z_host(&grain, 0);
#endif
	}

	grain_round_host = NORMAL;

	generate_keystream_host(&grain, &data, h_key_2_guess, out);
	
	//return (out[0]&BIT_POSITION)>>BIT_POSITION_APP; // return fisrt bit position (MSB), 1 or 0
	return out[BIT_POSITION];
}

uint8_t encrypt_host(uint8_t *iv, uint8_t *key) // you have to add out_result
{
	
	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain_host(&grain, key, iv);
	grain_round_host = INIT;

	for (int i = 0; i < N_ROUND; i++) {
#ifdef INIT
		//printf("%d", next_z(&grain));
		next_z_host(&grain);
#else
		next_z_host(&grain, 0);
#endif
	}

	grain_round_host = NORMAL;

	generate_keystream_host(&grain, &data, key, out);
	
	//return (out[0]&BIT_POSITION)>>BIT_POSITION_APP; // return fisrt bit position (MSB), 1 or 0
	return out[BIT_POSITION];
}



/*

GPU Grain

*/



__device__ void init_grain_unroll(grain_state *grain, uint8_t *key, uint8_t *iv){
	// expand the packed bytes and place one bit per array cell (like a flip flop in HW)
	for (int i = 0; i < 12; i+=4) {
		grain->lfsr[8 * i] = (iv[i] & (1 <<7)) >> 7;
		grain->lfsr[8 * i + 1] = (iv[i] & (1 << 6)) >> 6;
		grain->lfsr[8 * i + 2] = (iv[i] & (1 << 5)) >> 5;
		grain->lfsr[8 * i + 3] = (iv[i] & (1 << 4)) >> 4;
		grain->lfsr[8 * i + 4] = (iv[i] & (1 << 3)) >> 3;
		grain->lfsr[8 * i + 5] = (iv[i] & (1 << 2)) >> 2;
		grain->lfsr[8 * i + 6] = (iv[i] & (1 << 1)) >> 1;
		grain->lfsr[8 * i + 7] = (iv[i] &1);

		grain->lfsr[8 * (i+1)] = (iv[i+1] & (1 <<7)) >> 7;
		grain->lfsr[8 * (i+1) + 1] = (iv[i+1] & (1 << 6)) >> 6;
		grain->lfsr[8 * (i+1) + 2] = (iv[i+1] & (1 << 5)) >> 5;
		grain->lfsr[8 * (i+1) + 3] = (iv[i+1] & (1 << 4)) >> 4;
		grain->lfsr[8 * (i+1) + 4] = (iv[i+1] & (1 << 3)) >> 3;
		grain->lfsr[8 * (i+1) + 5] = (iv[i+1] & (1 << 2)) >> 2;
		grain->lfsr[8 * (i+1) + 6] = (iv[i+1] & (1 << 1)) >> 1;
		grain->lfsr[8 * (i+1) + 7] = (iv[i+1] &1);

		grain->lfsr[8 * (i+2)] = (iv[i+2] & (1 <<7)) >> 7;
		grain->lfsr[8 * (i+2) + 1] = (iv[i+2] & (1 << 6)) >> 6;
		grain->lfsr[8 * (i+2) + 2] = (iv[i+2] & (1 << 5)) >> 5;
		grain->lfsr[8 * (i+2) + 3] = (iv[i+2] & (1 << 4)) >> 4;
		grain->lfsr[8 * (i+2) + 4] = (iv[i+2] & (1 << 3)) >> 3;
		grain->lfsr[8 * (i+2) + 5] = (iv[i+2] & (1 << 2)) >> 2;
		grain->lfsr[8 * (i+2) + 6] = (iv[i+2] & (1 << 1)) >> 1;
		grain->lfsr[8 * (i+2) + 7] = (iv[i+2] &1);

		grain->lfsr[8 * (i+3)] = (iv[i+3] & (1 <<7)) >> 7;
		grain->lfsr[8 * (i+3) + 1] = (iv[i+3] & (1 << 6)) >> 6;
		grain->lfsr[8 * (i+3) + 2] = (iv[i+3] & (1 << 5)) >> 5;
		grain->lfsr[8 * (i+3) + 3] = (iv[i+3] & (1 << 4)) >> 4;
		grain->lfsr[8 * (i+3) + 4] = (iv[i+3] & (1 << 3)) >> 3;
		grain->lfsr[8 * (i+3) + 5] = (iv[i+3] & (1 << 2)) >> 2;
		grain->lfsr[8 * (i+3) + 6] = (iv[i+3] & (1 << 1)) >> 1;
		grain->lfsr[8 * (i+3) + 7] = (iv[i+3] &1);


	}

	grain->lfsr[96] = 1;
	for (int i = 0; i < 30; i+=3) {
		grain->lfsr[i+97] = 1;
		grain->lfsr[i+97+1] = 1;
		grain->lfsr[i+97+2] = 1;
	}

	grain->lfsr[127] = 0;

	for (int i = 0; i < 16; i+=4) {

		grain->nfsr[8 * i] = (key[i] & (1 << 7)) >> 7;
		grain->nfsr[8 * i + 1] = (key[i] & (1 << 6)) >> 6;
		grain->nfsr[8 * i + 2] = (key[i] & (1 << 5)) >> 5;
		grain->nfsr[8 * i + 3] = (key[i] & (1 << 4)) >> 4;
		grain->nfsr[8 * i + 4] = (key[i] & (1 << 3)) >> 3;
		grain->nfsr[8 * i + 5] = (key[i] & (1 << 2)) >> 2;
		grain->nfsr[8 * i + 6] = (key[i] & (1 << 1)) >> 1;
		grain->nfsr[8 * i + 7] = (key[i] & 1);

		grain->nfsr[8 * (i+1)] = (key[i+1] & (1 << 7)) >> 7;
		grain->nfsr[8 * (i+1) + 1] = (key[i+1] & (1 << 6)) >> 6;
		grain->nfsr[8 * (i+1) + 2] = (key[i+1] & (1 << 5)) >> 5;
		grain->nfsr[8 * (i+1) + 3] = (key[i+1] & (1 << 4)) >> 4;
		grain->nfsr[8 * (i+1) + 4] = (key[i+1] & (1 << 3)) >> 3;
		grain->nfsr[8 * (i+1) + 5] = (key[i+1] & (1 << 2)) >> 2;
		grain->nfsr[8 * (i+1) + 6] = (key[i+1] & (1 << 1)) >> 1;
		grain->nfsr[8 * (i+1) + 7] = (key[i+1] & 1);

		grain->nfsr[8 * (i+2)] = (key[i+2] & (1 << 7)) >> 7;
		grain->nfsr[8 * (i+2) + 1] = (key[i+2] & (1 << 6)) >> 6;
		grain->nfsr[8 * (i+2) + 2] = (key[i+2] & (1 << 5)) >> 5;
		grain->nfsr[8 * (i+2) + 3] = (key[i+2] & (1 << 4)) >> 4;
		grain->nfsr[8 * (i+2) + 4] = (key[i+2] & (1 << 3)) >> 3;
		grain->nfsr[8 * (i+2) + 5] = (key[i+2] & (1 << 2)) >> 2;
		grain->nfsr[8 * (i+2) + 6] = (key[i+2] & (1 << 1)) >> 1;
		grain->nfsr[8 * (i+2) + 7] = (key[i+2] & 1);

		grain->nfsr[8 * (i+3)] = (key[i+3] & (1 << 7)) >> 7;
		grain->nfsr[8 * (i+3) + 1] = (key[i+3] & (1 << 6)) >> 6;
		grain->nfsr[8 * (i+3) + 2] = (key[i+3] & (1 << 5)) >> 5;
		grain->nfsr[8 * (i+3) + 3] = (key[i+3] & (1 << 4)) >> 4;
		grain->nfsr[8 * (i+3) + 4] = (key[i+3] & (1 << 3)) >> 3;
		grain->nfsr[8 * (i+3) + 5] = (key[i+3] & (1 << 2)) >> 2;
		grain->nfsr[8 * (i+3) + 6] = (key[i+3] & (1 << 1)) >> 1;
		grain->nfsr[8 * (i+3) + 7] = (key[i+3] & 1);

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

	uint8_t h_out = (grain->nfsr[12] & grain->lfsr[8]) ^ (grain->lfsr[13] & grain->lfsr[20]) ^ (grain->nfsr[95] & grain->lfsr[42]) ^ (grain->lfsr[60] & grain->lfsr[79]) ^ (grain->nfsr[12] & grain->nfsr[95] & grain->lfsr[94]);
	return h_out;
}


__device__ uint8_t next_z_unroll(grain_state *grain, uint8_t keybit){
	uint8_t lfsr_fb = next_lfsr_fb(grain);
	uint8_t nfsr_fb = next_nfsr_fb(grain);
	uint8_t h_out = next_h(grain);

	/* y = h + s_{i+93} + sum(b_{i+j}), j \in A */
	uint8_t A[] = {2, 15, 36, 45, 64, 73, 89};

	uint8_t nfsr_tmp = 0;
	nfsr_tmp ^= grain->nfsr[A[0]];
	nfsr_tmp ^= grain->nfsr[A[1]];
	nfsr_tmp ^= grain->nfsr[A[2]];
	nfsr_tmp ^= grain->nfsr[A[3]];
	nfsr_tmp ^= grain->nfsr[A[4]];
	nfsr_tmp ^= grain->nfsr[A[5]];
	nfsr_tmp ^= grain->nfsr[A[6]];

	uint8_t y = h_out ^ grain->lfsr[93] ^ nfsr_tmp;
	
	uint8_t lfsr_out;

	/* feedback y if we are in the initialization instance */
	lfsr_out = grain->lfsr[0];
	for(int i=0;i<126;i+=6){
		grain->lfsr[i] = grain->lfsr[i+1];
		grain->lfsr[i+1] = grain->lfsr[i+2];
		grain->lfsr[i+2] = grain->lfsr[i+3];
		grain->lfsr[i+3] = grain->lfsr[i+4];
		grain->lfsr[i+4] = grain->lfsr[i+5];
		grain->lfsr[i+5] = grain->lfsr[i+6];

		grain->nfsr[i] = grain->nfsr[i+1];
		grain->nfsr[i+1] = grain->nfsr[i+2];
		grain->nfsr[i+2] = grain->nfsr[i+3];
		grain->nfsr[i+3] = grain->nfsr[i+4];
		grain->nfsr[i+4] = grain->nfsr[i+5];
		grain->nfsr[i+5] = grain->nfsr[i+6];
	}
	grain->lfsr[126] = grain->lfsr[127];
	grain->nfsr[126] = grain->nfsr[127];
	if (grain_round == INIT) {
		grain->lfsr[127]=lfsr_fb ^ y;
		grain->nfsr[127]=nfsr_fb ^ lfsr_out ^ y;
	} else if (grain_round == FP1) {
		grain->lfsr[127]=lfsr_fb ^ keybit;
		grain->nfsr[127]=nfsr_fb ^ lfsr_out;
	} else if (grain_round == NORMAL) {
		grain->lfsr[127]=lfsr_fb;
		grain->nfsr[127]=nfsr_fb ^ lfsr_out;
	}

	return y;
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


__device__ void generate_keystream_unroll(grain_state *grain, grain_data *data, uint8_t *key,uint8_t *ks){
	grain_round = FP1;
	uint8_t key_idx = 0;
	/* inititalize GRAIN - change it will affect even preout result*/
	uint8_t fp1_fb;
	for (int i = 0; i < 8; i+=2) {
		/*
			uint8_t fp1_fb = (key[key_idx] & (1 << (7-j))) >> (7-j);
			grain->auth_acc[8 * i + j] = next_z(grain, fp1_fb);
		*/
		fp1_fb = (key[key_idx] & (1 << 7)) >> 7;
		grain->auth_acc[8 * i] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 6)) >> 6;
		grain->auth_acc[8 * i + 1] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 5)) >> 5;
		grain->auth_acc[8 * i + 2] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 4)) >> 4;
		grain->auth_acc[8 * i + 3] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 3)) >> 3;
		grain->auth_acc[8 * i + 4] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 2)) >> 2;
		grain->auth_acc[8 * i + 5] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 1)) >> 1;
		grain->auth_acc[8 * i + 6] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & 1);
		grain->auth_acc[8 * i + 7] = next_z_unroll(grain, fp1_fb);

		key_idx++;
		
		fp1_fb = (key[key_idx] & (1 << 7)) >> 7;
		grain->auth_acc[8 * (i+1)] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 6)) >> 6;
		grain->auth_acc[8 * (i+1) + 1] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 5)) >> 5;
		grain->auth_acc[8 * (i+1) + 2] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 4)) >> 4;
		grain->auth_acc[8 * (i+1) + 3] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 3)) >> 3;
		grain->auth_acc[8 * (i+1) + 4] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 2)) >> 2;
		grain->auth_acc[8 * (i+1) + 5] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 1)) >> 1;
		grain->auth_acc[8 * (i+1) + 6] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & 1);
		grain->auth_acc[8 * (i+1) + 7] = next_z_unroll(grain, fp1_fb);

		key_idx++;
		
		
	}

	for (int i = 0; i < 8; i+=2) {
			
		fp1_fb = (key[key_idx] & (1 << 7)) >> 7;
		grain->auth_sr[8 * i] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 6)) >> 6;
		grain->auth_sr[8 * i + 1] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 5)) >> 5;
		grain->auth_sr[8 * i + 2] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 4)) >> 4;
		grain->auth_sr[8 * i + 3] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 3)) >> 3;
		grain->auth_sr[8 * i + 4] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 2)) >> 2;
		grain->auth_sr[8 * i + 5] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 1)) >> 1;
		grain->auth_sr[8 * i + 6] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & 1);
		grain->auth_sr[8 * i + 7] = next_z_unroll(grain, fp1_fb);

		key_idx++;
		
		fp1_fb = (key[key_idx] & (1 << 7)) >> 7;
		grain->auth_sr[8 * (i+1)] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 6)) >> 6;
		grain->auth_sr[8 * (i+1) + 1] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 5)) >> 5;
		grain->auth_sr[8 * (i+1) + 2] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 4)) >> 4;
		grain->auth_sr[8 * (i+1) + 3] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 3)) >> 3;
		grain->auth_sr[8 * (i+1) + 4] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 2)) >> 2;
		grain->auth_sr[8 * (i+1) + 5] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & (1 << 1)) >> 1;
		grain->auth_sr[8 * (i+1) + 6] = next_z_unroll(grain, fp1_fb);
		fp1_fb = (key[key_idx] & 1);
		grain->auth_sr[8 * (i+1) + 7] = next_z_unroll(grain, fp1_fb);

		key_idx++;

	}
	
	grain_round = NORMAL;
	uint16_t ks_cnt = 0;
	uint8_t z_next;
	/* generate keystream */
	for (int i = 0; i < STREAM_BYTES; i++) {
		/* every second bit is used for keystream, the others for MAC */
		for (int j = 0; j < 16; j+=4) {
			z_next = next_z_unroll(grain, 0);
			if (j % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 
			z_next = next_z_unroll(grain, 0);
			if ((j+1) % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 
			z_next = next_z_unroll(grain, 0);
			if ((j+2) % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 
			z_next = next_z_unroll(grain, 0);
			if ((j+3) % 2 == 0) {
				ks[ks_cnt] = z_next;
				ks_cnt++;
			} 
		}
		
	}

	

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


__device__ uint8_t encrypt(uint8_t *iv, uint8_t *key){
	

	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain_unroll(&grain, key, iv);
	/* initialize grain and skip output */
	grain_round = INIT;

	for (int i = 0; i < N_ROUND; i++) {
#ifdef INIT
		next_z_unroll(&grain);
#else
		next_z_unroll(&grain, 0);
#endif
	}

	grain_round = NORMAL;

	generate_keystream_unroll(&grain, &data, key, out);
	//return (out[0]&BIT_POSITION)>>BIT_POSITION_APP; // return fisrt bit position (MSB), 1 or 0
	return out[BIT_POSITION];
}

__device__ uint8_t encrypt_exploit(uint8_t *iv){
	

	uint8_t out[STREAM_BYTES * 8];
	grain_state grain;
	grain_data data;
	init_grain_unroll(&grain, d_key_2_guess, iv);
	/* initialize grain and skip output */
	grain_round = INIT;

	for (int i = 0; i < N_ROUND; i++) {
#ifdef INIT
		next_z_unroll(&grain);
#else
		next_z_unroll(&grain, 0);
#endif
	}

	grain_round = NORMAL;

	generate_keystream_unroll(&grain, &data, d_key_2_guess, out);
	//return (out[0]&BIT_POSITION)>>BIT_POSITION_APP; // return fisrt bit position (MSB), 1 or 0
	return out[BIT_POSITION];
}
