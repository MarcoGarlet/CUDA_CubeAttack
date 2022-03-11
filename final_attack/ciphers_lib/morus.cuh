#include <string.h>

#ifdef _MSC_VER
#define inline __inline
#endif

#define nn1 5 
#define nn2 31
#define nn3 7
#define nn4 22
#define nn5 13

#define K_dim   16
#define IV_dim  16

#define BYTE_POSITION 3
#define BIT_POSITION_APP 7//1//3
#define BIT_POSITION 1<<BIT_POSITION_APP//2//8

#define N_ROUND 4

#define BIT_K (K_dim*8)
#define BIT_I (IV_dim*8)

#define rotl(x,n)   (((x) << (n)) | ((x) >> (32-n)))  

__device__ uint8_t d_key_2_guess[K_dim];
uint8_t h_key_2_guess[K_dim];






inline void morus_stateupdate_host(unsigned int msgblk[], unsigned int state[][4])    
{   
        unsigned int temp;  

        state[0][0] ^= state[3][0]; state[0][1] ^= state[3][1]; state[0][2] ^= state[3][2]; state[0][3] ^= state[3][3]; 
        state[0][0] ^= state[1][0] & state[2][0]; state[0][1] ^= state[1][1] & state[2][1]; state[0][2] ^= state[1][2] & state[2][2]; state[0][3] ^= state[1][3] & state[2][3];     
        state[0][0] = rotl(state[0][0],nn1);  state[0][1] = rotl(state[0][1],nn1);       state[0][2] = rotl(state[0][2],nn1);       state[0][3] = rotl(state[0][3],nn1);  
        temp = state[3][3];    state[3][3] = state[3][2];  state[3][2] = state[3][1];  state[3][1] = state[3][0];  state[3][0] = temp;  

        state[1][0] ^= msgblk[0];   state[1][1] ^= msgblk[1];   state[1][2] ^= msgblk[2];   state[1][3] ^= msgblk[3];
        state[1][0] ^= state[4][0]; state[1][1] ^= state[4][1]; state[1][2] ^= state[4][2]; state[1][3] ^= state[4][3]; 
        state[1][0] ^= (state[2][0] & state[3][0]); state[1][1] ^= (state[2][1] & state[3][1]); state[1][2] ^= (state[2][2] & state[3][2]); state[1][3] ^= (state[2][3] & state[3][3]);     
        state[1][0] = rotl(state[1][0],nn2);  state[1][1] = rotl(state[1][1],nn2);       state[1][2] = rotl(state[1][2],nn2);       state[1][3] = rotl(state[1][3],nn2); 
        temp = state[4][3];    state[4][3] = state[4][1];  state[4][1] = temp;     
        temp = state[4][2];    state[4][2] = state[4][0];  state[4][0] = temp;     

        state[2][0] ^= msgblk[0];   state[2][1] ^= msgblk[1];   state[2][2] ^= msgblk[2];   state[2][3] ^= msgblk[3];
        state[2][0] ^= state[0][0]; state[2][1] ^= state[0][1]; state[2][2] ^= state[0][2]; state[2][3] ^= state[0][3]; 
        state[2][0] ^= state[3][0] & state[4][0]; state[2][1] ^= state[3][1] & state[4][1]; state[2][2] ^= state[3][2] & state[4][2]; state[2][3] ^= state[3][3] & state[4][3];     
        state[2][0] = rotl(state[2][0],nn3);  state[2][1] = rotl(state[2][1],nn3);       state[2][2] = rotl(state[2][2],nn3);       state[2][3] = rotl(state[2][3],nn3);  
        temp = state[0][0];    state[0][0] = state[0][1];  state[0][1] = state[0][2];  state[0][2] = state[0][3];  state[0][3] = temp;  

        state[3][0] ^= msgblk[0];   state[3][1] ^= msgblk[1];   state[3][2] ^= msgblk[2];   state[3][3] ^= msgblk[3];
        state[3][0] ^= state[1][0]; state[3][1] ^= state[1][1]; state[3][2] ^= state[1][2]; state[3][3] ^= state[1][3]; 
        state[3][0] ^= state[4][0] & state[0][0]; state[3][1] ^= state[4][1] & state[0][1]; state[3][2] ^= state[4][2] & state[0][2]; state[3][3] ^= state[4][3] & state[0][3];     
        state[3][0] = rotl(state[3][0],nn4);  state[3][1] = rotl(state[3][1],nn4);       state[3][2] = rotl(state[3][2],nn4);       state[3][3] = rotl(state[3][3],nn4);  
        temp = state[1][3];    state[1][3] = state[1][1];  state[1][1] = temp;     
        temp = state[1][2];    state[1][2] = state[1][0];  state[1][0] = temp;     

        state[4][0] ^= msgblk[0];   state[4][1] ^= msgblk[1];   state[4][2] ^= msgblk[2];   state[4][3] ^= msgblk[3];
        state[4][0] ^= state[2][0]; state[4][1] ^= state[2][1]; state[4][2] ^= state[2][2]; state[4][3] ^= state[2][3]; 
        state[4][0] ^= state[0][0] & state[1][0]; state[4][1] ^= state[0][1] & state[1][1]; state[4][2] ^= state[0][2] & state[1][2]; state[4][3] ^= state[0][3] & state[1][3];     
        state[4][0] = rotl(state[4][0],nn5);  state[4][1] = rotl(state[4][1],nn5);       state[4][2] = rotl(state[4][2],nn5);       state[4][3] = rotl(state[4][3],nn5);  
        temp = state[2][3];    state[2][3] = state[2][2];  state[2][2] = state[2][1];  state[2][1] = state[2][0];  state[2][0] = temp;  
}

/*The input to the initialization is the 128-bit key; 128-bit IV;*/
 void morus_initialization_host(const unsigned char *key, const unsigned char *iv, unsigned int state[][4])
{
        int i;
        unsigned int temp[4]  = {0,0,0,0}; 
        unsigned char con0[16] = {0x0,0x1,0x01,0x02,0x03,0x05,0x08,0x0d,0x15,0x22,0x37,0x59,0x90,0xe9,0x79,0x62}; 
    unsigned char con1[16] = {0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd}; 

    memcpy(state[0], iv,   16);
        memcpy(state[1], key,  16);  
        memset(state[2], 0xff, 16);   
        memcpy(state[3], con0, 16);  
        memcpy(state[4], con1, 16);  

        for (i = 0; i < 4;  i++) temp[i] = 0;  
        for (i = 0; i < N_ROUND; i++) morus_stateupdate_host(temp, state); 
        for (i = 0; i < 4;  i++) state[1][i] ^= ((unsigned int*)key)[i]; 
}








//encrypt a message
uint8_t encrypt_host(const unsigned char *npub,const unsigned char *k)
{
        unsigned long long i;
        unsigned char plaintextblock[16], ciphertextblock[16];
        unsigned int morus_state[5][4];

        //initialization 
        morus_initialization_host(k, npub, morus_state);

        unsigned int o[] = {morus_state[0][0] ^ morus_state[1][1] ^ (morus_state[2][0] & morus_state[3][0]),morus_state[0][1] ^ morus_state[1][2] ^ (morus_state[2][1] & morus_state[3][1]),morus_state[0][2] ^ morus_state[1][3] ^ (morus_state[2][2] & morus_state[3][2]),morus_state[0][3] ^ morus_state[1][0] ^ (morus_state[2][3] & morus_state[3][3])};
        return ((o[BYTE_POSITION])&BIT_POSITION)>>BIT_POSITION_APP;

}




//encrypt a message
uint8_t encrypt_exploit_host(const unsigned char *npub)
{
        unsigned long long i;
        unsigned char plaintextblock[16], ciphertextblock[16];
        unsigned int morus_state[5][4];

        //initialization 
        morus_initialization_host(h_key_2_guess, npub, morus_state);

        unsigned int o[] = {morus_state[0][0] ^ morus_state[1][1] ^ (morus_state[2][0] & morus_state[3][0]),morus_state[0][1] ^ morus_state[1][2] ^ (morus_state[2][1] & morus_state[3][1]),morus_state[0][2] ^ morus_state[1][3] ^ (morus_state[2][2] & morus_state[3][2]),morus_state[0][3] ^ morus_state[1][0] ^ (morus_state[2][3] & morus_state[3][3])};
        return ((o[BYTE_POSITION])&BIT_POSITION)>>BIT_POSITION_APP;

}











__device__ inline void morus_stateupdate(unsigned int msgblk[], unsigned int state[][4])    
{   
        unsigned int temp;  

        state[0][0] ^= state[3][0]; state[0][1] ^= state[3][1]; state[0][2] ^= state[3][2]; state[0][3] ^= state[3][3]; 
        state[0][0] ^= state[1][0] & state[2][0]; state[0][1] ^= state[1][1] & state[2][1]; state[0][2] ^= state[1][2] & state[2][2]; state[0][3] ^= state[1][3] & state[2][3];     
        state[0][0] = rotl(state[0][0],nn1);  state[0][1] = rotl(state[0][1],nn1);       state[0][2] = rotl(state[0][2],nn1);       state[0][3] = rotl(state[0][3],nn1);  
        temp = state[3][3];    state[3][3] = state[3][2];  state[3][2] = state[3][1];  state[3][1] = state[3][0];  state[3][0] = temp;  

        state[1][0] ^= msgblk[0];   state[1][1] ^= msgblk[1];   state[1][2] ^= msgblk[2];   state[1][3] ^= msgblk[3];
        state[1][0] ^= state[4][0]; state[1][1] ^= state[4][1]; state[1][2] ^= state[4][2]; state[1][3] ^= state[4][3]; 
        state[1][0] ^= (state[2][0] & state[3][0]); state[1][1] ^= (state[2][1] & state[3][1]); state[1][2] ^= (state[2][2] & state[3][2]); state[1][3] ^= (state[2][3] & state[3][3]);     
        state[1][0] = rotl(state[1][0],nn2);  state[1][1] = rotl(state[1][1],nn2);       state[1][2] = rotl(state[1][2],nn2);       state[1][3] = rotl(state[1][3],nn2); 
        temp = state[4][3];    state[4][3] = state[4][1];  state[4][1] = temp;     
        temp = state[4][2];    state[4][2] = state[4][0];  state[4][0] = temp;     

        state[2][0] ^= msgblk[0];   state[2][1] ^= msgblk[1];   state[2][2] ^= msgblk[2];   state[2][3] ^= msgblk[3];
        state[2][0] ^= state[0][0]; state[2][1] ^= state[0][1]; state[2][2] ^= state[0][2]; state[2][3] ^= state[0][3]; 
        state[2][0] ^= state[3][0] & state[4][0]; state[2][1] ^= state[3][1] & state[4][1]; state[2][2] ^= state[3][2] & state[4][2]; state[2][3] ^= state[3][3] & state[4][3];     
        state[2][0] = rotl(state[2][0],nn3);  state[2][1] = rotl(state[2][1],nn3);       state[2][2] = rotl(state[2][2],nn3);       state[2][3] = rotl(state[2][3],nn3);  
        temp = state[0][0];    state[0][0] = state[0][1];  state[0][1] = state[0][2];  state[0][2] = state[0][3];  state[0][3] = temp;  

        state[3][0] ^= msgblk[0];   state[3][1] ^= msgblk[1];   state[3][2] ^= msgblk[2];   state[3][3] ^= msgblk[3];
        state[3][0] ^= state[1][0]; state[3][1] ^= state[1][1]; state[3][2] ^= state[1][2]; state[3][3] ^= state[1][3]; 
        state[3][0] ^= state[4][0] & state[0][0]; state[3][1] ^= state[4][1] & state[0][1]; state[3][2] ^= state[4][2] & state[0][2]; state[3][3] ^= state[4][3] & state[0][3];     
        state[3][0] = rotl(state[3][0],nn4);  state[3][1] = rotl(state[3][1],nn4);       state[3][2] = rotl(state[3][2],nn4);       state[3][3] = rotl(state[3][3],nn4);  
        temp = state[1][3];    state[1][3] = state[1][1];  state[1][1] = temp;     
        temp = state[1][2];    state[1][2] = state[1][0];  state[1][0] = temp;     

        state[4][0] ^= msgblk[0];   state[4][1] ^= msgblk[1];   state[4][2] ^= msgblk[2];   state[4][3] ^= msgblk[3];
        state[4][0] ^= state[2][0]; state[4][1] ^= state[2][1]; state[4][2] ^= state[2][2]; state[4][3] ^= state[2][3]; 
        state[4][0] ^= state[0][0] & state[1][0]; state[4][1] ^= state[0][1] & state[1][1]; state[4][2] ^= state[0][2] & state[1][2]; state[4][3] ^= state[0][3] & state[1][3];     
        state[4][0] = rotl(state[4][0],nn5);  state[4][1] = rotl(state[4][1],nn5);       state[4][2] = rotl(state[4][2],nn5);       state[4][3] = rotl(state[4][3],nn5);  
        temp = state[2][3];    state[2][3] = state[2][2];  state[2][2] = state[2][1];  state[2][1] = state[2][0];  state[2][0] = temp;  
}

/*The input to the initialization is the 128-bit key; 128-bit IV;*/
__device__ void morus_initialization(const unsigned char *key, const unsigned char *iv, unsigned int state[][4])
{
        int i;
        unsigned int temp[4]  = {0,0,0,0}; 
        unsigned char con0[16] = {0x0,0x1,0x01,0x02,0x03,0x05,0x08,0x0d,0x15,0x22,0x37,0x59,0x90,0xe9,0x79,0x62}; 
    unsigned char con1[16] = {0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd}; 

    memcpy(state[0], iv,   16);
        memcpy(state[1], key,  16);  
        memset(state[2], 0xff, 16);   
        memcpy(state[3], con0, 16);  
        memcpy(state[4], con1, 16);  

        for (i = 0; i < 4;  i++) temp[i] = 0;  
        for (i = 0; i < N_ROUND; i++) morus_stateupdate(temp, state); 
        for (i = 0; i < 4;  i++) state[1][i] ^= ((unsigned int*)key)[i]; 
}








//encrypt a message
__device__ uint8_t encrypt(const unsigned char *npub,const unsigned char *k)
{
        unsigned long long i;
        unsigned char plaintextblock[16], ciphertextblock[16];
        unsigned int morus_state[5][4];

        morus_initialization(k, npub, morus_state);

        unsigned int o[] = {morus_state[0][0] ^ morus_state[1][1] ^ (morus_state[2][0] & morus_state[3][0]),morus_state[0][1] ^ morus_state[1][2] ^ (morus_state[2][1] & morus_state[3][1]),morus_state[0][2] ^ morus_state[1][3] ^ (morus_state[2][2] & morus_state[3][2]),morus_state[0][3] ^ morus_state[1][0] ^ (morus_state[2][3] & morus_state[3][3])};
        return ((o[BYTE_POSITION])&BIT_POSITION)>>BIT_POSITION_APP;


}




//encrypt a message
__device__ uint8_t encrypt_exploit(const unsigned char *npub)
{
        unsigned long long i;
        unsigned char plaintextblock[16], ciphertextblock[16];
        unsigned int morus_state[5][4];

        //initialization 
        morus_initialization(d_key_2_guess, npub, morus_state);

        
        unsigned int o[] = {morus_state[0][0] ^ morus_state[1][1] ^ (morus_state[2][0] & morus_state[3][0]),morus_state[0][1] ^ morus_state[1][2] ^ (morus_state[2][1] & morus_state[3][1]),morus_state[0][2] ^ morus_state[1][3] ^ (morus_state[2][2] & morus_state[3][2]),morus_state[0][3] ^ morus_state[1][0] ^ (morus_state[2][3] & morus_state[3][3])};
        return ((o[BYTE_POSITION])&BIT_POSITION)>>BIT_POSITION_APP;



}



