
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define K_dim	10
#define IV_dim	10

#define BYTE_POSITION	0

/*
      
  576R - bit 579 8 - 3
  672R - bit 673 2 - 1
*/

#define BIT_POSITION_APP 3//1//3
#define BIT_POSITION 1<<BIT_POSITION_APP//2//8


#define BIT_K (K_dim*8)
#define BIT_I (IV_dim*8)

#define N_ROUND 700//576//672//2*288

__device__ uint8_t d_key_2_guess[K_dim];
uint8_t h_key_2_guess[K_dim];

#define STATE_SIZE 36


void change_bit_host(uint8_t *arr, uint16_t n, uint8_t value)
{
  uint8_t nbyte = (n - 1) / 8;
  uint8_t nbit = ((n - 1) % 8) + 1;

  arr[nbyte] = ((255 << (9 - nbit)) & arr[nbyte]) |
               (value << (8 - nbit)) |
               ((255 >> nbit) & arr[nbyte]);
}

uint8_t nbit_host(uint8_t *arr, uint16_t n)
{
  uint8_t nbyte = (n - 1) / 8;
  uint8_t nbit = ((n - 1) % 8) + 1;
  return (arr[nbyte] & (1 << (8 - nbit))) >> (8 - nbit);
}

uint8_t rotate_host(uint8_t *arr, uint8_t arr_size)
{
  uint8_t i;

  uint8_t a1 = nbit_host(arr, 91) & nbit_host(arr, 92);
  uint8_t a2 = nbit_host(arr, 175) & nbit_host(arr, 176);
  uint8_t a3 = nbit_host(arr, 286) & nbit_host(arr, 287);

  uint8_t t1 = nbit_host(arr, 66) ^ nbit_host(arr, 93);
  uint8_t t2 = nbit_host(arr, 162) ^ nbit_host(arr, 177);
  uint8_t t3 = nbit_host(arr, 243) ^ nbit_host(arr, 288);

  uint8_t out = t1 ^ t2 ^ t3;

  uint8_t s1 = a1 ^ nbit_host(arr, 171) ^ t1;
  uint8_t s2 = a2 ^ nbit_host(arr, 264) ^ t2;
  uint8_t s3 = a3 ^ nbit_host(arr, 69) ^ t3;

  /* Begin rotate */

  for(i = arr_size - 1; i > 0; i--)
  {
    arr[i] = (arr[i - 1] << 7) | (arr[i] >> 1);
  }
  arr[0] = arr[0] >> 1;

  /* End rotate */

  change_bit_host(arr, 1, s3);
  change_bit_host(arr, 94, s1);
  change_bit_host(arr, 178, s2);

  return out;
}

void insert_bits_host(uint8_t *arr, uint16_t n, uint8_t *source, uint16_t ssize)
{
  uint16_t i;
  for(i = 0; i < ssize; i++)
  {
    change_bit_host(arr, n + i, nbit_host(source, i + 1));
  }
}

void initialize_state_host(uint8_t *arr)
{
  uint16_t i;
  for(i = 0; i < N_ROUND; i++)
  {
    rotate_host(arr, STATE_SIZE);
  }
}

uint8_t get_byte_from_gamma_host(uint8_t *arr)
{
  uint8_t buf = 0;
  uint8_t i = 0;
  while(i != 8)
  {
    uint8_t z = rotate_host(arr, STATE_SIZE);
    buf = buf | (z << i);
    i += 1;
  }
  return buf;
}




uint8_t encrypt_host(uint8_t *iv,uint8_t *k){
  uint8_t b[STATE_SIZE]={0x0};
  //for(int i = 0; i < STATE_SIZE; i++) b[i] = 0;
  insert_bits_host(b, 1, k, 80);
  insert_bits_host(b, 94, iv, 80);
  change_bit_host(b, 286, 1);
  change_bit_host(b, 287, 1);
  change_bit_host(b, 288, 1);
  //print_array(b, STATE_SIZE);
  initialize_state_host(b);
  /*
  for(int i=0;i<BIT_POSITION/8;i++)
    get_byte_from_gamma_host(b);
  
  return get_byte_from_gamma_host(b)&(1<<(BIT_POSITION%8));*/
   return (get_byte_from_gamma_host(b)&BIT_POSITION)>>BIT_POSITION_APP;
}

uint8_t encrypt_exploit_host(uint8_t *iv){
  uint8_t b[STATE_SIZE];
  for(int i = 0; i < STATE_SIZE; i++) b[i] = 0; 
  insert_bits_host(b, 1, h_key_2_guess, 80);
  insert_bits_host(b, 94, iv, 80);
  change_bit_host(b, 286, 1);
  change_bit_host(b, 287, 1);
  change_bit_host(b, 288, 1);
  //print_array(b, STATE_SIZE);
  initialize_state_host(b);
  /*
  for(int i=0;i<BIT_POSITION/8;i++)
    get_byte_from_gamma_host(b);
  return get_byte_from_gamma_host(b)&(1<<(BIT_POSITION%8));
  */
   return (get_byte_from_gamma_host(b)&BIT_POSITION)>>BIT_POSITION_APP;
}
/*
int main(int argc, char **argv)
{
  uint8_t key[10];
  uint8_t iv[10];
  uint8_t b[STATE_SIZE];
  uint8_t buffer;
  uint8_t encbuffer;
  FILE * pFile;
  FILE * outFile;
  uint8_t i;
  srand(time(NULL));
  for(i = 0; i < STATE_SIZE; i++) b[i] = 0;
  for(i = 0; i < 10; i++)
  {
    key[i] = 0;
    iv[i] = get_random_byte();
  }
  
  printf("Trivium (encryption) by sinhbad. January, 2017.\n");
  printf("Using: trivium.exe input.file output.file\n");
  if (argc != 3)
  {
    printf("Bad input data in argv\n");
    return 0;
  }
  pFile = fopen(argv[1] , "rb");
  outFile = fopen(argv[2], "wb");
  if (pFile==NULL) {fputs ("Input file error",stderr); exit (1);}
  if (outFile==NULL) {fputs ("Output file error",stderr); exit (1);}
  for (i = 0; i < 10; i++)
  {
    fwrite(&iv[i], 1, 1, outFile);
  }
  
  printf("Type key in hexadecimal format (80 bit):\n");
  for (i = 0; i < 10; i++)
  {
    key[i] = get_byte_from_console_input();
  }
  
  insert_bits(b, 1, key, 80);
  insert_bits(b, 94, iv, 80);
  change_bit(b, 286, 1);
  change_bit(b, 287, 1);
  change_bit(b, 288, 1);
  print_array(b, STATE_SIZE);
  initialize_state(b);
  while(fread(&buffer, 1, 1, pFile) != 0)
  {
    encbuffer = buffer ^ get_byte_from_gamma(b);
    fwrite(&encbuffer, 1, 1, outFile);
  }
  fclose(pFile);
  fclose(outFile);
  return 0;
}
*/


__device__ void change_bit(uint8_t *arr, uint16_t n, uint8_t value)
{
  uint8_t nbyte = (n - 1) / 8;
  uint8_t nbit = ((n - 1) % 8) + 1;

  arr[nbyte] = ((255 << (9 - nbit)) & arr[nbyte]) |
               (value << (8 - nbit)) |
               ((255 >> nbit) & arr[nbyte]);
}

__device__ uint8_t nbit(uint8_t *arr, uint16_t n)
{
  uint8_t nbyte = (n - 1) / 8;
  uint8_t nbit = ((n - 1) % 8) + 1;
  return (arr[nbyte] & (1 << (8 - nbit))) >> (8 - nbit);
}

__device__ uint8_t rotate(uint8_t *arr, uint8_t arr_size)
{
  uint8_t i;

  uint8_t a1 = nbit(arr, 91) & nbit(arr, 92);
  uint8_t a2 = nbit(arr, 175) & nbit(arr, 176);
  uint8_t a3 = nbit(arr, 286) & nbit(arr, 287);

  uint8_t t1 = nbit(arr, 66) ^ nbit(arr, 93);
  uint8_t t2 = nbit(arr, 162) ^ nbit(arr, 177);
  uint8_t t3 = nbit(arr, 243) ^ nbit(arr, 288);

  uint8_t out = t1 ^ t2 ^ t3;

  uint8_t s1 = a1 ^ nbit(arr, 171) ^ t1;
  uint8_t s2 = a2 ^ nbit(arr, 264) ^ t2;
  uint8_t s3 = a3 ^ nbit(arr, 69) ^ t3;

  /* Begin rotate */

  for(i = arr_size - 1; i > 0; i--)
  {
    arr[i] = (arr[i - 1] << 7) | (arr[i] >> 1);
  }
  arr[0] = arr[0] >> 1;

  /* End rotate */

  change_bit(arr, 1, s3);
  change_bit(arr, 94, s1);
  change_bit(arr, 178, s2);

  return out;
}

__device__ void insert_bits(uint8_t *arr, uint16_t n, uint8_t *source, uint16_t ssize)
{
  uint16_t i;
  for(i = 0; i < ssize; i++)
  {
    change_bit(arr, n + i, nbit(source, i + 1));
  }
}

__device__ void initialize_state(uint8_t *arr)
{
  uint16_t i;
  for(i = 0; i < N_ROUND; i++)
  {
    rotate(arr, STATE_SIZE);
  }
}

__device__ uint8_t get_byte_from_gamma(uint8_t *arr)
{
  uint8_t buf = 0;
  uint8_t i = 0;
  while(i != 8)
  {
    uint8_t z = rotate(arr, STATE_SIZE);
    buf = buf | (z << i);
    i += 1;
  }
  return buf;
}


__device__ uint8_t encrypt(uint8_t *iv,uint8_t *k){
  uint8_t b[STATE_SIZE];
  for(int i = 0; i < STATE_SIZE; i++) b[i] = 0;
  insert_bits(b, 1, k, 80);
  insert_bits(b, 94, iv, 80);
  change_bit(b, 286, 1);
  change_bit(b, 287, 1);
  change_bit(b, 288, 1);
  //print_array(b, STATE_SIZE);
  initialize_state(b);
  /*
  for(int i=0;i<BIT_POSITION/8;i++)
    get_byte_from_gamma(b);
  return get_byte_from_gamma(b)&(1<<(BIT_POSITION%8));
  */
   return (get_byte_from_gamma(b)&BIT_POSITION)>>BIT_POSITION_APP;
}

__device__ uint8_t encrypt_exploit(uint8_t *iv){
  uint8_t b[STATE_SIZE];
  for(int i = 0; i < STATE_SIZE; i++) b[i] = 0;
  insert_bits(b, 1, d_key_2_guess, 80);
  insert_bits(b, 94, iv, 80);
  change_bit(b, 286, 1);
  change_bit(b, 287, 1);
  change_bit(b, 288, 1);
  //print_array(b, STATE_SIZE);
  initialize_state(b);
  return (get_byte_from_gamma(b)&BIT_POSITION)>>BIT_POSITION_APP;
  /*
  for(int i=0;i<BIT_POSITION/8;i++)
    get_byte_from_gamma(b);
  return get_byte_from_gamma(b)&(1<<(BIT_POSITION%8));
  */
}
