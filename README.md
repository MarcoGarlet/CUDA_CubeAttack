# CUDA_CubeAttack



## Overview

**CUDA_CubeAttack** aims to offer a flexible implementation of cube attack exploiting CUDA framework.

## Break one of the ciphers in ./final_attack/ciphers_lib

Choose the proper bit/byte ordering in *cipher.cu* and *validator.py* file, 
setting the proper public and private key size.

Copy cipher's code in **cipher.cu** until the bottom delimiter.

Then just uncomment one of following lines in *IV_gen* and *IV_gen_host* functions:

```C
...
iv[(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8))); //Trivium
//iv[(IV_dim-1)-(cube_index[i]/8)] += to_MSB(c_i * (1<<(cube_index[i]%8))); // Grain
//iv[(cube_index[i]/8)] += (c_i * (1<<((cube_index[i]%8)))); // Morus
...
```

choose the same order in *key_gen_superpoly_reconstruction*:

```C
...
//k_start[(tid/8)] = (1<<(pos%8)); // Morus
k_start[(tid/8)] = to_MSB(1<<(pos%8)); // Trivium
//k_start[(K_dim-1) - (tid/8)] = to_MSB(1<<7-(pos%8)); //Grain 
...
```

and finally edit **validator.py**:


```Python3
...
bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1]) for c in ks]) # this depends on the cipher, OK for Trivium
#bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1])[::-1] for c in ks]) # this depends on the cipher, OK for MORUS
#bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1])[::-1] for c in ks][::-1]) # this depends on the cipher, OK for Grain
...
```

Remember to set the proper *K_dim*.


## Usage 

Once set the proper *N_ROUND* in **cipehr.cu**, after choosing the proper output bit position through *BIT_POSITION_APP* and *BIT_POSITION*, you can call **launch_attack.sh** as follow:



<p align='left'>
<img src='pics/launch_att.gif'>
</p>


Then two files are produced:
* **cubes_test.txt**
* **superpolies.txt**

Where the line position links each *maxterm* with the corresponding *superpoly*.


Check results using *validator.py*:

<p align='left'>
<img src='pics/validate.gif'>
</p>

## Attack a different cipher


## Results 

Results against Trivium, Morus-640-128 and Grain-128AEAD are reported [here](/docs/MasterThesis.pdf).







