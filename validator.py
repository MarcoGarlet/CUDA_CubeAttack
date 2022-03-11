import os
import subprocess
from functools import reduce
import random


K_dim = 10 # nbyte of private key
N_keys = 100
'''

PART 1:
    read cube tests and superpolies files, build a sort of dictionary with cubes and its superpoly filtering valid cubes.

PART 2:
    call online phase and on each possible assignment of private key bits addressed by the superpoly validate the cube

'''

def get_cubes(fname = './final_attack/offline/cubes_test.txt'):
    cubes = [l.strip() for l in open(fname,'r').readlines()]
    return cubes

def get_superpolies(fname = './final_attack/offline/superpolies.txt'):
    superpolies = [l.strip() for l in open(fname,'r').readlines()]
    return superpolies


        

if __name__ ==  '__main__':
    compose_cs = zip(get_cubes(),get_superpolies())
    compose_cs_dict = {}
    for el in compose_cs:
        if el[0] not in compose_cs_dict.keys() and el[1]!='X': #and int(eval('len(['+el[1]+'])') <=8):
            compose_cs_dict[el[0]]=el[1]
    print(compose_cs_dict)
    print('#'*10)
    
    for k in compose_cs_dict.keys():
        #print('{} => {}'.format(k, compose_cs_dict[k]))
        #keys = get_key_set(eval('['+compose_cs_dict[k]+']'))
        keys = [[random.randint(0,(2<<7)-1) for i in range(K_dim)] for i1 in range(N_keys)]
        #print(keys)
        
        constant = 1 if '-1' in compose_cs_dict[k] else 0
        
        #print("#####")
        with open('./final_attack/offline/cubes_test_val.txt','w') as f: 
            f.write(k+'\n')
        pass_check = True
        #print('key set = {}'.format(keys))

        compose_el = eval('['+compose_cs_dict[k]+']')
        compose_el = [int(x) for x in compose_el if x!=-1]
        #print(compose_el)

        for i,ks in enumerate(keys):
            # for each different key in keyset write the file keypass
            # check the bit or xored key bit if they have the exact values declare before
            #print("Check private key {}".format(i))
           
            


            
            with open('./final_attack/offline/key2guess.txt','w') as f: 
                f.write(str(ks)[1:-1]+'\n')

            out = subprocess.check_output('./final_attack/validate/only_gpu/att1').decode().split('\n')
            out = int([x for x in out if 'gpu sum' in x][0][-1])

            bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1]) for c in ks]) # this depends on the cipher, OK for Trivium
            #bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1])[::-1] for c in ks]) # this depends on the cipher, OK for MORUS
            #bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1]) for c in ks][::-1]) 
            #bin_assignment = reduce(lambda x,y:x+y,[('0'*(8-len(bin(c).split('b')[1]))+bin(c).split('b')[1])[::-1] for c in ks][::-1]) # this depends on the cipher, OK for Grain

            bin_assignment_v = [constant]+[int(bin_assignment[i]) for i in compose_el]
            #print('bin assignment = {}'.format(bin_assignment))

            check_out = reduce(lambda x,y:x^y,bin_assignment_v)
            #print(check_out)

            if(check_out != out):
                print('[NO] => compose_el = {} bin_assignment = {} superpolies {} = {} instead of {}, invalid, constant = {}'.format(compose_el,bin_assignment_v,compose_el,check_out,out,constant))
                pass_check = False
                #exit()
                break
        
        if pass_check:
            print('[YES] => cube {} valid guessed key bits {}'.format(k,compose_cs_dict[k]))
            print("#####")
        
        
    # compose dict is your dict



