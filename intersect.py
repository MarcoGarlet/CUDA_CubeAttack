

def read_file(fname = 'pos_bit'):
    return open(fname,'r').readlines()


if __name__ == '__main__':
    l = read_file()
    l = [set([int(el) for el in x.strip().split()]) for x in l]

    res = l[0]

    for el in l[1:]:
        res = res.intersection(el)
        print(res)

    print(res)




