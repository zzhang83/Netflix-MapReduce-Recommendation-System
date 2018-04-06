import pandas as pd
import numpy as np
from pprint import pprint
from mapreduce import mapreduce
from pymongo import *


# Sparse Matrix class
class SpMatrix:
    def __init__(self, m=1, n=1, ijv=[]):
        '''
        m: number rows
        n: number columns
        ijv: list of ((i, j), value) tuples
        '''
        self.m = m
        self.n = n
        self.ijv = ijv
    def np(self):
        '''Returns self as a NumPy Array'''
        r = np.zeros((self.m,self.n))
        for (ij,v) in self.ijv:
            r[ij[0],ij[1]]=v
        return r
    def __str__(self):
        return self.np().__str__()


def SpEye(n):
    I = SpMatrix(n,n,[((i,i),1) for i in range(n)])
    return I


def combinations(iterable, r):
    # http://docs.python.org/2/library/itertools.html#itertools.combinations
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(list(range(r))):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)



def mapper1(sijv):
    '''
    Given a record tuple (( a or b,i,j,value))
    Returns [(j or i,(a,i,v))]  key = j if from a matrix, key = i if from b matrix
    '''
    result = []
    if sijv[0]=='a':
        result.append((sijv[2],[(sijv[0],sijv[1],sijv[3])])) # column as key if it is a
    else:
        result.append((sijv[1],[(sijv[0],sijv[2],sijv[3])]))  # row as key if it is b
    print('mapper1: %s-> %s'%(sijv,result))
    return result


def reducer(a, b):
    return a + b



def mapper2(record):
    '''
    Given a tuple for example(0, [('a', 0, 1), ('a', 1, 3), ('b', 0, 1), ('b', 1, 0), ('b', 2, 3)])
    return [((0, 0), 1 ), ((1,0), 3 ), ((1,2), 9 )....]
    key: (i,j)
    value: value from a * value from b
    '''
    comb = list(combinations(record[1],2))     # get pair combinations of (i,j)
    result = []
    for i in comb:
        if i[0][0] != i[1][0]:   # only keep a,b pair
            result.append(((i[0][1],i[1][1]),i[0][2]*i[1][2]))
        else:
            pass
    print('mapper2: %s-> %s'%(record,result))
    return result


def SpMultiply(A, B):
    #     assert A.n == B.n
    sijv = [('a', ijv[0][0], ijv[0][1], ijv[1]) for ijv in A.ijv]
    sijv += [('b', ijv[0][0], ijv[0][1], ijv[1]) for ijv in B.ijv]
    map_reducer = mapreduce()
    print('MapReduce input:')
    pprint(sijv)

    matrix_multi = map_reducer.parallelize(sijv, 128) \
        .flatMap(mapper1) \
        .reduceByKey(reducer) \
        .flatMap(mapper2) \
        .reduceByKey(reducer)
    print('MapReduce Output:')
    final_matrix = []
    for item in matrix_multi.collect():
        print(item)
        final_matrix.append(item)
    return SpMatrix(A.m, B.n, final_matrix)


def test_SpMultiply():
    # (1x2)*(2x2)
    A = SpMatrix(1, 2, [((0, 0), 1), ])
    B = SpMatrix(2, 2, [((0, 0), 1), ((1, 0), 2), ((0, 1), 3), ((1, 1), 4)])
    r = SpMultiply(A, B)
    assert np.all(r.np()==np.matmul(A.np(),B.np()))

    # I*(2x2)
    I = SpEye(2)
    r = SpMultiply(I, B)
    assert np.all(r.np() == B.np())

    # (2x2)*(2x3)
    C = SpMatrix(2, 3, [((0, 0), 1), ((0, 1), 3), ((1, 2), 7),
                        ((1, 0), 2), ((1, 1), 4)])
    r = SpMultiply(B, C)
    assert(np.all(r.np()==np.matmul(B.np(),C.np())))

# pytest in terminal