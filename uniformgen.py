#!/usr/bin/env python3
## author: Leo Liberti
## purpose: generation of random uniform standard form LPs
## date: 220222

############################ imports #############################
import sys
import os
import math
import numpy as np
import random

######################### global params ##########################

myZero = 1e-8
myInf = 1e30

dens = 0.3

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 3:
    exit('syntax is [./uniformgen.py m n [density] ]')

m = int(sys.argv[1])
n = int(sys.argv[2])
density = dens
if len(sys.argv) >= 4:
    density = float(sys.argv[3])

if m < 1:
    exit('m must be >=1')

if n < 1:
    exit('n must be >= 1')

if density < 0 or density > 1:
    exit('density must be in [0,1]')
    
# output basename
outn = "uniform-" + str(m) + "_" + str(n) + "_" + str(density)

# random uniform constraint matrix
A = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        if random.uniform(0,1) <= density:
            A[i,j] = random.uniform(-1,1)

# RHS vector
b = np.random.uniform(0,1,m)

# cost vector
c = np.random.uniform(0,1,n)

# output to AMPL .dat
dat = outn + ".dat"
with open(dat, 'w') as out:
    print("# AMPL .dat file encoding uniform instance", file=out)
    print("# m={0:d} n={1:d} density={2:.2f}".format(m,n,density), file=out)
    print("param m := {0:d};".format(m), file=out)
    print("param n := {0:d};".format(n), file=out)
    print("param b :=", file=out)
    for i in range(m):
        print(" {0:d} {1:f}".format(i+1,b[i]), file=out)
    print(";", file=out)
    print("param c :=", file=out)
    for i in range(n):
        print(" {0:d} {1:f}".format(i+1,c[i]), file=out)
    print(";", file=out)
    print("param A :=", file=out)
    for i in range(m):
        for j in range(n):
            if abs(A[i,j]) > myZero:
                print(" {0:d} {1:d} {2:f}".format(i+1,j+1, A[i,j]), file=out)
    print(";", file=out)
    
