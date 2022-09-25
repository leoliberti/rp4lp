#!/usr/bin/env python3
## author: Leo Liberti
## purpose: generation of random instances of basis pursuit for sparse coding
## date: 220201

############################ imports #############################
import sys
import os
import math
import numpy as np
import random

######################### global params ##########################

myZero = 1e-8
myInf = 1e30

defaultSparsity = 0.2

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 3:
    exit('syntax is [./basispursuitgen.py len=n enclen=m [sparsity=s] ]')

n = int(sys.argv[1])
m = int(sys.argv[2])
s = defaultSparsity
if len(sys.argv) >= 4:
    s = float(sys.argv[3])

if m < 1:
    exit('length of encoded vector needs to be >=1')

if n < 2:
    exit('length of original vector needs to be >=2')

if m >= n:
    exit('original vector must be longer than encoded vector (n>m)')

if s > 1 or s < 0:
    exit('sparsity must be in [0,1]')

# output basename
outn = "basispursuit-" + str(m) + "_" + str(n) + "_" + str(s)

# sample original s-sparse  vector
org = np.zeros(n)
for i in range(n):
    if random.uniform(0,1) <= s:
        org[i] = np.random.randint(-10,10) / 10.0

# encoding matrix
A = np.random.normal(size=(m,n))

# encoded vector
enc = np.dot(A,org)

# output to AMPL .dat
dat = outn + ".dat"
with open(dat, 'w') as out:
    print("# AMPL .dat file encoding basis pursuit instance", file=out)
    print("# m={0:d} n={1:d} s={2:.3f}".format(m,n,s), file=out)
    print("param m := {0:d};".format(m), file=out)
    print("param n := {0:d};".format(n), file=out)
    print("param s := {0:f};".format(s), file=out)
    print("param org :=", file=out)
    for i in range(n):
        if abs(org[i]) > myZero:
            print(" {0:d} {1:.1f}".format(i+1, org[i]), file=out)
    print(";", file=out)
    print("param enc :=", file=out)
    for i in range(m):
        print(" {0:d} {1:f}".format(i+1, enc[i]), file=out)
    print(";", file=out)                  
    print("param A :=", file=out)
    for i in range(m):
        for j in range(n):
            print(" {0:d} {1:d} {2:f}".format(i+1,j+1, A[i,j]), file=out)
    print(";", file=out)
