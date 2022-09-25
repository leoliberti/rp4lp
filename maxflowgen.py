#!/usr/bin/env python3
## author: Leo Liberti
## purpose: generation of random network flow networks
## date: 220201

############################ imports #############################
import sys
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

######################### global params ##########################

myZero = 1e-8
myInf = 1e30

#showPlot = True
showPlot = False

#outPNG = True
outPNG = False

printConstraintMatrix = False

defaultArcGenProb = 0.05

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 2:
    exit('syntax is [./maxflowgen.py number_of_nodes [dummyarg [erdos_renyi_prob] ] ]')

n = int(sys.argv[1])

p = defaultArcGenProb
if len(sys.argv) >= 4:
    p = float(sys.argv[3])

if n < 2:
    exit('network flow networks need at least 2 nodes')

if p > 1 or p < 0:
    exit('Erdos-Renyi arc generation probability must be in [0,1]')

# output basename
outn = "maxflow-" + str(n) + "_" + str(p)

# generate random directed tree G on n-1 nodes, with source s=0
G = nx.random_tree(n=n-1, create_using=nx.DiGraph)
s = 0   # source node: no incoming nodes
t = n-1 # target node: no outgoing nodes

# enrich using Erdos-Renyi probability
m = n-1
for i in range(n-1):
    for j in range(1, n-1):
        if random.uniform(0,1) <= p:
            if i != j and (i,j) not in G.edges:
                G.add_edge(i,j)
                m += 1
                
# add arcs to target
degs = len(G[0])        # degree of source node - same degree with target
totarget = np.random.choice(n-2, degs, replace=False) # random (i,target)
for i in totarget:
    G.add_edge(i+1,t) # sampling indices from 0..n-2, add 1
    m += 1
        
# assign random capacities in [0,1]
for (i,j) in G.edges:
    G[i][j]['weight'] = random.uniform(0,1)

# edge to edge ID
edge2id = dict()
for id,(i,j) in enumerate(G.edges):
    edge2id[(i,j)] = id

# compute linear system Ax=b (first row of A is obj, last implied, b=0)
A = np.zeros((n,m))
for i in range(0,n):
    for j in G[i]:
        e = edge2id[(i,j)]
        A[i,e] = 1
        A[j,e] = -1

# output to AMPL .dat
dat = outn + ".dat"
with open(dat, 'w') as out:
    print("# AMPL .dat file encoding max flow instance", file=out)
    print("# n={0:d} m={1:d} p={2:f}".format(n,m,p), file=out)
    print("param n := {0:d};".format(n), file=out)
    print("param s := {0:d};".format(s+1), file=out)
    print("param t := {0:d};".format(t+1), file=out)
    print("param : A : u :=", file=out)
    for (i,j) in G.edges:
        print(" {0:d} {1:d} {2:f}".format(i+1,j+1,G[i][j]['weight']), file=out)
    print(";", file=out)
    if printConstraintMatrix:
        print("param m := {0:d}".format(m), file=out)
        print("param AA :=", file=out)
        for i in range(1,n-1):
            for j in range(m):
                if abs(A[i,j]) > myZero:
                    print(" {0:d} {1:d} {2:.1f}".format(i+1,j+1,A[i,j]),file=out)
        print(";", file=out)
    
# show and save digraph picture
png = outn + ".png"
if outPNG or showPlot:
    rlz = nx.spring_layout(G)
    labd = {i:i+1 for i in range(n)}
    lab = "p={0:.3f}".format(p)
    nx.draw(G, rlz, label=lab, node_color='cyan', with_labels=True, labels=labd, font_color='blue')
    plt.legend(labels=[lab])
    plt.savefig(png, format="PNG")
if showPlot:
    plt.show()
