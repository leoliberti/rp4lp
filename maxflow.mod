# maxflow.mod

param n integer, >2, default 2;
param m integer, >1, default 1;
set N := 1..n;
set A within {N,N};
param s integer, default 1;
param t integer, default n;
param u{A} >= 0;
param AA{N, 1..m} default 0; # constraint matrix, not used here
var x{(i,j) in A} >= 0, <= u[i,j];

maximize sourceflow:
  sum{j in N : (s,j) in A} x[s,j];

subject to flowcons{i in N diff {s,t}}:
  sum{j in N : (i,j) in A} x[i,j] = sum{j in N : (j,i) in A} x[j,i];

