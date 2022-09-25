# basispursuit.mod

param m integer, default 1;
param n integer, default 2;
set M := 1..m;
set N := 1..n;
param s >= 0, <= 1, default 0.2; # sparsity of original encoding
param org{N} integer, default 0;
param enc{M} default 0.0;
param A{M,N} default 0;

var x{N};
var sv{N};

minimize ell1error: sum{j in N} sv[j];

subject to decoding{i in M}: sum{j in N} A[i,j]*x[j] = enc[i];

subject to slack1{j in N}:   x[j] <= sv[j];
subject to slack2{j in N}: -sv[j] <=  x[j];

