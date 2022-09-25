# maxflowprj.mod - projected formulation

param rows integer, default 2;
param cols integer, default 1;
set R := 1..rows;
set C := 1..cols;
param s integer, in R, default 1;
param t integer, in R, default rows;
param u{C} >= 0;
param c{C} default 0.0; # objective vector

# projected params
param kmax integer, >1, default 1;
set K := 1..kmax;
param TA{K,C} default 0; # equality constraint matrix

# vars
var x{ij in C} >= 0, <= u[ij];

maximize sourceflow: sum{sj in C} c[sj]*x[sj];
subject to projflowcons{k in K}: sum{ij in C} TA[k,ij]*x[ij] = 0;

