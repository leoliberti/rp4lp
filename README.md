# rp4lp
research code used to write an academic paper

The RP4LP code (rp4lp.py) was written for two papers:
- numerical evaluation in ../../../papers/jll/computation/lp
- testing a strong solution retrieval with alternating projection

HOWTO:
- set jllEPS and runPerEps to define experiments
- run ./rp4lp.py on file set "dat/[structure]-[sizes].dat.gz" as args
  redirecting output to [structure]-[timestamp].log
- mv rp4lp.csv to [structure]-[timestamp]-avg.csv
- ./run_mktables [structure]-[timestamp]-avg.csv
  with additional command will copy latex tables and png graphs to
    /Users/liberti/maths/dr2/papers/jll/computation/lp
