#!/bin/bash

## run rp4lp.py in batch over a set of instances
## jllEPS and runPerEps are set into the config part of rp4lp.py
## rp4lp.py runs in batch mode natively; this script automates very little
##   specifically, rename the .csv output file from rp4lp
##                 save log
##                 call ./run_mktables

## from README.txt:
## - run ./rp4lp.py on file set "dat/[structure]-[sizes].dat.gz" as args
##   redirecting output to [structure]-[timestamp].log
## - mv rp4lp.csv to [structure]-[timestamp]-avg.csv
## - ./run_mktables [structure]-[timestamp]-avg.csv
##   with additional command will copy latex tables and png graphs to
##    /Users/liberti/maths/dr2/papers/jll/computation/lp

TIMESTAMP=`date "+%y%m%d%H%M%S"`
DATADIR=~/work/data/jll_lp/jea
#DATADIR=~/work/data/jll_lp/tmp
#DATADIR=test
EXE=rp4lp.py
PAPERDIR=~/maths/dr2/papers/jll/computation/lp
TOPAPER=
#TOPAPER=nonempty # in order to copy tables and png graphs to $PAPERDIR
DATEXT=dat.gz
RPLPCSV=rp4lp.csv
RUNMKT=run-mktables.sh

# find application types
TMPF1=ttt1.tmp
for i in `ls -1 ${DATADIR}/*.${DATEXT}` ; do
    echo `basename $i | cut -d '-' -f 1`
done > $TMPF1
appTypes=()
for i in `cat $TMPF1 | sort -u` ; do
    appTypes+=( $i )
done
rm $TMPF1

# delete rp4lp.csv if it exists from a previous run
if [ -f $RPLPCSV ] ; then
    rm -f $RPLPCSV
fi

# parse command line
if [ "$1" == "" ]; then
    # no command line
    echo "error: need application type on cmd line"
    echo "  available application types in ${DATADIR}:"
    echo "    ${appTypes[@]}"
    echo "  remember to set jllEPS and runPerEps in config part of $EXE"
    exit 1

elif [[ ${appTypes[@]} =~ (^|[[:space:]])"$1"($|[[:space:]]) ]]; then    
    # run batch
    bn=${1}-${TIMESTAMP}
    ./$EXE ${DATADIR}/${1}-*.${DATEXT} | tee ${bn}.log
    test $RPLPCSV || exit 2  # rp4lp.csv was not written, something wrong
    mv $RPLPCSV ${bn}-avg.csv
    cp ${bn}-avg.csv out/
    mv ${bn}.log out/
    ./$RUNMKT ${bn}-avg.csv $TOPAPER
else
    # application type not found
    echo "error: application type $1 not found"
    echo "  available application types in ${DATADIR}:"
    echo "  ${appTypes[@]}"
    exit 1    
fi

[[ ${array[*]} =~ (^|[[:space:]])"$find"($|[[:space:]]) ]]
