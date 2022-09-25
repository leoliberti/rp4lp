#!/bin/bash

OUT=out
CURRD=`pwd`
#PAPERD=~/maths/dr2/papers/jll/computation/lp
PAPERD=~/maths/dr1/papers/rp/lp_apm

if [ "$1" == "" ]; then
    echo "syntax is $0 out/file-avg.csv"
    echo "  this script will run maketables and update out/file-avg.zip"
    echo "  with an additional command argument it will also"
    echo "    copy relevant files to $PAPERD"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "cannot open file $1"
    exit 2
fi

./maketables.py $1
BN=`basename $1 .csv`
FP=`echo $BN | cut -d '-' -f 1`
PT=`echo $FP | cut -d '_' -f 1`
if [ "$PT" != "$FP" ]; then
    mmv "${PT}-*" "${FP}-#1"
fi
cd ${OUT}/
if [ -f ${BN}.zip ] ; then
    unzip ${BN}.zip
else
    mkdir ${BN}/
fi
mv ${CURRD}/${FP}-* ${BN}/
zip -rp ${BN}.zip ${BN}/

if [ "$2" != "" ]; then
    cd $BN/
    mv *.png ${PAPERD}/fig/
    mv *.tex ${PAPERD}/
    cd ..
else
    echo "WARNING: not updating files in $PAPERD"
    echo "  re-run with second cmd line arg to update"
fi

rm -rf ${BN}/
cd $CURRD/

