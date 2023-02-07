#! /bin/bash
# max=10
# for (( i=2; i <= $max; ++i ))
# do
#     echo "$i"
# done
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

rm *.log
for ((i=1; i<=8; ++i)); do 
    cd ${SCRIPT_DIR}
    make clean
    ILPconfig=${i}
    echo "ILP = ${ILPconfig}"
    make -k ILP=${ILPconfig}
    cd ${SCRIPT_DIR}/bin/
    for f in ./*; do
        echo "running $f microbenchmark"
        $f >> ${SCRIPT_DIR}/V100-ILP"${ILPconfig}".log
        echo "/////////////////////////////////"
    done
done
