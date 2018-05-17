#!/bin/bash

M=100
L=10

for logger in tree
do
    for metric in ERR NDCG
    do
        for temp in 1.0
        do
            for eval in lasso
            do
                for approach in CME_A
                do
                    python3 Parallel.py -m ${M} -l ${L} -v ${metric} -f ${logger} -e ${eval} -t ${temp} -a ${approach} -s 100000 -u 6 --start 0 --stop 10 &> eval.log.${metric}.${M}.${L}.${logger}-${temp}.${eval}.${approach} &
                done
            done
        done
    done
done
