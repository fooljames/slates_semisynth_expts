#!/bin/bash

M=100
L=10

for metric in NDCG
do
    for temp in 0.0 0.5 1.0 1.5 2.0
    do
        for eval in tree
        do
            for approach in OnPolicy
            do
                python3 Parallel.py -m ${M} -l ${L} -v ${metric} -e ${eval} -t ${temp} -a ${approach} --start 0 --stop 10 &> eval.log.${metric}.${M}.${L}.${temp}.${eval}.${approach} &
            done
	    done
    done
done