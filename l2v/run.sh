#!/usr/bin/env bash

for name in 'karate' 'cora' 'citeseer' 'pubmed'; do
    python3 src/main.py --scratch --input "data/${name}.edgelist" --output "embed/${name}.pkl" --line-graph "data/${name}_line.edgelist" --l2v-iter 10 --iter 5
done
