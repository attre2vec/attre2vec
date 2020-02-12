# pen-line2vec
Information network edge representation learning using edge-to-vertex dual graphs (a.k.a line graph). In addition to that, an optimisation problem is solved efficiently to generate the edge embeddings.

**Sample command to run line2vec**

python main.py --input ../data/karate/karate.edgelist --output ../embed/karate/karate_line.embed --dimensions=8 --line-graph ../data/karate/karate_line.edgelist --l2v-iter=1 --iter=1 --alpha=0.1 --beta=0.1 --eta=0.01 --scratch
