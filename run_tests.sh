#!/bin/bash

# source .venv/bin/activate
limit=5000
mkdir -p results
echo "# python client.py --export-tree-imgs --limit $limit" >> results/Any.txt
date >> results/Any.txt
python client.py --limit $limit >> results/Any.txt

for cat in "Analysis" "Exploits" "Normal" "DoS" "Reconnaissance" "Fuzzers" "Backdoor" "Generic" "Shellcode" "Worms" ;
  do
    echo "# python client.py --limit $limit --attack_cat $cat" >> "results/$cat.txt"
    date >> "results/$cat.txt"
    python client.py --limit $limit --attack_cat $cat >> "results/$cat.txt"
  done
