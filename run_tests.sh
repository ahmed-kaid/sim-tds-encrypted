#!/bin/bash

# source .venv/bin/activate
limit=5
mkdir -p results
echo "# python client.py --limit $limit" >> results/All.txt
date >> results/All.txt
python client.py --limit $limit >> results/All.txt

for cat in "Analysis" "Exploits" "Normal" "DoS" "Reconnaissance" "Fuzzers" "Backdoor" "Generic" "Shellcode" "Worms" ;
  do
    echo "# python client.py --limit $limit --attack_cat $cat" >> "results/$cat.txt"
    date >> "results/$cat.txt"
    python client.py --limit $limit --attack_cat $cat >> "results/$cat.txt"
  done
