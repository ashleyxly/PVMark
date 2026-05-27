#!/bin/bash

# bash run_groth16.sh -h mimc -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

# echo "run groth16 mimc token 200 done!"

# bash run_groth16.sh -h poseidon -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

# echo "run groth16 poseidon token 200 done!"

# bash run_groth16.sh -h poseidon2 -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

# echo "run groth16 poseidon2 token 200 done!"

bash run_plonk.sh -h mimc -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk mimc token 200 done!"

bash run_plonk.sh -h poseidon -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon token 200 done!"

bash run_plonk.sh -h poseidon2 -s 200 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon2 token 200 done!"