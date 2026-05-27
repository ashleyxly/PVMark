#!/bin/bash

bash run_groth16.sh -h mimc -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 mimc token 100 done!"

bash run_groth16.sh -h poseidon -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 poseidon token 100 done!"

bash run_groth16.sh -h poseidon2 -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 poseidon2 token 100 done!"

bash run_plonk.sh -h mimc -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk mimc token 100 done!"

bash run_plonk.sh -h poseidon -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon token 100 done!"

bash run_plonk.sh -h poseidon2 -s 100 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon2 token 100 done!"

###############

bash run_groth16.sh -h mimc -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 mimc token 50 done!"

bash run_groth16.sh -h poseidon -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 poseidon token 50 done!"

bash run_groth16.sh -h poseidon2 -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run groth16 poseidon2 token 50 done!"

bash run_plonk.sh -h mimc -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk mimc token 50 done!"

bash run_plonk.sh -h poseidon -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon token 50 done!"

bash run_plonk.sh -h poseidon2 -s 50 -w 4 -d 30 || { echo "运行失败"; exit 1; }

echo "run plonk poseidon2 token 50 done!"