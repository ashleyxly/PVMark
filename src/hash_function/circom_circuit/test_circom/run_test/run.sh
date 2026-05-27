#!/bin/bash


result_file_path="res_new_1_token.txt"
# 清空或创建输出文件
> "$result_file_path"

# 设置根文件夹
root_folder="/mnt/disk2/username/ZKLLMWatermark/hash_function/circom_circuit/test_circom"


# 初始化数组来存储文件路径
circom_files=()

# 查找所有 .circom 文件并将文件路径存储到数组中
while IFS= read -r -d '' file; do
    circom_files+=("$file")
done < <(find "$root_folder" -type f -name '*.circom' -print0)


# 获取数组的总长度
total_files=${#circom_files[@]}
current_file=0

# 打印进度条函数
print_progress() {
    local progress=$(( (current_file * 100) / total_files ))
    local done=$(( (progress * 4) / 10 ))
    local left=$(( 40 - done ))

    printf "\rProcessing: [%.*s%*s] %d%% (%d/%d)" \
        $done "########################################" \
        $left "" \
        $progress \
        $current_file \
        $total_files
}


# time
start_total=$(date +%s.%3N)

# 输出数组中的所有文件路径
for file_full in "${circom_files[@]}"; do
    echo "Processing $file_full"

    filenametemp=$(basename "$file_full")
    filename=$(basename "$filenametemp" .circom)

    echo "****Circom Compile****"
    start=$(date +%s.%3N)
    output=$(circom $file_full --r1cs --sym --c)
    non_linear_constraints=$(echo "$output" | grep -oP 'non-linear constraints:\s*\K\d+')
    wires=$(echo "$output" | grep -oP 'wires:\s*\K\d+')
    max_value=$(( non_linear_constraints > wires ? non_linear_constraints : wires ))
    log2_ceil=$(python3 -c "import math; print(math.ceil(math.log2($max_value)))")
    end=$(date +%s.%3N)
    runtime=$(echo "scale=3; $end - $start" | bc)
    # echo "DONE ($runtime)s" > "${filename}_compile_time.txt" 2>&1
    echo "Compile time for $filename: $runtime" >> "$result_file_path" 2>&1
    echo "non_linear_constraints: $non_linear_constraints" >> "$result_file_path" 2>&1
    echo "wires: $wires" >> "$result_file_path" 2>&1
    echo "max_value: $max_value" >> "$result_file_path" 2>&1
    echo "log2_ceil: $log2_ceil" >> "$result_file_path" 2>&1


    cd "./${filename}_cpp" || exit
    make
    # 根据文件名包含的关键字进行不同处理
    if [[ "$filename" == *sort* ]]; then
        ./"${filename}" ../../input_sort.json "./${filename}_witness.wtns"
    elif [[ "$filename" == *fixed* ]]; then
        ./"${filename}" ../../input_fixed.json "./${filename}_witness.wtns"
    else
        echo "Skipping $filename as it does not match any criteria"
    fi
    cd ../

    # Setup
    echo "****Setup****"
    start=$(date +%s.%3N)
    snarkjs groth16 setup "./${filename}.r1cs" "/mnt/disk2/username/circom_circuit/setup_file/powersOfTau28_hez_final_${log2_ceil}.ptau" "${filename}_circuit_0.zkey" -v
    echo "1234" | snarkjs zkey contribute "${filename}_circuit_0.zkey" "${filename}_circuit_1.zkey" --name="1st Contributor Name" -v
    snarkjs zkey export verificationkey "${filename}_circuit_1.zkey" "${filename}_verification_key.json"
    end=$(date +%s.%3N)
    runtime=$(echo "scale=3; $end - $start" | bc)
    # echo "DONE ($runtime)s" > "${filename}_setup_time.txt" 2>&1
    echo "Setup time for $filename: $runtime" >> "$result_file_path" 2>&1

    # Generating a Proof
    echo "****Generate Proof****"
    start=$(date +%s.%3N)
    snarkjs groth16 prove "${filename}_circuit_1.zkey" "./${filename}_cpp/${filename}_witness.wtns" "${filename}_proof.json" "${filename}_public.json"
    end=$(date +%s.%3N)
    runtime=$(echo "scale=3; $end - $start" | bc)
    # echo "DONE ($runtime)s" > "${filename}_prove_time.txt" 2>&1
    echo "Prove time for $filename: $runtime" >> "$result_file_path" 2>&1

    # Verifying a Proof
    echo "****Verify Proof****"
    start=$(date +%s.%3N)
    snarkjs groth16 verify "${filename}_verification_key.json" "${filename}_public.json" "${filename}_proof.json" 
    end=$(date +%s.%3N)
    runtime=$(echo "scale=3; $end - $start" | bc)
    # echo "DONE ($runtime)s" > "${filename}_verication_time.txt" 2>&1
    echo "Verify time for $filename: $runtime" >> "$result_file_path" 2>&1

    # 更新当前处理的文件计数
    current_file=$((current_file + 1))
    print_progress

done

end_total=$(date +%s.%3N)
runtime=$(echo "scale=3; $end_total - $start_total" | bc)
# echo "DONE ($runtime)s" > "total_time.txt" 2>&1
echo "Total time: $runtime" >> "$result_file_path" 2>&1