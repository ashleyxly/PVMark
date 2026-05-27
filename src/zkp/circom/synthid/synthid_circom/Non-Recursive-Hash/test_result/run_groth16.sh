#!/bin/bash

hash_type=""
seq_len=""
window_size=""
depth=""


while getopts "h:s:w:d:" opt; do
  case "$opt" in
    h) hash_type="$OPTARG"  ;;   # hash type
    s) seq_len="$OPTARG"      ;;   # The number of token per sub-circuit
    w) window_size="$OPTARG"      ;;   # The sliding window size
    d) depth="$OPTARG"      ;;   # The watermarking depth per sub-circuit
    # f) folding_num="$OPTARG"     ;;   # Folding number
    *) echo "无效的选项: -$opt"; exit 1 ;;
  esac
done


# 确保所有必需的参数都已传入
if [ -z "$hash_type" ] || [ -z "$seq_len" ] || [ -z "$window_size" ] || [ -z "$depth" ] ; then
  echo "错误: 必须提供文件夹名、三个参数和值和文件名"
  echo "用法: $0 -h <hash_type> -s <seq_len> -w <window_size> -d <depth>"
  exit 1
fi

export NODE_OPTIONS="--max-old-space-size=20480000"


result_file_path="${PVMark_ZKLLM_ROOT:-external/ZKLLMWatermark_Codes}/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/test_result/Groth16_h_${hash_type}_s_${seq_len}_w_${window_size}_d_${depth}.txt"

> "$result_file_path"

# 设置根文件夹
root_folder="${PVMark_ZKLLM_ROOT:-external/ZKLLMWatermark_Codes}/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/${hash_type}"

cd "$root_folder" || exit

public_json_path="${PVMark_ZKLLM_ROOT:-external/ZKLLMWatermark_Codes}/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/${hash_type}/detect_directly_public.json"

echo '{"output_count": 3000}' > $public_json_path

${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/no-recursive-inputs -s "$seq_len" -w "$window_size" -d "$depth"

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
# print_progress() {
#     local progress=$(( (current_file * 100) / total_files ))
#     local done=$(( (progress * 4) / 10 ))
#     local left=$(( 40 - done ))

#     printf "\rProcessing: [%.*s%*s] %d%% (%d/%d)" \
#         $done "########################################" \
#         $left "" \
#         $progress \
#         $current_file \
#         $total_files
# }


# time
start_total=$(date +%s.%3N)

# 输出数组中的所有文件路径
for file_full in "${circom_files[@]}"; do
    echo "Processing $file_full"

    filenametemp=$(basename "$file_full")
    filename=$(basename "$filenametemp" .circom)

    #使用 sed 修改文件中的参数值
    sed -i "/detect_directly([0-9]*, [0-9]*, [0-9]*)/s/\(detect_directly(\)[0-9]\+, [0-9]\+, [0-9]\+/\1$seq_len, $window_size, $depth/" "$file_full"
    echo "文件 $file_full 中的参数已修改为：$seq_len, $window_size, $depth"


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
    if [ $log2_ceil -gt 24 ]; then
        echo "log2_ceil: $log2_ceil is too large, skip this file" >> "$result_file_path" 2>&1
        continue
    fi

    cd "./${filename}_cpp" || exit
    make

    ./"${filename}" ../input.json "./${filename}_witness.wtns"

    # # 根据文件名包含的关键字进行不同处理
    # if [[ "$filename" == *sort* ]]; then
    #     ./"${filename}" ../../input_sort.json "./${filename}_witness.wtns"
    # elif [[ "$filename" == *fixed* ]]; then
    #     ./"${filename}" ../../input_fixed.json "./${filename}_witness.wtns"
    # else
    #     echo "Skipping $filename as it does not match any criteria"
    # fi
    cd ../

    # Setup
    echo "****Setup****"
    start=$(date +%s.%3N)
    snarkjs groth16 setup "./${filename}.r1cs" "${PVMark_PTAU_DIR:-external/snarkjs_setup_file}/powersOfTau28_hez_final_${log2_ceil}.ptau" "${filename}_circuit_0.zkey" -v
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
    # print_progress

done

end_total=$(date +%s.%3N)
runtime=$(echo "scale=3; $end_total - $start_total" | bc)
# echo "DONE ($runtime)s" > "total_time.txt" 2>&1
echo "Total time: $runtime" >> "$result_file_path" 2>&1
