#!/bin/bash

# 初始化变量
hash_type=""
seq_len=""
window_size=""
depth=""
folding_num=""

# 解析命令行参数
while getopts "h:s:w:d:f:" opt; do
  case "$opt" in
    h) hash_type="$OPTARG"  ;;   # hash type
    s) seq_len="$OPTARG"      ;;   # The number of token per sub-circuit
    w) window_size="$OPTARG"      ;;   # The sliding window size
    d) depth="$OPTARG"      ;;   # The watermarking depth per sub-circuit
    f) folding_num="$OPTARG"     ;;   # Folding number
    *) echo "无效的选项: -$opt"; exit 1 ;;
  esac
done

# 确保所有必需的参数都已传入
if [ -z "$hash_type" ] || [ -z "$seq_len" ] || [ -z "$window_size" ] || [ -z "$depth" ] || [ -z "$folding_num" ]; then
  echo "错误: 必须提供文件夹名、三个参数和值和文件名"
  echo "用法: $0 -h <hash_type> -s <seq_len> -w <window_size> -d <depth> -f <folding_num>"
  exit 1
fi

# 检查目标文件夹是否存在
if [ ! -d "$hash_type" ]; then
  echo "文件夹 $hash_type 不存在"
  exit 1
fi

# 进入目标文件夹
cd "$hash_type" || exit

# 查找并修改 detect_recursive.circom 文件中的特定行
circom_file="detect_recursive.circom"

if grep -q "component main" "$circom_file"; then
  # 使用 sed 修改文件中的参数值
  sed -i "/detect_recursive([0-9]*, [0-9]*, [0-9]*)/s/\(detect_recursive(\)[0-9]\+, [0-9]\+, [0-9]\+/\1$seq_len, $window_size, $depth/" "$circom_file"
  echo "文件 $circom_file 中的参数已修改为：$seq_len, $window_size, $depth"
else
  echo "没有找到 detect_recursive 函数"
  exit 1
fi

# 执行 circom 命令
echo "正在执行 circom 命令..."
circom "$circom_file" --r1cs --sym --c

# 进入 detect_recursive_cpp 文件夹并执行 make 命令
cpp_folder="detect_recursive_cpp"
if [ -d "$cpp_folder" ]; then
  cd "$cpp_folder" || exit
  echo "正在执行 make 命令..."
  make
else
  echo "文件夹 $cpp_folder 不存在"
  exit 1
fi

# 执行 Nova-Scotia 相关命令
echo "正在执行 ${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/$hash_type-detect 命令..."
# ${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/"$hash_type"-detect -s "$seq_len" -w "$window_size" -d "$depth" -f "$folding_num"

output=$(${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/"$hash_type"-detect -s "$seq_len" -w "$window_size" -d "$depth" -f "$folding_num")

# 输出命令执行结果（可选，调试用）
echo "$output"


# 提取 Number of constraints per step 和 Number of variables per step
primary_constraints=$(echo "$output" | grep "Number of constraints per step (primary circuit)" | awk -F': ' '{print $2}' | awk '{print $1}')
secondary_constraints=$(echo "$output" | grep "Number of constraints per step (secondary circuit)" | awk -F': ' '{print $2}' | awk '{print $1}')
primary_variables=$(echo "$output" | grep "Number of variables per step (primary circuit)" | awk -F': ' '{print $2}' | awk '{print $1}')
secondary_variables=$(echo "$output" | grep "Number of variables per step (secondary circuit)" | awk -F': ' '{print $2}' | awk '{print $1}')


# 提取 Total Setup time 和 consist 的时间
setup_time=$(echo "$output" | grep "Total setup time" | awk -F': ' '{print $2}' | awk '{print $1}' | sed 's/s//')
setup_consist_times=$(echo "$output" | grep "Total setup time" | awk -F'consists of ' '{print $2}' | sed 's/s//g')

# 提取 Total Prove time 和 consist 的时间
prove_time=$(echo "$output" | grep "Total prove time" | awk -F': ' '{print $2}' | awk '{print $1}' | sed 's/s//')
prove_consist_times=$(echo "$output" | grep "Total prove time" | awk -F'consists of ' '{print $2}' | sed 's/s//g')

# 提取 Total Verify time 和 consist 的时间
verify_time=$(echo "$output" | grep "Total verify time" | awk -F': ' '{print $2}' | awk '{print $1}' | sed 's/s//')
verify_consist_times=$(echo "$output" | grep "Total verify time" | awk -F'consists of ' '{print $2}' | sed 's/ms//g' | sed 's/s//g')

token_num=$((seq_len * folding_num))

# 创建token_num文件夹如果不存在
token_num_dir="${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/src/synthid-detect-circom/hash-rand/results_more_2/${token_num}"
if [ ! -d "$token_num_dir" ]; then
  mkdir -p "$token_num_dir"
  echo "已创建文件夹: $token_num_dir"
fi

# 创建hash_type子文件夹如果不存在
hash_type_dir="${token_num_dir}/${hash_type}"
if [ ! -d "$hash_type_dir" ]; then
  mkdir -p "$hash_type_dir"
  echo "已创建文件夹: $hash_type_dir"
fi

output_file="${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/src/synthid-detect-circom/hash-rand/results_more_2/${token_num}/${hash_type}/${hash_type}_s_${seq_len}_w_${window_size}_d_${depth}_f_${folding_num}.txt"

# 创建输出文件并写入结果
echo "Total Setup time: $setup_time seconds" > "$output_file"
echo "Consists of: $setup_consist_times" >> "$output_file"
echo "Total Prove time: $prove_time seconds" >> "$output_file"
echo "Consists of: $prove_consist_times" >> "$output_file"
echo "Total Verify time: $verify_time seconds" >> "$output_file"
echo "Consists of: $verify_consist_times (ms)" >> "$output_file"
echo "****************************************" >> "$output_file"
# 提取的额外内容
echo "Number of constraints per step (primary circuit): $primary_constraints" >> "$output_file"
echo "Number of constraints per step (secondary circuit): $secondary_constraints" >> "$output_file"
echo "Number of variables per step (primary circuit): $primary_variables" >> "$output_file"
echo "Number of variables per step (secondary circuit): $secondary_variables" >> "$output_file"


# 输出到终端
echo "结果已保存到 $output_file"


echo "脚本执行完成"
