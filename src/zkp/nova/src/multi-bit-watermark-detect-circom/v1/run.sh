#!/bin/bash

# 初始化变量
type_sampling=""
hash_type=""
# circom each_seq_len != rust partial_seq_len
# rust partial_seq_len == folding_num * circom each_seq_len
seq_len=""
##
merkle_tree_nlevels=""
each_position_info=""
position_total_num=""
folding_num=""

# 解析命令行参数
while getopts "t:h:s:m:e:p:f:" opt; do
  case "$opt" in
    t) type_sampling="$OPTARG" ;;   # type of sampling
    h) hash_type="$OPTARG"  ;;   # hash type
    s) seq_len="$OPTARG"      ;;   # The number of token per sub-circuit
    m) merkle_tree_nlevels="$OPTARG"      ;;   # The number of levels in the Merkle tree
    e) each_position_info="$OPTARG"      ;;   # The information length per position
    p) position_total_num="$OPTARG"      ;;   # Total number of positions
    f) folding_num="$OPTARG"     ;;   # Folding number
    *) echo "无效的选项: -$opt"; exit 1 ;;
  esac
done

# 确保所有必需的参数都已传入
if [ -z "$type_sampling" ] || [ -z "$hash_type" ] || [ -z "$seq_len" ] || [ -z "$merkle_tree_nlevels" ] || [ -z "$each_position_info" ] || [ -z "$position_total_num" ] || [ -z "$folding_num" ]; then
  echo "错误: 必须提供文件夹名、三个参数和值和文件名"
  echo "用法: $0 -t <type_sampling> -h <hash_type> -s <seq_len> -m <merkle_tree_nlevels> -e <each_position_info> -p <position_total_num> -f <folding_num>"
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
  sed -i "/each_position_detect([0-9]*, [0-9]*, [0-9]*)/s/\(each_position_detect(\)[0-9]\+, [0-9]\+, [0-9]\+/\1$seq_len, $each_position_info, $merkle_tree_nlevels/" "$circom_file"
  echo "文件 $circom_file 中的参数已修改为：$seq_len, $each_position_info, $merkle_tree_nlevels"
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
echo "正在执行 ${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/multi-bit-$hash_type-detect-v1 命令..."
# ${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/"$hash_type"-detect -s "$seq_len" -w "$window_size" -d "$depth" -f "$folding_num"

seq_len_used_in_rust=$((seq_len * folding_num))

output=$(${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/target/release/multi-bit-"$hash_type"-detect-v1 -t "$type_sampling" --partial-seq-len "$seq_len_used_in_rust" -m "$merkle_tree_nlevels" -e "$each_position_info" --position-total-num "$position_total_num" -f "$folding_num")

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

token_num=$((seq_len * folding_num * position_total_num))


output_file_dir="${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/src/multi-bit-watermark-detect-circom/v1/results_more/${token_num}/${hash_type}"
if [ ! -d "$output_file_dir" ]; then
    echo "目录 $output_file_dir 不存在，正在创建..."
    
    # 创建目录，-p选项允许创建多级不存在的目录
    mkdir -p "$output_file_dir"
    
    # 检查创建是否成功
    if [ $? -eq 0 ]; then
        echo "目录 $output_file_dir 创建成功"
    else
        echo "错误：创建目录 $output_file_dir 失败"
        exit 1
    fi
else
    echo "目录 $output_file_dir 已存在"
fi

output_file="${PVMark_NOVA_SCOTIA_ROOT:-external/Nova-Scotia}/src/multi-bit-watermark-detect-circom/v1/results_more_bit/${token_num}/${hash_type}/${hash_type}_s_${seq_len}_m_${merkle_tree_nlevels}_e_${each_position_info}_p_${position_total_num}_f_${folding_num}.txt"

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
