#!/bin/bash

# 参数列表
parameters=(
  # # 60 (10)
  # "multi-bit-watermark mimc 1 15 16 6 10"
  # "multi-bit-watermark mimc 2 15 16 6 5"
  # "multi-bit-watermark mimc 5 15 16 6 2"
  # "multi-bit-watermark mimc 10 15 16 6 1"

  # "multi-bit-watermark poseidon 1 15 16 6 10"
  # "multi-bit-watermark poseidon 2 15 16 6 5"
  # "multi-bit-watermark poseidon 5 15 16 6 2"
  # "multi-bit-watermark poseidon 10 15 16 6 1"

  # "multi-bit-watermark poseidon2 1 15 16 6 10"
  # "multi-bit-watermark poseidon2 2 15 16 6 5"
  # "multi-bit-watermark poseidon2 5 15 16 6 2"
  # "multi-bit-watermark poseidon2 10 15 16 6 1"

  # # 120 (20)
  # "multi-bit-watermark mimc 1 15 16 6 20"
  # "multi-bit-watermark mimc 2 15 16 6 10"
  # "multi-bit-watermark mimc 4 15 16 6 5"
  # "multi-bit-watermark mimc 5 15 16 6 4"
  # "multi-bit-watermark mimc 10 15 16 6 2"
  # "multi-bit-watermark mimc 20 15 16 6 1"

  # "multi-bit-watermark poseidon 1 15 16 6 20"
  # "multi-bit-watermark poseidon 2 15 16 6 10"
  # "multi-bit-watermark poseidon 4 15 16 6 5"
  # "multi-bit-watermark poseidon 5 15 16 6 4"
  # "multi-bit-watermark poseidon 10 15 16 6 2"
  # "multi-bit-watermark poseidon 20 15 16 6 1"

  # "multi-bit-watermark poseidon2 1 15 16 6 20"
  # "multi-bit-watermark poseidon2 2 15 16 6 10"
  # "multi-bit-watermark poseidon2 4 15 16 6 5"
  # "multi-bit-watermark poseidon2 5 15 16 6 4"
  # "multi-bit-watermark poseidon2 10 15 16 6 2"
  # "multi-bit-watermark poseidon2 20 15 16 6 1"

  # # 240 (40)
  # "multi-bit-watermark mimc 2 15 16 6 20"
  # "multi-bit-watermark mimc 4 15 16 6 10"
  # "multi-bit-watermark mimc 5 15 16 6 8"
  # "multi-bit-watermark mimc 10 15 16 6 4"
  # "multi-bit-watermark mimc 20 15 16 6 2"

  # "multi-bit-watermark poseidon 2 15 16 6 20"
  # "multi-bit-watermark poseidon 4 15 16 6 10"
  # "multi-bit-watermark poseidon 5 15 16 6 8"
  # "multi-bit-watermark poseidon 10 15 16 6 4"
  # "multi-bit-watermark poseidon 20 15 16 6 2"

  # "multi-bit-watermark poseidon2 2 15 16 6 20"
  # "multi-bit-watermark poseidon2 4 15 16 6 10"
  # "multi-bit-watermark poseidon2 5 15 16 6 8"
  # "multi-bit-watermark poseidon2 10 15 16 6 4"
  # "multi-bit-watermark poseidon2 20 15 16 6 2"

  # # 480 (80)
  # "multi-bit-watermark mimc 2 15 16 6 40"
  # "multi-bit-watermark mimc 4 15 16 6 20"
  # "multi-bit-watermark mimc 5 15 16 6 16"
  # "multi-bit-watermark mimc 8 15 16 6 10"
  # "multi-bit-watermark mimc 10 15 16 6 8"
  # "multi-bit-watermark mimc 16 15 16 6 5"
  # "multi-bit-watermark mimc 20 15 16 6 4"
  # "multi-bit-watermark mimc 40 15 16 6 2"

  # "multi-bit-watermark poseidon 2 15 16 6 40"
  # "multi-bit-watermark poseidon 4 15 16 6 20"
  # "multi-bit-watermark poseidon 5 15 16 6 16"
  # "multi-bit-watermark poseidon 8 15 16 6 10"
  # "multi-bit-watermark poseidon 10 15 16 6 8"
  # "multi-bit-watermark poseidon 16 15 16 6 5"
  # "multi-bit-watermark poseidon 20 15 16 6 4"
  # "multi-bit-watermark poseidon 40 15 16 6 2"

  # "multi-bit-watermark poseidon2 2 15 16 6 40"
  # "multi-bit-watermark poseidon2 4 15 16 6 20"
  # "multi-bit-watermark poseidon2 5 15 16 6 16"
  # "multi-bit-watermark poseidon2 8 15 16 6 10"
  # "multi-bit-watermark poseidon2 10 15 16 6 8"
  # "multi-bit-watermark poseidon2 16 15 16 6 5"
  # "multi-bit-watermark poseidon2 20 15 16 6 4"
  # "multi-bit-watermark poseidon2 40 15 16 6 2"

  # # 960 (160)
  # "multi-bit-watermark mimc 4 15 16 6 40"
  # "multi-bit-watermark mimc 8 15 16 6 20"
  # "multi-bit-watermark mimc 10 15 16 6 16"
  # "multi-bit-watermark mimc 16 15 16 6 10"
  # "multi-bit-watermark mimc 20 15 16 6 8"
  # "multi-bit-watermark mimc 32 15 16 6 5"
  # "multi-bit-watermark mimc 40 15 16 6 4"
  # "multi-bit-watermark mimc 80 15 16 6 2"

  # "multi-bit-watermark poseidon 4 15 16 6 40"
  # "multi-bit-watermark poseidon 8 15 16 6 20"
  # "multi-bit-watermark poseidon 10 15 16 6 16"
  # "multi-bit-watermark poseidon 16 15 16 6 10"
  # "multi-bit-watermark poseidon 20 15 16 6 8"
  # "multi-bit-watermark poseidon 32 15 16 6 5"
  # "multi-bit-watermark poseidon 40 15 16 6 4"
  # "multi-bit-watermark poseidon 80 15 16 6 2"

  # "multi-bit-watermark poseidon2 4 15 16 6 40"
  # "multi-bit-watermark poseidon2 8 15 16 6 20"
  # "multi-bit-watermark poseidon2 10 15 16 6 16"
  # "multi-bit-watermark poseidon2 16 15 16 6 10"
  # "multi-bit-watermark poseidon2 20 15 16 6 8"
  # "multi-bit-watermark poseidon2 32 15 16 6 5"
  # "multi-bit-watermark poseidon2 40 15 16 6 4"
  # "multi-bit-watermark poseidon2 80 15 16 6 2"

  # 1920 (320)
  # "multi-bit-watermark mimc 4 15 16 6 80"
  # "multi-bit-watermark mimc 8 15 16 6 40"
  # "multi-bit-watermark mimc 10 15 16 6 32"
  # "multi-bit-watermark mimc 16 15 16 6 20"
  # "multi-bit-watermark mimc 20 15 16 6 16"
  # "multi-bit-watermark mimc 32 15 16 6 10"
  # "multi-bit-watermark mimc 40 15 16 6 8"
  # "multi-bit-watermark mimc 64 15 16 6 5"
  # "multi-bit-watermark mimc 80 15 16 6 4"

  # "multi-bit-watermark poseidon 4 15 16 6 80"
  # "multi-bit-watermark poseidon 8 15 16 6 40"
  # "multi-bit-watermark poseidon 10 15 16 6 32"
  # "multi-bit-watermark poseidon 16 15 16 6 20"
  # "multi-bit-watermark poseidon 20 15 16 6 16"
  # "multi-bit-watermark poseidon 32 15 16 6 10"
  # "multi-bit-watermark poseidon 40 15 16 6 8"
  # "multi-bit-watermark poseidon 64 15 16 6 5"
  # "multi-bit-watermark poseidon 80 15 16 6 4"

  # "multi-bit-watermark poseidon2 4 15 16 6 80"
  # "multi-bit-watermark poseidon2 8 15 16 6 40"
  # "multi-bit-watermark poseidon2 10 15 16 6 32"
  # "multi-bit-watermark poseidon2 16 15 16 6 20"
  # "multi-bit-watermark poseidon2 20 15 16 6 16"
  # "multi-bit-watermark poseidon2 32 15 16 6 10"
  # "multi-bit-watermark poseidon2 40 15 16 6 8"
  # "multi-bit-watermark poseidon2 64 15 16 6 5"
  # "multi-bit-watermark poseidon2 80 15 16 6 4"

  # "multi-bit-watermark poseidon 25 15 16 6 40"
  # "multi-bit-watermark poseidon 40 15 16 6 50"
  # "multi-bit-watermark poseidon 50 15 16 6 200"
  # "multi-bit-watermark poseidon 100 15 16 6 100"


  # "multi-bit-watermark poseidon 5 15 16 6 8"
  # "multi-bit-watermark poseidon 5 15 32 6 8"
  # "multi-bit-watermark poseidon 10 15 64 6 4"
  # "multi-bit-watermark poseidon 10 15 128 6 4"
  # "multi-bit-watermark poseidon 5 15 64 6 8"
  # "multi-bit-watermark poseidon 5 15 128 6 8"

  "multi-bit-watermark poseidon 5 15 16 6 32"
  "multi-bit-watermark poseidon 5 15 32 6 32"
  "multi-bit-watermark poseidon 8 15 32 6 20"
  "multi-bit-watermark poseidon 8 15 64 6 20"
  "multi-bit-watermark poseidon 10 15 64 6 16"
  "multi-bit-watermark poseidon 10 15 128 6 16"
  "multi-bit-watermark poseidon 16 15 128 6 10"

  "multi-bit-watermark poseidon 8 15 16 6 40"
  "multi-bit-watermark poseidon 8 15 32 6 40"
  "multi-bit-watermark poseidon 10 15 32 6 32"
  "multi-bit-watermark poseidon 10 15 64 6 32"
  "multi-bit-watermark poseidon 16 15 64 6 20"
  "multi-bit-watermark poseidon 16 15 128 6 20"
  "multi-bit-watermark poseidon 20 15 128 6 16"


  # "multi-bit-watermark mimc 4 15 16 6 80"
  # "multi-bit-watermark mimc 8 15 16 6 40"
  # "multi-bit-watermark mimc 10 15 16 6 32"
  # "multi-bit-watermark mimc 16 15 16 6 20"
  # "multi-bit-watermark mimc 20 15 16 6 16"
  # "multi-bit-watermark mimc 32 15 16 6 10"
  # "multi-bit-watermark mimc 40 15 16 6 8"
  # "multi-bit-watermark mimc 64 15 16 6 5"
  # "multi-bit-watermark mimc 80 15 16 6 4"



)

# 获取参数总数
total=${#parameters[@]}

# 进度文件
progress_file="progress.txt"

# 恢复上次中断的进度 (如果存在)
if [[ -f "$progress_file" ]]; then
  last_index=$(cat "$progress_file")
else
  last_index=-1
fi

# 显示初始进度条
echo -n "Progress: ["
for ((i=0; i<50; i++)); do echo -n " "; done
echo -n "]"
echo -n "\r"

# 循环遍历每组参数并执行
for i in "${!parameters[@]}"; do
  # 如果该任务已经执行过，则跳过
  if ((i <= last_index)); then
    continue
  fi

  param="${parameters[$i]}"
  
  # 将每组参数拆分成单独的变量
  read -r type_sampling host s_value m_value e_value p_value f_value <<< "$param"
  
  # 计算当前进度
  progress=$(( (i + 1) * 50 / total ))
  
  # 更新进度条
  echo -n "Progress: ["
  for ((j=0; j<progress; j++)); do echo -n "#"; done
  for ((j=progress; j<50; j++)); do echo -n " "; done
  echo -n "]"
  echo -n "\r"
  
  # 执行命令
  bash run.sh -t "$type_sampling" -h "$host" -s "$s_value" -m "$m_value" -e "$e_value" -p "$p_value" -f "$f_value"
  
  # 更新进度文件：记录当前任务的索引
  echo "$i" > "$progress_file"
  
  # 为了避免进度条覆盖，稍微延时一下
  sleep 0.1
done

# 完成
echo -e "\nExecution completed!"
# 清除进度文件
rm "$progress_file"
