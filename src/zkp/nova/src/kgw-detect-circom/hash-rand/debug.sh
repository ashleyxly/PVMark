#!/bin/bash

# 参数列表
parameters=(
  "mimc 20 1 1 10"
)

# 获取参数总数
total=${#parameters[@]}

# 进度文件
progress_file="progress_debug.txt"

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
  read -r host s_value w_value d_value f_value <<< "$param"
  
  # 计算当前进度
  progress=$(( (i + 1) * 50 / total ))
  
  # 更新进度条
  echo -n "Progress: ["
  for ((j=0; j<progress; j++)); do echo -n "#"; done
  for ((j=progress; j<50; j++)); do echo -n " "; done
  echo -n "]"
  echo -n "\r"
  
  # 执行命令
  bash run.sh -h "$host" -s "$s_value" -w "$w_value" -d "$d_value" -f "$f_value"
  
  # 更新进度文件：记录当前任务的索引
  echo "$i" > "$progress_file"
  
  # 为了避免进度条覆盖，稍微延时一下
  sleep 0.1
done

# 完成
echo -e "\nExecution completed!"
# 清除进度文件
rm "$progress_file"
