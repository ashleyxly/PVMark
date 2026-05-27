pragma circom 2.1.1;
// 定义一个函数，接受n个输入并计算它们的总和
template Summation(n) {
    // 声明输入数组
    signal input in[n];
    // 声明输出信号，用来保存总和
    signal output sum;

    // 声明递归结构
    signal temp[n];
    
    // 初始化第一个元素为输入的第一个值
    temp[0] <== in[0];
    
    // 递归地计算累加的结果
    for (var i = 1; i < n; i++) {
        temp[i] <== temp[i-1] + in[i];
    }

    // 最终将累加的结果赋值给sum
    sum <== temp[n-1];
}

// component main = Sum(5); // 举例使用5个输入
