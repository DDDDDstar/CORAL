
#include "uni.h"

#include <cuda_runtime.h>

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "fileout.h"

using namespace efanna2e;

int numDigits(int n) {
    n = std::abs(n);       // 处理负数
    if (n == 0) return 1;  // 特判 0
    return static_cast<int>(std::log10(n)) + 1;
}

BatchResult::BatchResult(int *d_knn, const int k, const int batch, const double time_ms)
    : batch(batch), time_ms(time_ms) {
    knn.resize(k * batch, -1);
    cudaMemcpy(knn.data(), d_knn, batch * k * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch * k; i++) {
        assert(knn[i] >= 0);
    }
}

BatchResult::BatchResult(const int k, const int batch) : batch(batch) { knn.resize(k * batch); }

void writeCSV(const std::vector<std::vector<std::string>> &data, const std::string &filename,
              const std::ios_base::openmode mode) {
    std::ofstream file(filename, mode);
    if (!file.is_open()) fo.eprint("writeCSV: Can not open file: " + filename);

    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";  // 逗号分隔
        }
        file << "\n";
    }

    file.close();
}

std::string getCurrentDateTimeString() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为 time_t 类型（秒级精度）
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // 转换为本地时间（线程不安全，需注意）
    std::tm local_tm = *std::localtime(&now_time);

    // 使用 stringstream 格式化输出
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d-%H:%M:%S");

    return oss.str();
}