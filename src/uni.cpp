
#include "uni.h"

#include <cuda_runtime.h>

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#define MAX_MSG_LEN 20

namespace efanna2e {
ProConfig PC;

double welford_variance(const uint32_t* v, int N) {
    double mean = 0.0;
    double M2 = 0.0;
    int n = 0;

    for (int i = 0; i < N; i++) {
        const uint32_t x = v[i];
        n++;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    if (n < 2) return 0.0;

    return M2 / n;  // 总体方差
    // return M2 / (n - 1); // 样本方差
}

void normalize_L2_omp(float* x, size_t n, size_t d) {
#pragma omp parallel for
    for (long long i = 0; i < (long long)n; ++i) {
        float* row = x + i * d;
        float norm2 = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            norm2 += row[j] * row[j];
        }
        float norm = std::sqrt(norm2);
        if (norm > 0.0f) {
            float inv = 1.0f / norm;
            for (size_t j = 0; j < d; ++j) {
                row[j] *= inv;
            }
        }
    }
}
int testAndSetBit(int* bitset, int bit_index) {
    int *addr = bitset + (bit_index >> 5), mask = 1 << (bit_index & 31), old = *addr;
    if (old & mask) return 1;  // 已经是 1

    *addr = old | mask;  // 直接置位
    return 0;            // 成功从 0 -> 1
}

int numDigits(int n) {
    n = std::abs(n);       // 处理负数
    if (n == 0) return 1;  // 特判 0
    return static_cast<int>(std::log10(n)) + 1;
}

// BatchResult::BatchResult(int* d_knn, const int k, const int batch, const double time_s)
//     : batch(batch), time_s(time_s) {
//     knn.assign(k * batch, -1);
//     cudaMemcpy(knn.data(), d_knn, batch * k * sizeof(int), cudaMemcpyDeviceToHost);
//     std::string res;
//     bool right = true;
//     for (int i = 0; i < batch * k; i++) {
//         res += TOS(knn[i]) + " ";
//         if (knn[i] < 0) right = false;
//     }
//     if (!right) fo.eprint("knn error: " + res);
// }

void writeCSV(const std::vector<std::vector<std::string>>& data, const std::string& filename,
              const std::ios_base::openmode mode) {
    std::ofstream file(filename, mode);
    if (!file.is_open()) fo.eprint("writeCSV: Can not open file: " + filename);

    for (const auto& row : data) {
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

std::unordered_map<std::string, Avg_Time> avg_times;

void record_time(const std::string name, float time_s) {
    if (!avg_times.contains(name)) avg_times[name] = Avg_Time();
    avg_times[name].add(time_s);
}

void print_times() {
    std::vector<std::pair<std::string, Avg_Time>> avg_times_vec(avg_times.begin(),
                                                                avg_times.end());
    std::sort(avg_times_vec.begin(), avg_times_vec.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    std::string s;
    for (auto& pair : avg_times_vec) {
        const std::string& func = pair.first;
        s += (func.size() >= MAX_MSG_LEN ? func
                                         : func + std::string(MAX_MSG_LEN - func.size(), ' ')) +
             " 任务平均耗时: " + std::to_string(pair.second.get()) +
             " s; 总耗时: " + std::to_string(pair.second.get_total()) + " s\n";
    }
    fo.iprint(s);
}

std::string avg_time_serialize() {
    std::string str = std::to_string(avg_times.size()) + ",";
    for (auto& it : avg_times) {
        str += it.first + "," + it.second.serialize() + ",";
    }
    return str;
}
void avg_time_deserialize(const std::string& str) {
    // fo.print("avg_time_deserialize str:\n" + str);

    std::istringstream iss(str);
    int size;
    std::string temp;
    std::getline(iss, temp, ',');
    size = std::stoi(temp);
    for (int i = 0; i < size; i++) {
        std::string time_str, n_str;
        std::getline(iss, temp, ',');
        std::getline(iss, time_str, ',');
        std::getline(iss, n_str, ',');
        avg_times[temp] = Avg_Time(time_str, n_str);
    }

    print_times();  // 打印反序列化后的结果
}
};  // namespace efanna2e