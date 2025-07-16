#include <future>

#include "fileout.h"

using namespace efanna2e;

FileOut fo;

void FileOut::Init(std::string filename)
{
    if (filename == "std")
        return;
    file_.open(filename);
    if (!file_)
    { // 检查文件是否成功打开
        std::cerr << "无法打开文件！" << std::endl;
        exit(EXIT_FAILURE);
    }
}

FileOut::FileOut() {}

FileOut::~FileOut()
{
    if (file_.is_open())
        file_.close();
}

// 打印消息到文件或标准输出
void FileOut::print_flash(std::string msg)
{
    std::async(std::launch::async, [this, msg]()
               {
        std::lock_guard<std::mutex> lock(mtx);
        // 如果文件已打开，则将消息写入文件，否则写入标准输出
        auto &out = file_.is_open() ? file_ : std::cout;
        // 将消息写入输出流，并换行
        out << msg << std::endl; });
}

void FileOut::print(std::string msg)
{
    std::async(std::launch::async, [this, msg]()
               {
        std::lock_guard<std::mutex> lock(mtx);
        auto &out = file_.is_open() ? file_ : std::cout;
        out << msg << "\n"; });
}

void FileOut::iprint(std::string msg)
{
    std::async(std::launch::async, [this, msg]()
               {
        std::lock_guard<std::mutex> lock(mtx);
        auto &out = file_.is_open() ? file_ : std::cout;
        out << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
        out << msg << "\n";
        out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl; });
}

void FileOut::eprint(std::string msg)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto &out = file_.is_open() ? file_ : std::cout;
    out << "--------------------ERROR--------------------\n";
    out << msg << "\n";
    out << "---------------------------------------------" << std::endl;
    exit(EXIT_FAILURE);
}

void FileOut::qprint(
    const int head, int tail, const int len, const int r, const int w)
{
    std::async(std::launch::async, [this, head, tail, len, r, w]()
               {
        std::lock_guard<std::mutex> lock(mtx);
        auto &out = file_.is_open() ? file_ : std::cout;
        out << "KNN QUEUE: [ ";
        for (int i = 0; i < len; ++i) {
            if (i == r) out << "x";
            else if (i == w) out << "*";
            else if (head == tail) out << "_";
            else if (head < tail) {
                if (i < head || i >= tail) out << "_";
                else out << i;
            } else {
                if (i < head && i >= tail) out << "_";
                else out << i;
            }
        }
        out << " ]" << std::endl; });
}