#include <future>
#include <filesystem>

#include "fileout.h"

namespace efanna2e
{

    FileOut fo;

    void FileOut::Init(std::string filename)
    {
        if (filename == "std")
            return;

        std::filesystem::path dir_path = std::filesystem::path(filename).parent_path();
        if (!dir_path.empty() && !std::filesystem::exists(dir_path))
            std::filesystem::create_directories(dir_path); // 递归创建目录

        file_.open(filename);
        if (!file_)
        { // 检查文件是否成功打开
            std::cerr << "无法打开文件！" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    FileOut::FileOut() : worker(&FileOut::process_messages, this) {}

    FileOut::~FileOut()
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop_flag = true;
        }
        cv.notify_one();
        if (worker.joinable())
            worker.join();

        if (file_.is_open())
            file_.close();
    }

    // 后台消息处理线程
    void FileOut::process_messages()
    {
        while (true)
        {
            std::string msg;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]
                        { return !msg_queue.empty() || stop_flag; });

                if (stop_flag && msg_queue.empty())
                    break;
                if (msg_queue.empty())
                    continue;

                msg = std::move(msg_queue.front());
                msg_queue.pop();
            }

            auto &out = file_.is_open() ? file_ : std::cout;
            out << msg << std::endl;
        }
    }

    void FileOut::print(std::string msg)
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            msg_queue.push(msg);
        }
        cv.notify_one();
    }

    void FileOut::iprint(std::string msg)
    {
        std::string decorated_msg =
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" +
            msg + "\n" +
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";

        {
            std::lock_guard<std::mutex> lock(mtx);
            msg_queue.push(decorated_msg);
        }
        cv.notify_one();
    }

    void FileOut::eprint(std::string msg)
    {
        std::string decorated_msg =
            "--------------------ERROR--------------------\n" +
            msg + "\n" +
            "---------------------------------------------";

        {
            std::lock_guard<std::mutex> lock(mtx);
            auto &out = file_.is_open() ? file_ : std::cout;
            out << decorated_msg << std::endl;
        }
        exit(EXIT_FAILURE); // 错误消息直接退出
    }

    void FileOut::qprint(
        const int head, const int tail, const int len, const int r, const int w)
    {
        std::string queue_state = "KNN QUEUE: [ ";
        for (int i = 0; i < len; ++i)
        {
            if (i == r)
                queue_state += "x";
            else if (i == w)
                queue_state += "*";
            else if (head == tail)
                queue_state += "_";
            else if (head < tail)
            {
                if (i < head || i >= tail)
                    queue_state += "_";
                else
                    queue_state += std::to_string(i);
            }
            else
            {
                if (i < head && i >= tail)
                    queue_state += "_";
                else
                    queue_state += std::to_string(i);
            }
        }
        queue_state += " ]";

        {
            std::lock_guard<std::mutex> lock(mtx);
            msg_queue.push(queue_state);
        }
        cv.notify_one();
    }
}