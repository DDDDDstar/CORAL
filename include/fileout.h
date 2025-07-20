#include <fstream>
#include <iostream>
#include <string>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <future>
#include <atomic>

namespace efanna2e
{

    class FileOut
    {
    public:
        FileOut();
        ~FileOut();
        void Init(std::string filename);
        void print(std::string msg);
        void iprint(std::string msg); // important
        void eprint(std::string msg); // error
        void qprint(                  // queue
            const int head, const int tail, const int len, const int r = -1, const int w = -1);

    private:
        std::ofstream file_;
        std::mutex mtx;
        std::queue<std::string> msg_queue;
        std::condition_variable cv;
        std::atomic<bool> stop_flag{false};
        std::thread worker;

        void process_messages();
    };

    extern FileOut fo;
}