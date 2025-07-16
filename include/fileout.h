#include <fstream>
#include <iostream>
#include <string>
#include <mutex>

namespace efanna2e
{

    class FileOut
    {
    public:
        FileOut();
        ~FileOut();
        void Init(std::string filename);
        void print_flash(std::string msg);
        void print(std::string msg);
        void iprint(std::string msg); // important
        void eprint(std::string msg); // error
        void qprint(                  // queue
            const int head, int tail, const int len, const int r = -1, const int w = -1);

    private:
        std::ofstream file_;
        std::mutex mtx;
    };

    extern FileOut fo;
}