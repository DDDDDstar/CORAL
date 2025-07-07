#include "efanna2e/neighbor.h"
#include "efanna2e/distance.h"
#include <shared_mutex>
#include <mutex>
#include <vector>
#include <limits>

namespace efanna2e {
class Candidates {
public:
    Candidates() = default;
    Candidates(const uint32_t M_pjbp, uint32_t self_id);
    ~Candidates() = default;

    void Set(const uint32_t M_pjbp, uint32_t self_id);

    void AddNbr(const uint32_t id, const float dist);
    void FilterNbrs(const float *data, 
                    const size_t dim, 
                    const Distance* distance,
                    std::vector<uint32_t> &res);

private:
    std::shared_ptr<Neighbor> cnbrs_head, cnbrs_tail;
    std::shared_mutex cnbrs_lock;
    uint32_t M;
    uint32_t self_id;
    uint32_t cnbrs_num;
    bool cnbrs_full;
};

}