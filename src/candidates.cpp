#include "candidates.h"
#include <cassert>

#define PROJECTION_SLACK 5

namespace efanna2e {
Candidates::Candidates(const uint32_t M_pjbp, const uint32_t self_id)
    : M(M_pjbp), self_id(self_id) {}

void Candidates::Set(const uint32_t M_pjbp, uint32_t self_id) {
    M = M_pjbp;
    self_id = self_id;
}

void Candidates::AddNbr(const uint32_t id, const float dist) {
    std::unique_lock<std::shared_mutex> lock(cnbrs_lock);

    assert(!cnbrs_full || (cnbrs_head && cnbrs_tail));
    // 快速路径：不需要添加的情况直接返回
    if (cnbrs_full && dist >= cnbrs_tail->distance) return;

    std::shared_ptr<Neighbor> prev, cur = cnbrs_head;
    if (cur && cur->id == id) return;  // 已存在则不插入
    // 插入前定位插入位置（按升序排列）
    while (cur && cur->distance < dist) {
        if (cur->id == id) return;  // 已存在则不插入
        prev = cur;
        cur = cur->next;
    }
    // 插入新节点
    auto new_nbr = std::make_shared<Neighbor>(id, dist, cur, prev, false);
    if (!prev) cnbrs_head = new_nbr;
    else prev->next = new_nbr;
    if (cur) cur->prev = new_nbr;
    else cnbrs_tail = new_nbr;
    // 若插入之前已满则需要修剪尾部
    if (cnbrs_full) {
        auto new_tail = cnbrs_tail;
        if (new_tail->prev.expired()) {
            std::cout << "new_tail->prev.expired(), num: " << cnbrs_num << std::endl;
            cnbrs_head = cnbrs_tail = new_tail;
        } else {
            new_tail = new_tail->prev.lock();
            if (new_tail) {
                new_tail->next.reset();
                cnbrs_tail = new_tail;
            } else {
                // 异常情况：清空链表
                std::cout << "Abnormal situation: empty the linked list" << std::endl;
                cnbrs_head.reset();
                cnbrs_tail.reset();
                cnbrs_num = 0;
                cnbrs_full = false;
                return;
            }
        }
    }
    // 不满时增加计数，可能翻满
    else if (++cnbrs_num >= M * PROJECTION_SLACK) cnbrs_full = true;
}

void Candidates::FilterNbrs(const float *data, 
                            const size_t dim, 
                            const Distance* distance,
                            std::vector<uint32_t> &res) {
    if (cnbrs_head == nullptr) return;
    
    std::vector<uint32_t> result, alternative;
    result.reserve(M);
    alternative.reserve(M);
    result.push_back(cnbrs_head->id);
    for (auto cnbr = cnbrs_head->next; cnbr != nullptr && result.size() < M; cnbr = cnbr->next) {
        bool occlude = false;
        for (const auto &res_nbr_id: result) {
            if (distance->compare(
                    data + dim * cnbr->id, 
                    data + dim * res_nbr_id, dim) <= cnbr->distance) {
                if (alternative.size() < M) alternative.push_back(cnbr->id);
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            result.push_back(cnbr->id);
        }
    }
    for (uint32_t i = 0; i < alternative.size() && result.size() < M; ++i) {
        result.push_back(alternative[i]);
    }
    res = std::move(result);
    cnbrs_head = nullptr;  // 清空候选者列表
}
}