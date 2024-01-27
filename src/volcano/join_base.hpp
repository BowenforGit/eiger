#pragma once
#include <vector>

enum class MaterializeAlgo {
    GFUR,
    GFTR,
    ID_CLUSTER
};

enum class JoinAlgo {
    SMJ,
    SHJ
};

template<typename T>
class JoinBase {
public:
    virtual T join() = 0;
    virtual ~JoinBase() = default;
    virtual void print_stats() = 0;
    virtual std::vector<float> all_stats() = 0;
};