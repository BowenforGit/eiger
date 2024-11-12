#pragma once

#include <string>

class Operator {
public:
    float next_this;
    std::string op_name;
    Operator(std::string&& str) : op_name(str), next_this(0) {}
    virtual float get_op_time() = 0;
};
