#include "mem_manager.hpp"

constexpr size_t mem_pool_size = 25769803776*0.9;
UserSpaceMM<mem_pool_size>* mm {nullptr};