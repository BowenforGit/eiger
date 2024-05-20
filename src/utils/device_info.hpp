#pragma once

#ifdef SM860
constexpr int SM_COUNT = 82;
constexpr int MAX_SMEM_BYTES  = (100*1024);
#elif defined (SM800)
constexpr int SM_COUNT = 108;
constexpr int MAX_SMEM_BYTES = (163*1024);
#elif
static_assert(false, "Unsupported compute capability.");
#endif