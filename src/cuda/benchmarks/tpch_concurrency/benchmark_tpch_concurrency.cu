// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Two-query TPC-H concurrency benchmark (single-invocation harness).
//
// One invocation measures ONE mode:
//   --mode isolated   --qa N                  -> T_Q  (single-query latency)
//   --mode concurrent --qa N --qb M           -> C_a, C_b, makespan, start_skew
//   --mode batch-isolated   --qa N --iters K  -> B_Q  (K-iter batch total + per-iter)
//   --mode batch-concurrent --qa N --qb M --iters K --policy dedicated|alternating
//
// Design/spec: src/cuda/benchmarks/tpch_concurrency/DESIGN.md
//
// Why host threads (not the operation-concurrency gate): TPC-H execute() reads
// data-dependent sizes back to the host mid-query, so a whole query cannot be
// pre-enqueued behind a gate. Each query runs on its own thread + stream; the
// two threads self-align on a host barrier and stamp a common t0.
//
// Primary metric = host wall-clock (steady_clock); secondary = per-stream CUDA
// events. Sources are pre-warmed (resolver keeps tables GPU-resident); only
// execute()+sync is timed. No clock pinning available on this box, so we record
// the live SM clock + throttle reasons per sample via NVML.

#include <cuda_runtime.h>
#include <nvml.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "exec/executor.hpp"
#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/column_id.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"
#include "tpch/verify.hpp"

// Green-context SM-partition helper (green_setup / green_setup_pair /
// green_teardown). Routes a query (or a co-running pair) onto a disjoint SM
// slice so the two contend spatially instead of time-sharing the full device.
#include "benchmarks/green-ctx/green_ctx.hpp"

namespace fs = std::filesystem;
namespace {

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error(m); }

void ck(cudaError_t s, const char* what) {
    if (s != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(s));
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double event_ms(cudaEvent_t a, cudaEvent_t b) {
    float v = 0.f;
    ck(cudaEventElapsedTime(&v, a, b), "cudaEventElapsedTime");
    return v;
}

// ---------------------------------------------------------------------------
// NVML SM-clock + throttle sampler (no pinning available; record instead)
// ---------------------------------------------------------------------------
struct NvmlProbe {
    nvmlDevice_t dev{};
    bool ok{false};
    void init(int cuda_dev) {
        if (nvmlInit_v2() != NVML_SUCCESS) return;
        char bus[64] = {0};
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) != cudaSuccess) return;
        if (nvmlDeviceGetHandleByPciBusId_v2(bus, &dev) != NVML_SUCCESS) return;
        ok = true;
    }
    unsigned sm_mhz() {
        unsigned c = 0;
        if (ok) nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c);
        return c;
    }
    unsigned long long throttle() {
        unsigned long long r = 0;
        if (ok) nvmlDeviceGetCurrentClocksThrottleReasons(dev, &r);
        return r;
    }
    ~NvmlProbe() { if (ok) nvmlShutdown(); }
};

// ---------------------------------------------------------------------------
// statistics: distribution + bootstrap CI on the median
// ---------------------------------------------------------------------------
struct Dist {
    int n{};
    double median{}, mean{}, stddev{}, p5{}, p95{}, ci_lo{}, ci_hi{};
    double rel_hw() const {  // relative CI half-width on the median
        return median > 0 ? (ci_hi - ci_lo) / (2.0 * median) : 0.0;
    }
};

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    double idx = q * (v.size() - 1);
    std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

Dist summarize(const std::vector<double>& x, std::uint64_t seed, int boot = 2000) {
    Dist d;
    d.n = static_cast<int>(x.size());
    if (x.empty()) {
        d.median = d.mean = d.stddev = d.p5 = d.p95 = d.ci_lo = d.ci_hi =
            std::numeric_limits<double>::quiet_NaN();
        return d;  // empty (e.g. iter_b of a self-pair) → NaN, not a misleading 0
    }
    d.median = percentile(x, 0.5);
    d.mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double var = 0;
    for (double v : x) var += (v - d.mean) * (v - d.mean);
    d.stddev = x.size() > 1 ? std::sqrt(var / (x.size() - 1)) : 0.0;
    d.p5 = percentile(x, 0.05);
    d.p95 = percentile(x, 0.95);
    // bootstrap CI on the median
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, x.size() - 1);
    std::vector<double> meds(boot);
    std::vector<double> resample(x.size());
    for (int b = 0; b < boot; ++b) {
        for (auto& r : resample) r = x[pick(rng)];
        meds[b] = percentile(resample, 0.5);
    }
    d.ci_lo = percentile(meds, 0.025);
    d.ci_hi = percentile(meds, 0.975);
    return d;
}

// ---------------------------------------------------------------------------
// L2 purge between samples (own tiny implementation; avoids linking concurrency)
// ---------------------------------------------------------------------------
struct L2Purger {
    void* buf{};
    std::size_t bytes{};
    cudaStream_t stream{};
    void init(int dev) {
        int l2 = 0;
        ck(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev), "l2 size");
        bytes = static_cast<std::size_t>(std::max(l2, 1));
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "purge stream");
        ck(cudaMalloc(&buf, bytes), "purge malloc");
    }
    void purge() {
        if (!buf) return;
        ck(cudaMemsetAsync(buf, 0, bytes, stream), "purge memset");
        ck(cudaStreamSynchronize(stream), "purge sync");
    }
    ~L2Purger() {
        if (buf) cudaFree(buf);
        if (stream) cudaStreamDestroy(stream);
    }
};

// ---------------------------------------------------------------------------
// query plan builder (mirrors run_tpch_ir.cpp)
// ---------------------------------------------------------------------------
gamax::ir::PlanPtr build_query(int qid, const gamax::ir::QueryBuilder& qb) {
    using namespace gamax::ir::tpch;
    switch (qid) {
        case 1:  return build_q1(qb);   case 2:  return build_q2(qb);
        case 3:  return build_q3(qb);   case 4:  return build_q4(qb);
        case 5:  return build_q5(qb);   case 6:  return build_q6(qb);
        case 7:  return build_q7(qb);   case 8:  return build_q8(qb);
        case 9:  return build_q9(qb);   case 10: return build_q10(qb);
        case 11: return build_q11(qb);  case 12: return build_q12(qb);
        case 13: return build_q13(qb);  case 14: return build_q14(qb);
        case 15: return build_q15(qb);  case 16: return build_q16(qb);
        case 17: return build_q17(qb);  case 18: return build_q18(qb);
        case 19: return build_q19(qb);  case 20: return build_q20(qb);
        case 21: return build_q21(qb);  case 22: return build_q22(qb);
        default: fail("query " + std::to_string(qid) + " not wired");
    }
    return nullptr;
}

std::string golden_path(int qid, int sf) {
    const char* dp = std::getenv("TPCH_DATA_PATH");
    if (!dp || !*dp) fail("TPCH_DATA_PATH not set");
    std::ostringstream os;
    os << dp << "/sf" << sf << "/golden/q" << (qid < 10 ? "0" : "") << qid << ".parquet";
    return os.str();
}

// Plan build + lowering touches shared IR rule/registry state; guard the
// (untimed, one-time) setup so two workers never build concurrently. execute()
// itself runs unguarded and concurrent — that is the contention we measure.
std::mutex g_setup_mutex;

// ---------------------------------------------------------------------------
// QueryRunner: owns one query's plan/ExecPlan/reader/resolver, bound to a
// stream. execute_once() runs execute() on that stream and returns the result
// (caller syncs + retires).
// ---------------------------------------------------------------------------
class QueryRunner {
public:
    QueryRunner(int qid, int sf, cudaStream_t stream) : qid_(qid), stream_(stream) {
        std::lock_guard<std::mutex> lk(g_setup_mutex);
        catalog_ = std::make_unique<gamax::ir::Catalog>(gamax::ir::tpch::make_tpch_catalog());
        gamax::ir::IdGenerator gen;
        gamax::ir::QueryBuilder qb(*catalog_, gen);
        auto plan = build_query(qid, qb);
        plan = gamax::ir::apply_rules(plan);
        gamax::ir::LowerOptions lower_opts;
        gamax::ir::tpch::apply_query_hints(qid, lower_opts);
        ep_ = std::make_unique<gamax::exec::ExecPlan>(
            gamax::ir::lower_to_exec_plan(plan, *catalog_, lower_opts));
        params_ = std::make_unique<gamax::tpch::tpch_params>(sf);
        params_->stream = stream_;
        reader_ = std::make_unique<gamax::tpch::TpchTableReader>(*params_);
        resolver_ = std::make_unique<gamax::tpch::IrSourceResolver>(*reader_, *catalog_);
        source_fn_ = [this](const std::string& n, const std::vector<std::string>& c) {
            return (*resolver_)(n, c);
        };
    }

    // Execute on this runner's bound stream (or an override), returning the
    // result table. Does NOT synchronize.
    std::unique_ptr<gamax::table> execute_once(cudaStream_t stream_override = nullptr) {
        cudaStream_t s = stream_override ? stream_override : stream_;
        gamax::phys::Context ctx{s};
        return gamax::exec::execute(*ep_, source_fn_, ctx);
    }

    int qid() const { return qid_; }
    cudaStream_t stream() const { return stream_; }

private:
    int qid_;
    cudaStream_t stream_;
    std::unique_ptr<gamax::ir::Catalog> catalog_;
    std::unique_ptr<gamax::exec::ExecPlan> ep_;
    std::unique_ptr<gamax::tpch::tpch_params> params_;
    std::unique_ptr<gamax::tpch::TpchTableReader> reader_;
    std::unique_ptr<gamax::tpch::IrSourceResolver> resolver_;
    gamax::exec::SourceResolver source_fn_;
};

void validate_or_die(gamax::table& result, int qid, int sf) {
    auto gp = golden_path(qid, sf);
    auto vr = gamax::tpch::verify_result(result.view(), gp);
    if (!vr.ok) fail("VERIFY MISMATCH Q" + std::to_string(qid) + ": " + vr.diagnostic);
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------
struct Options {
    std::string mode;       // isolated | concurrent | batch-isolated | batch-concurrent
    int qa{-1}, qb{-1};
    int sf{10};
    int iters{10};          // K for batch
    std::string policy{"dedicated"};
    int warmups{3};
    int min_samples{10}, max_samples{30};
    double ci{0.03};
    bool profile{false};
    int device{0};
    int refresh_round{0};
    std::uint64_t seed{42};
    std::string out;        // path prefix -> <out>.raw.csv / <out>.summary.csv
    // Green-context SM partitioning (0 = native full device):
    int sms{0};             // isolated mode: run the single query on an N-SM partition
    int sms_a{0};           // concurrent mode: A requests N SMs, B gets the remainder
                            // (--policy carries the intent label: balanced|proportional|sweep)
    int total_sms{0};       // device SM count (filled in main; recorded on native rows)
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto eat = [&](const char* f) { return std::strcmp(argv[i], f) == 0; };
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + argv[i]);
            return argv[++i];
        };
        if (eat("--mode")) o.mode = val();
        else if (eat("--qa")) o.qa = std::stoi(val());
        else if (eat("--qb")) o.qb = std::stoi(val());
        else if (eat("--sf")) o.sf = std::stoi(val());
        else if (eat("--iters")) o.iters = std::stoi(val());
        else if (eat("--policy")) o.policy = val();
        else if (eat("--warmups")) o.warmups = std::stoi(val());
        else if (eat("--min-samples")) o.min_samples = std::stoi(val());
        else if (eat("--max-samples")) o.max_samples = std::stoi(val());
        else if (eat("--ci")) o.ci = std::stod(val());
        else if (eat("--profile")) o.profile = true;
        else if (eat("--device")) o.device = std::stoi(val());
        else if (eat("--refresh-round")) o.refresh_round = std::stoi(val());
        else if (eat("--seed")) o.seed = std::stoull(val());
        else if (eat("--out")) o.out = val();
        else if (eat("--sms")) o.sms = std::stoi(val());
        else if (eat("--sms-a")) o.sms_a = std::stoi(val());
        else fail(std::string("unknown arg: ") + argv[i]);
    }
    if (o.mode.empty()) fail("--mode required");
    if (o.qa < 0) fail("--qa required");
    if (o.out.empty()) fail("--out required");
    // Green-partition flags are mode-specific; reject the wrong one so a typo
    // (e.g. --sms in concurrent mode) can't silently run full-device instead.
    if (o.sms > 0 && o.mode != "isolated")
        fail("--sms (green partition) is only valid with --mode isolated");
    if (o.sms_a > 0 && o.mode != "concurrent")
        fail("--sms-a (green split) is only valid with --mode concurrent");
    if (o.sms_a > 0 && o.policy != "balanced" && o.policy != "proportional" &&
        o.policy != "sweep")
        fail("--sms-a requires --policy balanced|proportional|sweep (the split intent)");
    if (o.profile) { o.warmups = 1; o.min_samples = 2; o.max_samples = 2; }
    return o;
}

// ---------------------------------------------------------------------------
// CSV writers.  raw = wide (per measured sample); summary = long (per metric).
// ---------------------------------------------------------------------------
struct RawRow {
    std::string run_type, policy;
    int qa, qb, sf, rep, iter;
    double a_wall, b_wall, a_gpu, b_gpu, makespan, skew;
    unsigned sm_mhz;
    unsigned long long throttle;
    double wall_ts;
    int sms_a, sms_b;   // actual SM-partition sizes (native full-device = device total)
};

void write_raw(const std::string& path, const std::vector<RawRow>& rows) {
    std::ofstream f(path);
    if (!f) fail("cannot open for write: " + path);
    // (no stream precision set: doubles are emitted via std::to_string in num())
    f << "run_type,policy,qa,qb,sf,rep,iter,a_wall_ms,b_wall_ms,a_gpu_ms,b_gpu_ms,"
         "makespan_ms,start_skew_ms,sm_clock_mhz,throttle,wall_ts,sms_a,sms_b\n";
    auto num = [](double v) {
        return std::isnan(v) ? std::string("nan") : std::to_string(v);
    };
    for (auto& r : rows) {
        f << r.run_type << ',' << r.policy << ',' << r.qa << ',' << r.qb << ',' << r.sf << ','
          << r.rep << ',' << r.iter << ',' << num(r.a_wall) << ',' << num(r.b_wall) << ','
          << num(r.a_gpu) << ',' << num(r.b_gpu) << ',' << num(r.makespan) << ','
          << num(r.skew) << ',' << r.sm_mhz << ',' << r.throttle << ',' << num(r.wall_ts)
          << ',' << r.sms_a << ',' << r.sms_b << '\n';
    }
    f.flush();
    if (!f) fail("write failed (disk full?): " + path);
}

struct SummaryRow {
    std::string run_type, policy, metric;
    int qa, qb, sf, n, refresh_round;
    bool stable;
    Dist d;
    double wall_ts;
    int sms_a, sms_b;   // actual SM-partition sizes (native full-device = device total)
};

void write_summary(const std::string& path, const std::vector<SummaryRow>& rows) {
    std::ofstream f(path);
    if (!f) fail("cannot open for write: " + path);
    f << "run_type,policy,qa,qb,sf,metric,n,stable,refresh_round,"
         "median,mean,std,p5,p95,ci_lo,ci_hi,wall_ts,sms_a,sms_b\n";
    f.setf(std::ios::fixed);
    f.precision(6);
    for (auto& r : rows) {
        f << r.run_type << ',' << r.policy << ',' << r.qa << ',' << r.qb << ',' << r.sf << ','
          << r.metric << ',' << r.n << ',' << (r.stable ? 1 : 0) << ',' << r.refresh_round << ','
          << r.d.median << ',' << r.d.mean << ',' << r.d.stddev << ',' << r.d.p5 << ','
          << r.d.p95 << ',' << r.d.ci_lo << ',' << r.d.ci_hi << ',' << r.wall_ts
          << ',' << r.sms_a << ',' << r.sms_b << '\n';
    }
    f.flush();
    if (!f) fail("write failed (disk full?): " + path);
}

// ---------------------------------------------------------------------------
// one concurrent execution: two threads, host barrier, common t0.
// keep_results=true returns the result tables (for warmup validation).
// ---------------------------------------------------------------------------
struct ConcTiming { double c_a, c_b, makespan, skew, a_gpu, b_gpu; };

// A runs on stream sA, B on stream sB (callers alternate the stream assignment
// across reps to cancel per-stream arbitration bias). Exceptions thrown in a
// worker are captured and rethrown on the caller thread after join, so a
// co-runner OOM/CUDA error surfaces cleanly instead of calling std::terminate.
ConcTiming run_concurrent(QueryRunner& A, cudaStream_t sA, QueryRunner& B, cudaStream_t sB,
                          int device, std::unique_ptr<gamax::table>* out_a,
                          std::unique_ptr<gamax::table>* out_b) {
    std::barrier sync(2);
    struct Slot { double w_start, w_end, gpu; std::unique_ptr<gamax::table> res; } sa, sb;
    std::exception_ptr err_a, err_b;

    auto worker = [&](QueryRunner& R, cudaStream_t st, Slot& S, const char* tag,
                      std::exception_ptr& err) {
        cudaSetDevice(device);
        sync.arrive_and_wait();  // reached before any throwing work → no barrier deadlock
        try {
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "conc event create");
            ck(cudaEventCreate(&e1), "conc event create");
            S.w_start = now_ms();
            nvtxRangePushA(tag);
            ck(cudaEventRecord(e0, st), "conc event record");
            auto r = R.execute_once(st);
            ck(cudaEventRecord(e1, st), "conc event record");
            ck(cudaStreamSynchronize(st), "conc worker sync");
            nvtxRangePop();
            S.w_end = now_ms();
            S.gpu = event_ms(e0, e1);
            S.res = std::move(r);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        } catch (...) {
            S.w_end = now_ms();
            err = std::current_exception();
        }
    };

    std::string la = "conc/Q" + std::to_string(A.qid());
    std::string lb = "conc/Q" + std::to_string(B.qid());
    std::thread ta(worker, std::ref(A), sA, std::ref(sa), la.c_str(), std::ref(err_a));
    std::thread tb(worker, std::ref(B), sB, std::ref(sb), lb.c_str(), std::ref(err_b));
    ta.join();
    tb.join();
    if (err_a) std::rethrow_exception(err_a);
    if (err_b) std::rethrow_exception(err_b);

    double t0 = std::min(sa.w_start, sb.w_start);
    ConcTiming t;
    t.c_a = sa.w_end - t0;
    t.c_b = sb.w_end - t0;
    t.makespan = std::max(sa.w_end, sb.w_end) - t0;
    t.skew = std::fabs(sa.w_start - sb.w_start);
    t.a_gpu = sa.gpu;
    t.b_gpu = sb.gpu;
    if (out_a) *out_a = std::move(sa.res);
    if (out_b) *out_b = std::move(sb.res);
    return t;
}

// ---------------------------------------------------------------------------
// mode: isolated single-shot
// ---------------------------------------------------------------------------
int run_isolated(const Options& o, NvmlProbe& nvml, L2Purger& purger) {
    // Optional green-context SM partition. green_setup carves ~o.sms SMs (rounded
    // up to alignment) and a stream on that partition; the query loads + executes
    // on it. `g` lives at function scope so its RAII teardown fires AFTER the
    // inner QueryRunner block (the runner holds resident sources on the green
    // stream and must be destroyed before the stream is). 0 = native full device.
    const bool use_green = o.sms > 0;
    GreenCtx g;
    cudaStream_t stream;
    int part_sms = o.total_sms;              // native: the single query owns all SMs
    if (use_green) {
        part_sms = static_cast<int>(green_setup(o.device, o.sms, g));
        stream = reinterpret_cast<cudaStream_t>(g.gstream);
    } else {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "iso stream");
    }
    const char* rt = use_green ? "iso_green" : "iso_single";
    {
    QueryRunner R(o.qa, o.sf, stream);

    // warmups (untimed) + validate on the first
    for (int w = 0; w < o.warmups; ++w) {
        auto r = R.execute_once();
        ck(cudaStreamSynchronize(stream), "iso warmup sync");
        if (w == 0) validate_or_die(*r, o.qa, o.sf);
        r.reset();
    }

    std::vector<double> wall, gpu;
    std::vector<RawRow> raw;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    bool stable = false;
    for (int rep = 0; rep < o.max_samples; ++rep) {
        purger.purge();
        double ts = now_ms();
        cudaEventRecord(e0, stream);
        double w0 = now_ms();
        auto r = R.execute_once();
        cudaEventRecord(e1, stream);
        ck(cudaStreamSynchronize(stream), "iso sync");
        double w1 = now_ms();
        double gms = event_ms(e0, e1);
        r.reset();
        wall.push_back(w1 - w0);
        gpu.push_back(gms);
        RawRow row{rt, "-", o.qa, -1, o.sf, rep, 0,
                   w1 - w0, std::nan(""), gms, std::nan(""), w1 - w0, std::nan(""),
                   nvml.sm_mhz(), nvml.throttle(), ts, part_sms, 0};
        raw.push_back(row);
        int cnt = rep + 1;
        if (!o.profile && cnt >= o.min_samples && (cnt == o.max_samples || cnt % 5 == 0)) {
            if (summarize(wall, o.seed).rel_hw() <= o.ci) { stable = true; break; }
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    double ts = now_ms();
    Dist dw = summarize(wall, o.seed), dg = summarize(gpu, o.seed ^ 1);
    std::vector<SummaryRow> sum;
    sum.push_back({rt, "-", "T", o.qa, -1, o.sf, (int)wall.size(),
                   o.refresh_round, stable, dw, ts, part_sms, 0});
    sum.push_back({rt, "-", "T_gpu", o.qa, -1, o.sf, (int)gpu.size(),
                   o.refresh_round, stable, dg, ts, part_sms, 0});
    write_raw(o.out + ".raw.csv", raw);
    write_summary(o.out + ".summary.csv", sum);
    std::cout << "Q" << o.qa << " T=" << dw.median << " ms n=" << wall.size()
              << (use_green ? " sms=" + std::to_string(part_sms) : "")
              << (stable ? " stable" : " unstable") << "\n";
    }  // destroy R (frees resident sources) before the stream
    if (!use_green) cudaStreamDestroy(stream);  // green stream torn down by g's dtor
    return 0;
}

// ---------------------------------------------------------------------------
// mode: concurrent single-shot
// ---------------------------------------------------------------------------
int run_concurrent_mode(const Options& o, NvmlProbe& nvml, L2Purger& purger) {
    if (o.qb < 0) fail("--qb required for concurrent");
    // Optional DISJOINT green-context SM partitions: A gets ~o.sms_a SMs, B gets
    // the remainder (full-device tiling, zero idle SMs). The two green ctxs live
    // at function scope so their RAII teardown fires AFTER the inner runner block
    // (runners hold resident sources on the green streams). 0 = native full-device
    // time-sharing (both queries on the whole device, regular streams).
    const bool use_green = o.sms_a > 0;
    GreenCtx ga, gb;
    cudaStream_t sa, sb;
    int sms_a_act = o.total_sms, sms_b_act = o.total_sms;  // native: both own all SMs
    if (use_green) {
        green_setup_pair(o.device, o.sms_a, ga, gb);
        sa = reinterpret_cast<cudaStream_t>(ga.gstream);
        sb = reinterpret_cast<cudaStream_t>(gb.gstream);
        sms_a_act = static_cast<int>(ga.sms);
        sms_b_act = static_cast<int>(gb.sms);
    } else {
        ck(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking), "sa");
        ck(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking), "sb");
    }
    const char* rt = use_green ? "conc_green" : "conc_single";
    const std::string pol = use_green ? o.policy : "-";
    // Runners must be destroyed while their streams are still alive (RMM arena
    // dealloc synchronizes the buffer's stream). Scope them before the destroys.
    {
    QueryRunner A(o.qa, o.sf, sa);
    QueryRunner B(o.qb, o.sf, sb);

    // concurrent warmups; validate both on the first (catch concurrency races)
    for (int w = 0; w < o.warmups; ++w) {
        std::unique_ptr<gamax::table> ra, rb;
        run_concurrent(A, sa, B, sb, o.device, &ra, &rb);
        if (w == 0) {
            validate_or_die(*ra, o.qa, o.sf);
            validate_or_die(*rb, o.qb, o.sf);
        }
    }

    std::vector<double> vca, vcb, vm, vsk;
    std::vector<RawRow> raw;
    bool stable = false;
    for (int rep = 0; rep < o.max_samples; ++rep) {
        purger.purge();
        double ts = now_ms();
        // Native: alternate which STREAM each query runs on across reps to cancel
        // per-stream arbitration bias (c_a/c_b stay keyed to A(qa)/B(qb)). Green:
        // NEVER swap — each query must stay on its DESIGNATED partition (swapping
        // would put A on B's SM slice and mislabel the split).
        bool swap = !use_green && (rep % 2) == 1;
        ConcTiming t = run_concurrent(A, swap ? sb : sa, B, swap ? sa : sb,
                                      o.device, nullptr, nullptr);
        double ca = t.c_a, cb = t.c_b, gpa = t.a_gpu, gpb = t.b_gpu;
        vca.push_back(ca);
        vcb.push_back(cb);
        vm.push_back(t.makespan);
        vsk.push_back(t.skew);
        raw.push_back({rt, pol, o.qa, o.qb, o.sf, rep, 0,
                       ca, cb, gpa, gpb, t.makespan, t.skew,
                       nvml.sm_mhz(), nvml.throttle(), ts, sms_a_act, sms_b_act});
        int cnt = rep + 1;
        if (!o.profile && cnt >= o.min_samples && (cnt == o.max_samples || cnt % 5 == 0)) {
            bool c = summarize(vca, o.seed).rel_hw() <= o.ci &&
                     summarize(vcb, o.seed).rel_hw() <= o.ci &&
                     summarize(vm, o.seed).rel_hw() <= o.ci;
            if (c) { stable = true; break; }
        }
    }

    double ts = now_ms();
    Dist da = summarize(vca, o.seed), db = summarize(vcb, o.seed ^ 2),
         dm = summarize(vm, o.seed ^ 3), dk = summarize(vsk, o.seed ^ 4);
    std::vector<SummaryRow> sum;
    auto push = [&](const char* m, const Dist& d, int n) {
        sum.push_back({rt, pol, m, o.qa, o.qb, o.sf, n,
                       o.refresh_round, stable, d, ts, sms_a_act, sms_b_act});
    };
    push("C_a", da, (int)vca.size());
    push("C_b", db, (int)vcb.size());
    push("M", dm, (int)vm.size());
    push("skew", dk, (int)vsk.size());
    write_raw(o.out + ".raw.csv", raw);
    write_summary(o.out + ".summary.csv", sum);
    std::cout << "Q" << o.qa << "+Q" << o.qb << " Ca=" << da.median
              << " Cb=" << db.median << " M=" << dm.median << " n=" << vm.size()
              << (use_green ? " sms=" + std::to_string(sms_a_act) + "/" +
                                  std::to_string(sms_b_act) : "")
              << (stable ? " stable" : " unstable") << "\n";
    }  // destroy A,B (free resident sources) before the streams
    if (!use_green) {  // green streams torn down by ga/gb dtors (after A,B destroyed)
        cudaStreamDestroy(sa);
        cudaStreamDestroy(sb);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// batch mode: each thread runs a submission SEQUENCE of runners (K iters).
// Metric of interest = batch-total wall time; per-iteration times (tagged by
// query) recover per-query throughput slowdown in both submission policies.
// ---------------------------------------------------------------------------
struct IterRec { int qid; double wall; };
struct BatchOut {
    double start{}, end{}, total{};
    std::vector<IterRec> iters;
    std::unique_ptr<gamax::table> keep;   // first result, kept for warmup validation
};

// Run `seq` once in order, retiring each result. A per-iter stream sync gives a
// clean iteration boundary (RMM arena free already syncs the stream); the two
// threads still overlap because each syncs only its OWN stream, not the device.
BatchOut run_batch_thread(const std::vector<QueryRunner*>& seq, int device,
                          std::barrier<>* bar, bool keep_first) {
    cudaSetDevice(device);
    if (bar) bar->arrive_and_wait();
    BatchOut o;
    o.start = now_ms();
    double prev = o.start;
    for (std::size_t k = 0; k < seq.size(); ++k) {
        QueryRunner* R = seq[k];
        std::string lbl = "batch/Q" + std::to_string(R->qid());
        nvtxRangePushA(lbl.c_str());
        auto res = R->execute_once();
        ck(cudaStreamSynchronize(R->stream()), "batch thread sync");
        nvtxRangePop();
        double t = now_ms();
        o.iters.push_back({R->qid(), t - prev});
        prev = t;
        if (keep_first && k == 0) { o.keep = std::move(res); }
        else { res.reset(); }
    }
    o.end = now_ms();
    o.total = o.end - o.start;
    return o;
}

constexpr int k_min_batches = 5;
constexpr int k_max_batches = 15;

int run_batch_isolated(const Options& o, NvmlProbe& nvml, L2Purger& purger) {
    int const maxb = o.profile ? 1 : k_max_batches;
    int const minb = o.profile ? 1 : k_min_batches;
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "batch iso stream");
    {
    QueryRunner R(o.qa, o.sf, stream);
    std::vector<QueryRunner*> seq(o.iters, &R);
    for (int w = 0; w < o.warmups; ++w) {
        auto wb = run_batch_thread(seq, o.device, nullptr, w == 0);
        if (w == 0 && wb.keep) validate_or_die(*wb.keep, o.qa, o.sf);
    }

    std::vector<double> totals, iters;
    std::vector<RawRow> raw;
    bool stable = false;
    for (int rep = 0; rep < maxb; ++rep) {
        purger.purge();
        double ts = now_ms();
        auto bo = run_batch_thread(seq, o.device, nullptr, false);
        totals.push_back(bo.total);
        for (std::size_t it = 0; it < bo.iters.size(); ++it) {
            iters.push_back(bo.iters[it].wall);
            raw.push_back({"batch_isolated", "-", o.qa, -1, o.sf, rep, (int)it,
                           bo.iters[it].wall, std::nan(""), std::nan(""), std::nan(""),
                           bo.total, std::nan(""), nvml.sm_mhz(), nvml.throttle(), ts,
                           o.total_sms, 0});
        }
        int cnt = rep + 1;
        if (!o.profile && cnt >= minb && summarize(totals, o.seed).rel_hw() <= o.ci) {
            stable = true; break;
        }
    }
    double ts = now_ms();
    Dist dB = summarize(totals, o.seed), dI = summarize(iters, o.seed ^ 7);
    std::vector<SummaryRow> sum;
    sum.push_back({"batch_isolated", "-", "B", o.qa, -1, o.sf, (int)totals.size(),
                   o.refresh_round, stable, dB, ts, o.total_sms, 0});
    sum.push_back({"batch_isolated", "-", "iter", o.qa, -1, o.sf, (int)iters.size(),
                   o.refresh_round, stable, dI, ts, o.total_sms, 0});
    write_raw(o.out + ".raw.csv", raw);
    write_summary(o.out + ".summary.csv", sum);
    std::cout << "batch-iso Q" << o.qa << " K=" << o.iters
              << " B=" << dB.median << " iter=" << dI.median << " ms\n";
    }
    cudaStreamDestroy(stream);
    return 0;
}

int run_batch_concurrent(const Options& o, NvmlProbe& nvml, L2Purger& purger) {
    if (o.qb < 0) fail("--qb required for batch-concurrent");
    bool const alt = (o.policy == "alternating");
    if (!alt && o.policy != "dedicated") fail("policy must be dedicated|alternating");
    int const maxb = o.profile ? 1 : k_max_batches;
    int const minb = o.profile ? 1 : k_min_batches;
    cudaStream_t sa, sb;
    ck(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking), "batch sa");
    ck(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking), "batch sb");
    {
    // Per-thread runners bound to that thread's stream (never share a runner
    // across threads). dedicated needs 2; alternating needs 4.
    QueryRunner A0(o.qa, o.sf, sa);
    QueryRunner B1(o.qb, o.sf, sb);
    std::unique_ptr<QueryRunner> B0, A1;
    std::vector<QueryRunner*> seq0, seq1;
    if (!alt) {
        seq0.assign(o.iters, &A0);
        seq1.assign(o.iters, &B1);
    } else {
        B0 = std::make_unique<QueryRunner>(o.qb, o.sf, sa);
        A1 = std::make_unique<QueryRunner>(o.qa, o.sf, sb);
        for (int k = 0; k < o.iters; ++k) {
            seq0.push_back((k % 2 == 0) ? &A0 : B0.get());
            seq1.push_back((k % 2 == 0) ? &B1 : A1.get());
        }
    }
    for (int w = 0; w < o.warmups; ++w) {
        std::barrier bar(2);
        BatchOut bo0, bo1;
        std::exception_ptr eb0, eb1;
        std::thread t0([&] { try { bo0 = run_batch_thread(seq0, o.device, &bar, w == 0); }
                             catch (...) { eb0 = std::current_exception(); } });
        std::thread t1([&] { try { bo1 = run_batch_thread(seq1, o.device, &bar, w == 0); }
                             catch (...) { eb1 = std::current_exception(); } });
        t0.join(); t1.join();
        if (eb0) std::rethrow_exception(eb0);
        if (eb1) std::rethrow_exception(eb1);
        if (w == 0) {
            if (bo0.keep) validate_or_die(*bo0.keep, o.qa, o.sf);
            if (bo1.keep) validate_or_die(*bo1.keep, o.qb, o.sf);
        }
    }

    std::vector<double> bm, itera, iterb;
    std::vector<RawRow> raw;
    bool stable = false;
    for (int rep = 0; rep < maxb; ++rep) {
        purger.purge();
        double ts = now_ms();
        std::barrier bar(2);
        BatchOut bo0, bo1;
        std::exception_ptr eb0, eb1;
        std::thread t0([&] { try { bo0 = run_batch_thread(seq0, o.device, &bar, false); }
                             catch (...) { eb0 = std::current_exception(); } });
        std::thread t1([&] { try { bo1 = run_batch_thread(seq1, o.device, &bar, false); }
                             catch (...) { eb1 = std::current_exception(); } });
        t0.join(); t1.join();
        if (eb0) std::rethrow_exception(eb0);
        if (eb1) std::rethrow_exception(eb1);
        double const t0start = std::min(bo0.start, bo1.start);
        double const makespan = std::max(bo0.end, bo1.end) - t0start;
        bm.push_back(makespan);
        auto collect = [&](const BatchOut& bo) {
            for (std::size_t it = 0; it < bo.iters.size(); ++it) {
                const auto& r = bo.iters[it];
                (r.qid == o.qa ? itera : iterb).push_back(r.wall);
                raw.push_back({"batch_concurrent", o.policy, o.qa, o.qb, o.sf, rep, (int)it,
                               (r.qid == o.qa ? r.wall : std::nan("")),
                               (r.qid == o.qb ? r.wall : std::nan("")),
                               std::nan(""), std::nan(""), makespan, std::nan(""),
                               nvml.sm_mhz(), nvml.throttle(), ts,
                               o.total_sms, o.total_sms});
            }
        };
        collect(bo0); collect(bo1);
        int cnt = rep + 1;
        if (!o.profile && cnt >= minb && summarize(bm, o.seed).rel_hw() <= o.ci) {
            stable = true; break;
        }
    }
    double ts = now_ms();
    Dist dBM = summarize(bm, o.seed), dIA = summarize(itera, o.seed ^ 7),
         dIB = summarize(iterb, o.seed ^ 8);
    std::vector<SummaryRow> sum;
    auto push = [&](const char* m, const Dist& d, int n) {
        sum.push_back({"batch_concurrent", o.policy, m, o.qa, o.qb, o.sf, n,
                       o.refresh_round, stable, d, ts, o.total_sms, o.total_sms});
    };
    push("BM", dBM, (int)bm.size());
    push("iter_a", dIA, (int)itera.size());
    push("iter_b", dIB, (int)iterb.size());
    write_raw(o.out + ".raw.csv", raw);
    write_summary(o.out + ".summary.csv", sum);
    std::cout << "batch-conc Q" << o.qa << "+Q" << o.qb << " [" << o.policy << "] K=" << o.iters
              << " BM=" << dBM.median << " iterA=" << dIA.median
              << " iterB=" << dIB.median << " ms\n";
    }
    cudaStreamDestroy(sa);
    cudaStreamDestroy(sb);
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    Options o = parse(argc, argv);
    ck(cudaSetDevice(o.device), "cudaSetDevice");
    int sm_count = 0;
    ck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, o.device),
       "query device SM count");
    o.total_sms = sm_count;  // recorded on native rows; green rows record actual slice
    gamax::prewarm_device_memory_resource();
    NvmlProbe nvml;
    nvml.init(o.device);
    L2Purger purger;
    purger.init(o.device);

    if (o.out.find('/') != std::string::npos)
        fs::create_directories(fs::path(o.out).parent_path());

    if (o.mode == "isolated") return run_isolated(o, nvml, purger);
    if (o.mode == "concurrent") return run_concurrent_mode(o, nvml, purger);
    if (o.mode == "batch-isolated") return run_batch_isolated(o, nvml, purger);
    if (o.mode == "batch-concurrent") return run_batch_concurrent(o, nvml, purger);
    fail("unknown mode: " + o.mode);
} catch (const std::exception& e) {
    std::cerr << "benchmark_tpch_concurrency: " << e.what() << "\n";
    return 1;
}
