/*
Code adapted from  multicore-hashjoins-0.2@https://www.systems.ethz.ch/node/334
All credit to the original author: Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
*/
#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#include <omp.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "utils/transform.hpp"
#include "memory/mem_manager.hpp"

template<typename T> using zipf_distribution_t = std::discrete_distribution<T>;
// #define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RAND_RANGE48(N,STATE) ((double)nrand48(STATE)/((double)RAND_MAX+1)*(N))
#define LOCK(L) while ((L).test_and_set()) {}

template<typename T>
void knuth_shuffle(T * keys, const gamax::size_type num_tuples, const int user_seed=42)
{
    auto locks = new std::atomic_flag[num_tuples];
    for(gamax::size_type i = 0; i < num_tuples; i++) {
        locks[i].clear();
    }

    #pragma omp parallel 
    {
        union {
            unsigned short state[3];
            unsigned int seed_;
        } seed;
        seed.seed_ = omp_get_thread_num() + user_seed;
        seed.state[2] = 0;
        
        #pragma omp for
        for (gamax::size_type i = num_tuples - 1; i > 0; i--) {
            int64_t j = RAND_RANGE48(i, seed.state);
            LOCK(locks[i]);
            LOCK(locks[j]);
            std::swap(keys[i], keys[j]);
            locks[j].clear();
            locks[i].clear();
        }
    }
}

template<typename T, typename DistT>
void create_integral_relation_nonunique(T** relation, const gamax::size_type num_tuples,
                                   bool pinned,
                                   const T vmin = std::numeric_limits<T>::min(), 
                                   const T vmax = std::numeric_limits<T>::max(),
                                   int seed = 42) {
    static_assert(std::is_same_v<T, typename DistT::result_type>);
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        DistT dist(vmin, vmax);
        
        #pragma omp for
        for(gamax::size_type i = 0; i < num_tuples; i++) {
            (*relation)[i] = dist(e);
        }
    }
}

template<typename T>
void create_integral_relation_unique(T** relation, const gamax::size_type num_tuples,
                                   bool pinned,
                                   const T start = 0,
                                   const bool shuffle = true,
                                   int seed = 42, const bool sparse = false) {
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    auto ptr = *relation;
    
    if(sparse) {
        // FIXME: this could lead to duplicates
        std::uniform_int_distribution<T> dist;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::default_random_engine e(seed+tid);
            
            #pragma omp for
            for(gamax::size_type i = 0; i < num_tuples; i++) {
                ptr[i] = dist(e);
            }
        }
    } 
    else {
        #pragma omp parallel for
        for(gamax::size_type i = 0; i < num_tuples; i++) {
            ptr[i] = i+start;
        }
        if(shuffle) {
            // std::default_random_engine e(seed);
            // std::shuffle(ptr, ptr+num_tuples, e);
            knuth_shuffle(ptr, num_tuples);
        }
    }
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
inline std::vector<double> gen_zipf_lut(double zipf_factor, gamax::size_type alphabet_size) {
    std::vector<double> lut;

	/*
	 * Compute scaling factor such that
	 *
	 *   sum (lut[i], i=1..alphabet_size) = 1.0
	 *
	 */
	// scaling_factor = 0.0;
	// for (gamax::size_type i = 1; i <= alphabet_size; i++)
	// 	scaling_factor += 1.0 / pow(i, zipf_factor);

	/*
	 * Generate the lookup table
	 */
	for (gamax::size_type i = 1; i <= alphabet_size; i++) {
		// sum += 1.0 / pow(i, zipf_factor);
        // lut.push_back(sum / scaling_factor);
        lut.push_back(1.0 / pow(i, zipf_factor));
	}

    // cout << "Generate lookup table\n";

	return lut;
}

// overload to generate zipf distribution
template<typename T>
void create_integral_relation_nonunique(T** relation, const gamax::size_type num_tuples,
                                        bool pinned,
                                        double zipf_factor, 
                                        gamax::size_type alphabet_size,
                                        int seed = 42) {
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    auto lut = gen_zipf_lut(zipf_factor, alphabet_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        zipf_distribution_t<T> dist(lut.begin(), lut.end());
        
        #pragma omp for
        for(gamax::size_type i = 0; i < num_tuples; i++) {
            (*relation)[i] = dist(e);
        }
    }
}

template<typename T>
void create_fk_from_pk_uniform(T** fk, const gamax::size_type n_fk, const T* pk, const gamax::size_type n_pk, const int seed = 42, const bool pinned = false) {
    if(pinned) {
        cudaMallocHost((void**)fk, sizeof(T)*n_fk);
    }
    else {
        *fk = new T[n_fk];
    }

    auto n_fold = n_fk / n_pk;
    auto remainder = n_fk % n_pk;

    #pragma omp parallel for
    for(gamax::size_type i = 0; i < n_fold; i++) {
        memcpy((*fk) + i*n_pk, pk, sizeof(T)*n_pk);
    }

    if(remainder != 0) {
        memcpy((*fk) + n_fold*n_pk, pk, sizeof(T)*remainder);
    }

    // std::default_random_engine e(seed);
    // std::shuffle((*fk), (*fk)+n_fk, e);
    knuth_shuffle((*fk), n_fk);
}

template<typename T>
void create_fk_from_pk_zipf(T** fk, const gamax::size_type n_fk, const T* pk, const gamax::size_type n_pk, const double zipf_factor, const int seed = 42, const bool pinned = false) {
    auto lut = gen_zipf_lut(zipf_factor, n_pk);

    if(pinned) {
        cudaMallocHost((void**)fk, sizeof(T)*n_fk);
    }
    else {
        *fk = new T[n_fk];
    }
    
    zipf_distribution_t<T> dist(lut.begin(), lut.end());
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        
        #pragma omp for
        for(gamax::size_type i = 0; i < n_fk; i++) {
            (*fk)[i] = pk[dist(e)];
        }
    }
}

template<typename T>
void create_relation_with_identical_elem(T** relation, const gamax::size_type n, T elem, const bool pinned = false) {
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*n);
    }
    else {
        *relation = new T[n];
    }

    #pragma omp parallel for
    for(gamax::size_type i = 0; i < n; i++) {
        (*relation)[i] = elem;
    }
}

template<typename T>
void adjust_selectivity(T* keys, const gamax::size_type n, const double selectivity, const T start, const int seed = 42) {
    const gamax::size_type n_change = n * (1.0 - selectivity);
    const T max = std::numeric_limits<T>::max();

    T k = start;
    for(gamax::size_type i = 0; i < n_change; i++) {
        keys[i] = start;
        k = (k < max) ? k+1 : start;
    }

    std::default_random_engine e(seed);
    std::shuffle(keys, keys+n, e);
}

namespace gamax::datagen::device {

template<typename T>
struct uniform_int_gen
{
    using result_type = T;

    T a, b;

    __host__ __device__
    uniform_int_gen(T _a, T _b) : a(_a), b(_b) {};

    __host__ __device__
    T operator()(const gamax::size_type idx) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_int_distribution<T> dist(a, b);
        rng.discard(idx);

        return dist(rng);
    }
};

template<typename Generator,
         typename T = typename Generator::result_type>
std::unique_ptr<gamax::column> generate(
    Generator gen,
    const gamax::size_type n,
    cudaStream_t stream = cudaStreamDefault) 
{
    static_assert(std::is_same_v<T, typename Generator::result_type>);
    
    auto ret = gamax::make_column<T>(n);
    auto itr = thrust::make_counting_iterator<gamax::size_type>(0);
    thrust::transform(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        itr, itr+n,
        ret->template data<T>(), // TODO: figure out why sometimes only this form works
        gen
    );

    cudaStreamSynchronize(stream);

    return ret;
}

/**
 * @brief GPU-based Zipfian distribution generator functor.
 * 
 * This functor generates Zipfian-distributed indices using inverse transform sampling.
 * It uses the CDF (Cumulative Distribution Function) and binary search (upper_bound)
 * to map uniform random numbers to Zipfian-distributed values.
 * 
 * The semantic matches std::discrete_distribution: given weights w_i, the probability
 * of selecting index i is w_i / sum(w_j).
 * 
 * @tparam T The integral type of the generated indices.
 */
template<typename T>
struct zipf_gen
{
    static_assert(std::is_integral_v<T>, "zipf_gen requires an integral type");
    using result_type = T;

    const double* cdf;           // Device pointer to CDF array
    gamax::size_type alphabet_size;  // Number of categories (size of CDF)
    int seed;

    __host__ __device__
    zipf_gen(const double* _cdf, gamax::size_type _alphabet_size, int _seed = 42) 
        : cdf(_cdf), alphabet_size(_alphabet_size), seed(_seed) {}

    __host__ __device__
    T operator()(const gamax::size_type idx) const
    {
        // Generate uniform random number in [0, 1)
        thrust::default_random_engine rng(seed);
        rng.discard(idx);
        thrust::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng);

        // Binary search (upper_bound) to find the category
        // The CDF is normalized such that cdf[alphabet_size-1] = 1.0
        gamax::size_type lo = 0;
        gamax::size_type hi = alphabet_size;
        while (lo < hi) {
            gamax::size_type mid = lo + (hi - lo) / 2;
            if (cdf[mid] <= u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is now the smallest index where cdf[lo] > u
        // Clamp to valid range [0, alphabet_size-1]
        return static_cast<T>(lo < alphabet_size ? lo : alphabet_size - 1);
    }
};

/**
 * @brief Functor to compute Zipfian PMF weights on GPU.
 * 
 * Computes w[i] = 1/(i+1)^zipf_factor for use in GPU-based CDF generation.
 */
struct zipf_pmf_functor {
    double zipf_factor;
    
    __host__ __device__
    zipf_pmf_functor(double _zipf_factor) : zipf_factor(_zipf_factor) {}
    
    __host__ __device__
    double operator()(const gamax::size_type i) const {
        return 1.0 / pow(static_cast<double>(i + 1), zipf_factor);
    }
};

/**
 * @brief Functor to normalize CDF values by dividing by total sum.
 */
struct normalize_cdf_functor {
    double total_sum;
    
    __host__ __device__
    normalize_cdf_functor(double _total_sum) : total_sum(_total_sum) {}
    
    __host__ __device__
    double operator()(const double val) const {
        return val / total_sum;
    }
};

/**
 * @brief Compute the CDF for Zipfian distribution entirely on GPU.
 * 
 * Given a zipf_factor and alphabet_size, this function computes the probability
 * mass function (PMF) and then the CDF using GPU-accelerated operations:
 * 1. Compute PMF weights: w[i] = 1/(i+1)^zipf_factor using thrust::transform
 * 2. Compute inclusive prefix sum to get unnormalized CDF
 * 3. Get total sum from last element
 * 4. Normalize by dividing all values by total sum
 * 
 * The PMF is defined as:
 *   P(i) = (1 / (i+1)^zipf_factor) / sum_j(1 / (j+1)^zipf_factor)
 * 
 * @param zipf_factor The Zipfian skew parameter (s). Higher values = more skew.
 * @param alphabet_size The number of distinct values (categories).
 * @param stream CUDA stream for async execution.
 * @return std::unique_ptr<gamax::column> Device column containing CDF where cdf[i] = P(X <= i).
 */
inline std::unique_ptr<gamax::column> gen_zipf_cdf_device(
    double zipf_factor, 
    gamax::size_type alphabet_size,
    cudaStream_t stream = cudaStreamDefault) 
{
    // Allocate device memory for CDF
    auto d_cdf = gamax::make_column<double>(alphabet_size, false, stream);
    
    // Step 1: Compute PMF weights w[i] = 1/(i+1)^zipf_factor
    auto count_iter = thrust::make_counting_iterator<gamax::size_type>(0);
    thrust::transform(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        count_iter,
        count_iter + alphabet_size,
        d_cdf->data<double>(),
        zipf_pmf_functor(zipf_factor)
    );
    
    // Step 2: Compute inclusive prefix sum (unnormalized CDF)
    thrust::inclusive_scan(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        d_cdf->data<double>(),
        d_cdf->data<double>() + alphabet_size,
        d_cdf->data<double>()
    );
    
    // Step 3: Get total sum (last element of prefix sum)
    double total_sum;
    cudaMemcpyAsync(&total_sum, d_cdf->data<double>() + alphabet_size - 1, 
                    sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Step 4: Normalize CDF by dividing by total sum
    thrust::transform(
        thrust::cuda::par(*get_thrust_allocator()).on(stream),
        d_cdf->data<double>(),
        d_cdf->data<double>() + alphabet_size,
        d_cdf->data<double>(),
        normalize_cdf_functor(total_sum)
    );
    
    // Ensure the last value is exactly 1.0 to avoid floating-point issues
    double one = 1.0;
    cudaMemcpyAsync(d_cdf->data<double>() + alphabet_size - 1, &one, 
                    sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    return d_cdf;
}

/**
 * @brief Generate a column with Zipfian-distributed indices on the GPU.
 * 
 * This function generates n random indices following a Zipfian distribution
 * with the given zipf_factor and alphabet_size. The generated values are
 * in the range [0, alphabet_size-1].
 * 
 * The implementation is fully GPU-based:
 * 1. Computes the CDF on the GPU using gen_zipf_cdf_device()
 * 2. Uses thrust::transform with the zipf_gen functor to generate values
 * 
 * The semantic is equivalent to the host-based create_integral_relation_nonunique
 * with zipf distribution: each element is independently sampled from the
 * Zipfian distribution.
 * 
 * @tparam T The integral type of the generated indices (default: gamax::size_type).
 * @param n Number of elements to generate.
 * @param zipf_factor The Zipfian skew parameter (s).
 * @param alphabet_size The number of distinct values.
 * @param seed Random seed for reproducibility.
 * @param stream CUDA stream for async execution.
 * @return std::unique_ptr<gamax::column> Column containing n Zipfian-distributed indices.
 */
template<typename T = gamax::size_type>
std::unique_ptr<gamax::column> generate_zipf(
    const gamax::size_type n,
    const double zipf_factor,
    const gamax::size_type alphabet_size,
    const int seed = 42,
    cudaStream_t stream = cudaStreamDefault)
{
    static_assert(std::is_integral_v<T>, "generate_zipf requires an integral type");
    
    // Compute CDF entirely on GPU
    auto d_cdf = gen_zipf_cdf_device(zipf_factor, alphabet_size, stream);
    
    // Generate Zipfian-distributed indices using the functor
    auto gen = zipf_gen<T>(d_cdf->data<double>(), alphabet_size, seed);
    return generate(gen, n, stream);
}

} // namespace gamax::datagen::device