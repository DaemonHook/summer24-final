#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <type_traits>

/// @brief 测量cpu函数运行时间 (ms)
/// @return 如果函数返回 void，那么结果类型为 std::tuple<double>，否则为 std::tuple<double, Ret>
template <typename Func, typename... Args>
auto measureCPU(Func func, Args&&... args)
{
    using namespace std::chrono;

    // 定义返回类型
    using ResultType = decltype(func(std::forward<Args>(args)...));
    using ReturnType = std::conditional_t<std::is_void_v<ResultType>, std::tuple<double>, std::tuple<double, ResultType>>;

    // 记录开始时间
    auto start = high_resolution_clock::now();

    if constexpr (std::is_void_v<ResultType>) {
        // 如果函数返回 void，直接调用
        func(std::forward<Args>(args)...);
        // 记录结束时间
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        return std::make_tuple(duration);
    } else {
        // 如果函数有返回值，获取函数返回值
        auto result = func(std::forward<Args>(args)...);
        // 记录结束时间
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        return std::make_tuple(duration, result);
    }
}

/// @brief 测量gpu函数运行时间 (ms)
/// @return 如果函数返回 void，那么结果类型为 std::tuple<double>，否则为 std::tuple<double, ReturnType>
template <typename Func, typename... Args>
auto measureGPU(Func func, Args&&... args)
{
    cudaEvent_t start, stop;
    double duration;

    checkError(cudaEventCreate(&start));
    checkError(cudaEventCreate(&stop));

    cudaEventRecord(start);
    checkError(cudaMalloc(&d_cost, sizeOfVertices));

    // 定义返回类型
    using ResultType = decltype(func(std::forward<Args>(args)...));
    using ReturnType = std::conditional_t<std::is_void_v<ResultType>, std::tuple<double>, std::tuple<double, ResultType>>;

    if constexpr (std::is_void_v<ResultType>) {
        // 如果函数返回 void，直接调用
        func(std::forward<Args>(args)...);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&duration, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 记录结束时间
        return std::make_tuple(duration);
    } else {
        auto result = func(std::forward<Args>(args)...);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&duration, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // 记录结束时间
        return std::make_tuple(duration, result);
    }
}
