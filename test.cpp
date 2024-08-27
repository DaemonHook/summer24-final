#include <iostream>
#include <functional>

/// @brief 测量cpu函数运行时间 (ms)
/// @return 如果函数返回 void，那么结果类型为 std::tuple<double>，否则为 std::tuple<double, Ret>
template <typename Func, typename... Args>
auto allocateAndCall(Func func, Args&&... args) {
    
    // 定义返回类型
    using ResultType = decltype(func(std::forward<Args>(args)...));
    using ReturnType = std::conditional_t<std::is_void_v<ResultType>, std::tuple<double>, std::tuple<double, ResultType>>;
    
    if constexpr (std::is_void_v<ResultType>) {
        // 如果函数返回 void，直接调用
        func(std::forward<Args>(args)...);
    } else {
        // 如果函数有返回值，获取函数返回值
        auto result = func(std::forward<Args>(args)...);
        return result;
    }
}

// 示例函数
int exampleFunction(int x, int y) {
    return x + y;
}

double anotherFunction(int x, int y) {
    return static_cast<double>(x) / y;
}

int main() {
    // 创建std::function对象，并传入具体的函数
    std::function<int(int, int)> func1 = exampleFunction;
    std::function<double(int, int)> func2 = anotherFunction;

    // 调用allocateAndCall
    int result1 = allocateAndCall(exampleFunction, 5, 3);
    double result2 = allocateAndCall(anotherFunction, 10, 2);
    
    std::cout << "Result of exampleFunction: " << result1 << std::endl;
    std::cout << "Result of anotherFunction: " << result2 << std::endl;

    return 0;
}
