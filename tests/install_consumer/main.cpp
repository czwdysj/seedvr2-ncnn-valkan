// 本文件是安装包的最小外部消费者。
// 它只依赖公开 Engine 头文件和导出的 CMake target，用于在 CI 中验证下游项目
// 无需了解 NCNN、自定义层或静态库的内部链接细节即可完成编译和运行。
#include "seedvr2/engine.h"

#include <cstring>

int main()
{
    seedvr2::Video video;
    if (video.valid())
        return 1;

    return std::strcmp(seedvr2::status_message(seedvr2::Status::Ok), "ok") == 0 ? 0 : 2;
}
