# NCNN Vulkan 后端入门：ReLU

本目录用 NCNN 自带的 `ReLU_vulkan` 作为最小教学案例，解释一个 Vulkan layer 从创建到执行的完整过程。示例输入和输出都是同一个 `VkMat`，计算规则为：

```text
slope == 0: y = max(x, 0)
slope != 0: y = x（当 x >= 0），否则 y = x * slope
```

这里采用原项目已有的目录名 `learn_ncnn_valkan`，但图形 API 的正确拼写是 **Vulkan**。

## 文件职责

- `relu_vulkan_annotated.h`：声明 Vulkan layer、生命周期函数和持有的 Pipeline。
- `relu_vulkan_annotated.cpp`：在 CPU 侧创建 Pipeline、准备资源绑定和记录 GPU dispatch。
- `relu_annotated.comp`：GPU 实际执行的 GLSL compute shader。

这些文件是教学副本，不参与 SeedVR2 当前的 CMake 构建。对应的 NCNN 原始实现位于：

```text
/home/czw1/ncnn_learn/ncnn/src/layer/vulkan/relu_vulkan.h
/home/czw1/ncnn_learn/ncnn/src/layer/vulkan/relu_vulkan.cpp
/home/czw1/ncnn_learn/ncnn/src/layer/vulkan/shader/relu.comp
```

## 完整执行流程

```mermaid
flowchart LR
    A["模型加载"] --> B["ReLU_vulkan 构造函数"]
    B --> C["create_pipeline"]
    C --> D["编译或取得 SPIR-V"]
    D --> E["创建 VkPipeline"]
    E --> F["forward_inplace"]
    F --> G["绑定 VkMat 缓冲区"]
    G --> H["写入 push constant: n"]
    H --> I["record_pipeline"]
    I --> J["GPU 执行 relu.comp"]
    J --> K["destroy_pipeline"]
```

## 三类参数

### Specialization constant

`slope` 在 `create_pipeline()` 时写入。它通常在模型加载后不再变化，Vulkan 驱动可以针对该值优化 shader。

### Push constant

`n` 在每次 `forward_inplace()` 时写入，因为输入元素数量可能随动态形状变化。它适合少量、频繁变化的运行时参数。

### Storage buffer binding

`binding = 0` 对应 `bottom_top_blob`。这个 `VkMat` 同时作为输入和输出，因此 ReLU 不需要额外申请输出显存。

## 为什么一次处理四个值

Shader 使用 `vec4`，一个 invocation 处理四个标量。C++ 中：

```cpp
n = bottom_top_blob.total() * bottom_top_blob.elempack / 4;
```

`total() * elempack` 是实际标量总数，除以四得到 shader 需要处理的 `vec4` 数量。NCNN 的 Vulkan packing 机制会保证这里的内存布局和 shader 访问方式匹配。

## 与六个 SeedVR2 自定义层的关系

六个自定义层也遵循同样的骨架：

1. 构造函数声明 `support_vulkan = true`。
2. `create_pipeline()` 创建一个或多个 Pipeline。
3. `upload_model()` 上传权重层的常量张量；ReLU 没有权重，所以本例不需要该函数。
4. Vulkan `forward()`/`forward_inplace()` 分配输出、组织 bindings/constants 并记录命令。
5. `.comp` shader 完成真正的并行计算。
6. `destroy_pipeline()` 释放 Pipeline。

例如 `DynamicFramewiseGroupNorm` 比 ReLU 多两类工作：上传 weight/bias，以及用两阶段 reduction 计算每帧每组的均值和方差。但 Pipeline 的创建、绑定和 dispatch 方式仍与本例一致。

## 阅读顺序

建议依次阅读：

1. `relu_vulkan_annotated.h`
2. `relu_vulkan_annotated.cpp` 的构造函数
3. `create_pipeline()`
4. `forward_inplace()`
5. `relu_annotated.comp`
6. `destroy_pipeline()`

注意：`forward_inplace()` 只是**记录命令**，通常不会在函数返回前等待 GPU 完成。命令提交、同步和必要的数据下载由 NCNN 的 `VkCompute`/`Extractor` 调度层统一处理。
