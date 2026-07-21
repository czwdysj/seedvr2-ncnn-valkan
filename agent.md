# Agent Coding Rules

This file defines the coding rules that the agent must read and follow before making code changes in this project.

## Project Goal

- The final goal is to port SeedVR2 inference to NCNN so this project can run the SeedVR2 model with the NCNN Vulkan backend.
- The project should be comparable in spirit to `zimage-ncnn-vulkan`: a practical C++/NCNN/Vulkan inference project with custom support for SeedVR2-specific operators.
- The most important hard parts are adaptive/window attention export and implementation, Vulkan acceleration, pnnx export boundaries, and custom operator implementation.
- The codebase is also intended for interview presentation, so every completed task must include a clear explanation of what was done, why it was done, how it was verified, and how it moves the project toward NCNN/Vulkan SeedVR2 inference.

## General Workflow

- Before changing code, inspect the relevant source files, configuration, and existing project patterns.
- Keep changes scoped to the requested task.
- Do not rewrite unrelated files or reformat unrelated code.
- Preserve the existing directory layout unless the task explicitly requires new structure.
- At the end of every completed task, tell the user the recommended next-step plan.
- During implementation, explain important actions in enough detail that the user can reuse the reasoning in an interview.

## Testing Requirement

- Every code change must be tested by the agent before it is considered complete.
- Choose tests that match the changed area:
  - For Python conversion/export scripts, run the script or a focused dry run with representative inputs.
  - For C++ runtime code, build the affected target and run the smallest available inference or unit-level check.
  - For custom ncnn layers, verify numerical output against a PyTorch reference tensor when possible.
  - For model conversion changes, compare intermediate tensors, not only final images or videos.
- If a full test is too expensive, run the strongest practical smaller test and clearly state what was and was not verified.
- Do not report work as finished until the relevant test passes.

## Comments

- Important code must include clear, concise comments explaining why the code exists, what tensor/model behavior it implements, and why a non-obvious implementation choice is necessary.
- 本项目中，agent 新增或重写的文件级介绍和代码注释必须使用中文。
- Add comments for:
  - Tensor shape transformations.
  - Model partition boundaries.
  - Custom ncnn layers and Vulkan-specific behavior.
  - Numerical approximations, precision changes, or layout conversions.
  - Any workaround for pnnx, ncnn, PyTorch tracing, or unsupported operators.
- Avoid comments that simply restate the code.

## File Headers

- Every new code file must start with a clear file-level introduction.
- 文件级介绍必须使用中文，并且要详细到能在面试中说明该文件的职责、输入输出和运行假设。
- The header must explain:
  - What the file is responsible for.
  - The main inputs and outputs when applicable.
  - Any important dependency or runtime assumption.
- For existing code files that are substantially edited, add or update the file-level introduction if it is missing.

## Public API Architecture

- Prefer a thin API with a thick implementation.
- External users should see a simple `Engine`-style interface for loading models, configuring runtime options, and running video restoration.
- Hide implementation details such as VAE, DiT, adaptive window attention, text embedding handling, sampler steps, custom operators, and NCNN/Vulkan resource management behind internal modules.
- Keep internal architecture clear by separating video I/O, execution planning, model components, tensor/layout utilities, custom operators, and NCNN/Vulkan runtime code.
- Do not expose low-level SeedVR2 internals through the public interface unless there is a concrete debugging or benchmarking need.

## SeedVR2 NCNN Porting Notes

- Treat the PyTorch implementation as the numerical reference.
- Keep the first working target narrow: SeedVR2 3B, super-resolution task, precomputed text embeddings, batch size 1, and fixed test shapes.
- Prefer explicit tensor layout names in code and comments, such as `T,H,W,C`, `B,C,T,H,W`, or flattened `L,C`.
- Do not assume pnnx can convert the full model. Use pnnx for ordinary subgraphs and weights, and implement unsupported dynamic behavior explicitly.
- Pay special attention to adaptive window attention, RoPE, causal 3D convolution, slicing state, and classifier-free guidance.
