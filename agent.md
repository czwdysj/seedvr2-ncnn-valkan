# Agent Coding Rules

This file defines the coding rules that the agent must read and follow before making code changes in this project.

## General Workflow

- Before changing code, inspect the relevant source files, configuration, and existing project patterns.
- Keep changes scoped to the requested task.
- Do not rewrite unrelated files or reformat unrelated code.
- Preserve the existing directory layout unless the task explicitly requires new structure.

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

- Important code must include concise comments explaining why the code exists or why a non-obvious implementation choice is necessary.
- Add comments for:
  - Tensor shape transformations.
  - Model partition boundaries.
  - Custom ncnn layers and Vulkan-specific behavior.
  - Numerical approximations, precision changes, or layout conversions.
  - Any workaround for pnnx, ncnn, PyTorch tracing, or unsupported operators.
- Avoid comments that simply restate the code.

## File Headers

- Every new code file must start with a short file-level introduction.
- The header must explain:
  - What the file is responsible for.
  - The main inputs and outputs when applicable.
  - Any important dependency or runtime assumption.
- For existing code files that are substantially edited, add or update the file-level introduction if it is missing.

## SeedVR2 NCNN Porting Notes

- Treat the PyTorch implementation as the numerical reference.
- Keep the first working target narrow: SeedVR2 3B, super-resolution task, precomputed text embeddings, batch size 1, and fixed test shapes.
- Prefer explicit tensor layout names in code and comments, such as `T,H,W,C`, `B,C,T,H,W`, or flattened `L,C`.
- Do not assume pnnx can convert the full model. Use pnnx for ordinary subgraphs and weights, and implement unsupported dynamic behavior explicitly.
- Pay special attention to adaptive window attention, RoPE, causal 3D convolution, slicing state, and classifier-free guidance.
