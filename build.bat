@echo off
rem 一键构建 seedvr2-ncnn-vulkan（Windows，需要 Visual Studio 2022 + Vulkan SDK + git）。
rem 产物：build\Release\seedvr2-ncnn-vulkan.exe
rem
rem 注意：ncnn submodule 需要保持 LF 换行以正确应用 patches，
rem       如遇 patch 应用失败，请先执行：
rem       git -C ncnn config core.autocrlf false && git -C ncnn checkout -- .
cd /d "%~dp0"

if not exist ncnn\CMakeLists.txt (
    echo [build] initializing ncnn submodule ...
    git submodule update --init --recursive || goto :fail
)

if not exist ncnn\src\layer\vulkan\convolution3d_vulkan.cpp (
    echo [build] applying ncnn patches ...
    for %%p in (patches\*.patch) do (
        git -C ncnn apply --check %%p
        if errorlevel 1 (
            echo [build] patch does not apply: %%p
            echo [build] hint: git -C ncnn config core.autocrlf false ^&^& git -C ncnn checkout -- .
            goto :fail
        )
        git -C ncnn apply %%p
    )
) else (
    echo [build] ncnn patches already applied, skipping.
)

echo [build] configuring ...
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEEDVR2_ENABLE_VULKAN=ON || goto :fail
echo [build] compiling ...
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% || goto :fail

echo.
echo [build] done -^> build\Release\seedvr2-ncnn-vulkan.exe
echo [build] next: download-models.bat / download-models.sh, then:
echo [build]   build\Release\seedvr2-ncnn-vulkan.exe -i in.mp4 -o out.mp4
exit /b 0

:fail
echo [build] FAILED
exit /b 1
