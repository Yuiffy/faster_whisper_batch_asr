# ==========================================
#  DDTV Whisper 字幕生成启动器 (PowerShell版)
# ==========================================
param (
    [string]$VideoPath
)

# 设置控制台编码为 UTF-8，防止乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 获取当前脚本所在的文件夹路径
$ScriptDir = $PSScriptRoot
$PythonScript = Join-Path $ScriptDir "fast_sub_batch_fix.py"

Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   DDTV 自动字幕生成工具 (RTX 5080 尊享版)   " -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查是否有文件传入
if ([string]::IsNullOrWhiteSpace($VideoPath)) {
    Write-Host "❌ 错误：你没有拖入任何视频文件！" -ForegroundColor Red
    Write-Host "请直接把 mp4/flv 文件拖拽到图标上运行。" -ForegroundColor Gray
    Write-Host ""
    Read-Host "按回车键退出..."
    exit
}

# 2. 检查 Python 脚本是否存在
if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ 错误：找不到核心脚本: $PythonScript" -ForegroundColor Red
    Write-Host "请确保 run.ps1 和 fast_sub_batch_fix.py 在同一个文件夹里！" -ForegroundColor Gray
    Read-Host "按回车键退出..."
    exit
}

# 3. 检查 Python 环境
try {
    $pyVersion = python --version 2>&1
    Write-Host "✅ 检测到 Python 环境: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误：系统找不到 'python' 命令！" -ForegroundColor Red
    Write-Host "可能是环境变量没配好，或者你用的是 Conda 但没激活。" -ForegroundColor Yellow
    Read-Host "按回车键退出..."
    exit
}

# 4. 开始运行 Python 脚本
Write-Host "📂 正在处理文件: $VideoPath" -ForegroundColor Cyan
Write-Host "🚀 正在启动 Whisper 引擎..." -ForegroundColor Cyan
Write-Host ""

# 调用 Python，并实时显示输出
# $LastExitCode 记录了脚本是否报错退出
try {
    python $PythonScript "$VideoPath"
} catch {
    Write-Host "❌ 发生未知系统错误: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan

if ($LastExitCode -eq 0) {
    Write-Host "✨ 任务圆满结束！" -ForegroundColor Green
} else {
    Write-Host "⚠️  脚本似乎遇到了错误退出 (代码: $LastExitCode)" -ForegroundColor Red
    Write-Host "请向上滚动查看 Python 的报错信息。" -ForegroundColor Yellow
}

# 5. 暂停窗口，防止一闪而过
Write-Host ""
Read-Host "按回车键关闭窗口..."