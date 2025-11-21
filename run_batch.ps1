# ==========================================
#  daiFish 批量字幕生成器 - 夜间取暖版
# ==========================================
param (
    [string]$Path
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ScriptDir = $PSScriptRoot
$PythonScript = Join-Path $ScriptDir "batch_whisper.py"

Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   daiFish 批量字幕生成器 (RTX 5080 取暖专用)   " -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ([string]::IsNullOrWhiteSpace($Path)) {
    Write-Host "❌ 错误：请拖入一个【文件夹】！" -ForegroundColor Red
    Read-Host "按回车键退出..."
    exit
}

if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ 错误：找不到 batch_whisper.py" -ForegroundColor Red
    Read-Host "按回车键退出..."
    exit
}

Write-Host "📂 目标路径: $Path" -ForegroundColor Cyan
Write-Host "🚀 正在准备批量处理..." -ForegroundColor Cyan
Write-Host ""

try {
    python $PythonScript "$Path"
} catch {
    Write-Host "❌ 系统错误: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🔥 取暖结束，记得关机！" -ForegroundColor Yellow
Write-Host ""
Read-Host "按回车键关闭窗口..."
```

### 3. 拖拽入口 (`批量拖拽到我身上.bat`)

这个不需要变，还是那行代码，指向新的 `run_batch.ps1` 即可：

```bat
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_batch.ps1" "%~1"