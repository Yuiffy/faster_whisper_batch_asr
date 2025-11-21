@echo off
:: 使用 PowerShell 运行同目录下的 run.ps1，并传入拖拽的文件路径
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_batch.ps1" "%~1"