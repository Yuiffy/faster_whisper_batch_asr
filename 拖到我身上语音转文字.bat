@echo off
:: 切换到当前脚本所在的目录，防止路径问题
cd /d "%~dp0"

:: 检查是否拖入了文件
if "%~1"=="" (
    echo 请直接把视频文件(mp4/flv/mkv)拖拽到这个图标上！
    pause
    exit
)

:: 调用 Python 脚本，"%~1" 代表拖入的文件的绝对路径
python fast_sub_batch_pro.py "%~1"

:: 运行完暂停，让你看到结果，而不是直接闪退
echo.
pause