import sys
import os
import time
import shutil
import traceback
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= ❄️ RTX 5080 取暖配置 ❄️ =================
# 模型：Turbo (速度快，精度高，适合批量)
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# 显存够大，Batch Size 设为 16 或 24 均可
BATCH_SIZE = 16

# 支持的视频后缀 (大小写均可)
VIDEO_EXTS = {'.mp4', '.flv', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m4v'}
# ===========================================================

def is_video_file(filename):
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTS

def format_timestamp(seconds):
    if seconds is None: return "00:00:00,000"
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def process_one_video(model, batched_model, video_path, file_idx, total_files):
    filename = os.path.basename(video_path)
    output_dir = os.path.dirname(video_path)
    filename_no_ext = os.path.splitext(filename)[0]
    srt_path = os.path.join(output_dir, filename_no_ext + ".srt")

    # --- 智能跳过逻辑 ---
    if os.path.exists(srt_path):
        print(f"⏭️  [跳过] 已存在字幕: {filename}")
        return
    # ------------------

    print(f"\n🎬 [{file_idx}/{total_files}] 正在处理: {filename}")
    
    try:
        # VAD 参数配置
        vad_params = {
            "min_silence_duration_ms": 2000, 
            "speech_pad_ms": 1500,           
        }

        # 1. 快速分析时长
        print("   🔍 分析视频时长...", end="", flush=True)
        # 这里为了快，batch_size 用小一点探测即可，但用 batched_model 也行
        _, info = batched_model.transcribe(video_path, batch_size=BATCH_SIZE)
        total_duration = info.duration
        print(f" -> {format_timestamp(total_duration)}")

        # 2. 开始转写
        start_time = time.time()
        
        segments, _ = batched_model.transcribe(
            video_path, 
            batch_size=BATCH_SIZE,
            language="zh",
            initial_prompt="以下是二次元虚拟主播直播录像，主要用简体中文。",
            vad_filter=True,            
            vad_parameters=vad_params   
        )

        # 准备进度条
        term_width = shutil.get_terminal_size().columns
        bar_width = max(20, term_width - 50) 

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                current_time = segment.end
                percent = (current_time / total_duration) * 100
                if percent > 100: percent = 100
                
                elapsed = time.time() - start_time
                speed = current_time / elapsed if elapsed > 0 else 0 
                eta = (total_duration - current_time) / speed if speed > 0 else 0
                
                filled_len = int(bar_width * percent / 100)
                bar = '█' * filled_len + '-' * (bar_width - filled_len)
                
                # 进度条显示
                sys.stdout.write(f"\r   🚀 {percent:5.1f}% [{bar}] ETA:{int(eta)}s | {speed:.0f}x")
                sys.stdout.flush()

                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                text = segment.text.strip()
                f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
                
                if i % 10 == 0: f.flush() 

        total_time = time.time() - start_time
        print(f"\n   ✅ 完成！耗时: {total_time:.1f}s")

    except Exception as e:
        print(f"\n   ❌ 处理失败: {filename}")
        print(f"   错误信息: {e}")
        # 不抛出异常，为了让循环继续处理下一个文件

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if len(sys.argv) < 2:
        print("❌ 请把【文件夹】拖拽到 .bat 图标上！")
        return

    input_path = sys.argv[1]
    
    # 1. 扫描文件列表
    todo_list = []
    print(f"📂 正在扫描目录: {input_path}")
    
    if os.path.isfile(input_path):
        # 如果拖入的是单个文件
        if is_video_file(input_path):
            todo_list.append(input_path)
    else:
        # 如果拖入的是目录 (递归扫描)
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if is_video_file(file):
                    full_path = os.path.join(root, file)
                    todo_list.append(full_path)

    total_files = len(todo_list)
    if total_files == 0:
        print("⚠️  该目录下没有找到视频文件。")
        return

    print(f"📋 共找到 {total_files} 个视频文件。")
    print("=" * 60)

    # 2. 初始化模型 (只加载一次，极大节省时间)
    print(f"⏳ 正在预热 RTX 5080 ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
        print("🔥 引擎已就绪，取暖模式启动！")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 循环处理
    start_all = time.time()
    
    for idx, video_path in enumerate(todo_list, start=1):
        process_one_video(model, batched_model, video_path, idx, total_files)
    
    end_all = time.time()
    duration = end_all - start_all
    
    print("\n" + "=" * 60)
    print(f"🏆 所有任务全部完成！")
    print(f"⏱️  总耗时: {int(duration//3600)}小时 {int((duration%3600)//60)}分")
    print("🛌 祝你好梦！")

if __name__ == "__main__":
    main()