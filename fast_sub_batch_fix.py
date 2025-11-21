import sys
import os
import time
import shutil
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= 性能与精度配置 =================
# 推荐 small，既快又准
#MODEL_SIZE = "small" 
# 5080 显存够大，维持高并发
#BATCH_SIZE = 32  


# 使用 HuggingFace 上的转换版 Turbo 模型
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Batch Size 保持适中
BATCH_SIZE = 16
# ===============================================

def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    if len(sys.argv) < 2:
        print("❌ 请直接拖拽视频文件到 .bat 上")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"❌ 找不到文件: {video_path}")
        return

    print(f"📂 目标文件: {os.path.basename(video_path)}")

    try:
        # 1. 初始化
        print(f"⏳ 正在预热引擎 ({MODEL_SIZE})...")
        # device="cuda" 强制使用显卡
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        
        # 使用 Batch Pipeline
        batched_model = BatchedInferencePipeline(model=model)

        # 2. 准备 VAD 参数（解决烂尾的关键）
        print("🔧 配置 VAD 参数以防止吞字...")
        # 这些参数告诉模型：不要轻易丢弃结尾的声音
        vad_params = {
            "min_silence_duration_ms": 2000, # 必须要静音超过2秒才算静音（之前可能默认是0.5秒）
            "speech_pad_ms": 1500,           # 在人声前后强行多保留 1.5 秒的音频，防止掐头去尾
        }

        # 3. 预处理
        print("🔍 分析音频流...")
        # 获取时长
        dummy_gen, info = batched_model.transcribe(video_path, batch_size=BATCH_SIZE)
        total_duration = info.duration
        
        print(f"✅ 视频总长: {format_timestamp(total_duration)}")
        print("🚀 竞速模式启动 (已开启防烂尾补丁)...")
        print("=" * 50)

        output_dir = os.path.dirname(video_path)
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, filename_no_ext + ".srt")
        
        start_time = time.time()
        
        # 4. 开始转写 (带上 vad_parameters)
        segments, _ = batched_model.transcribe(
            video_path, 
            batch_size=BATCH_SIZE,
            language="zh",
            initial_prompt="以下是二次元虚拟主播直播录像，主要用简体中文。",
            vad_filter=True,            # 开启 VAD
            vad_parameters=vad_params   # 注入我们的宽松参数
        )

        # 准备进度条
        term_width = shutil.get_terminal_size().columns
        bar_width = max(20, term_width - 40) 

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
                
                sys.stdout.write(f"\r[{bar}] {percent:5.1f}% | ETA: {int(eta)}s | 倍速: {speed:.0f}x")
                sys.stdout.flush()

                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                text = segment.text.strip()
                f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
                
                if i % 10 == 0: f.flush() 

        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"🏆 任务完成！")
        print(f"⏱️  耗时: {total_time:.2f}秒 ({total_duration/total_time:.1f}倍速)")
        print(f"💾 字幕: {srt_path}")

    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

def format_timestamp(seconds):
    if seconds is None: return "00:00:00,000"
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    main()