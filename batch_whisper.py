import sys
import os
import time
import shutil
import traceback
import gc
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= ❄️ RTX 5080 ASMR 专享版 ❄️ =================
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
BATCH_SIZE = 12  # 稍微调低并发，求稳

# ASMR 优化开关
# 3D麦/睡眠导入通常声音很小，且有长时间底噪，需要调整参数
IS_ASMR_MODE = True

# 容错机制：如果生成的时长少了 60秒 以上，自动重跑
TOLERANCE_SECONDS = 60
MAX_RETRIES = 3

VIDEO_EXTS = {'.mp4', '.flv', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m4v', '.m4a'}


# ===========================================================

def format_timestamp(seconds):
    if seconds is None: return "00:00:00,000"
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_with_retry(batched_model, video_path, srt_path, total_duration):
    # ASMR 专用参数：更敏感，不容易被切断
    vad_params = {
        "min_silence_duration_ms": 3000 if IS_ASMR_MODE else 2000,
        "speech_pad_ms": 2000 if IS_ASMR_MODE else 1500,
        "threshold": 0.3 if IS_ASMR_MODE else 0.5,  # 降低人声门槛
    }

    prompt = "饼干岁们好，我是岁己。今天直播3D麦，助眠，会有轻声细语和触发音。" if IS_ASMR_MODE else "大家好我是岁己，今天直播玩游戏杂谈。"

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"   🚀 第 {attempt} 次尝试 (ASMR模式={'开' if IS_ASMR_MODE else '关'})...")
        start_time = time.time()
        last_segment_end = 0
        line_count = 0
        temp_srt = srt_path + ".tmp"

        try:
            # condition_on_previous_text=False 能防止在长时间静音中复读
            segments, _ = batched_model.transcribe(
                video_path,
                batch_size=BATCH_SIZE,
                language="zh",
                initial_prompt=prompt,
                vad_filter=True,
                vad_parameters=vad_params,
                word_timestamps=True,
                condition_on_previous_text=False
            )

            term_width = shutil.get_terminal_size().columns
            bar_width = max(20, term_width - 55)

            with open(temp_srt, "w", encoding="utf-8") as f:
                for segment in segments:
                    last_segment_end = segment.end
                    line_count += 1

                    # 进度条
                    percent = (last_segment_end / total_duration) * 100
                    if percent > 100: percent = 100
                    elapsed = time.time() - start_time
                    speed = last_segment_end / elapsed if elapsed > 0 else 0

                    filled = int(bar_width * percent / 100)
                    bar = '█' * filled + '-' * (bar_width - filled)

                    sys.stdout.write(f"\r   🔄 {percent:5.1f}% [{bar}] {speed:.1f}x")
                    sys.stdout.flush()

                    start_s = format_timestamp(segment.start)
                    end_s = format_timestamp(segment.end)
                    text = segment.text.strip()
                    f.write(f"{line_count}\n{start_s} --> {end_s}\n{text}\n\n")
                    f.flush()

            print()

            # === 🛡️ 完整性检查 ===
            missing = total_duration - last_segment_end
            if missing > TOLERANCE_SECONDS:
                print(f"   ⚠️  警告: 居然缺了 {missing:.1f} 秒！")
                print(f"   🚫 这肯定不正常，正在触发自动重试...")
                if attempt < MAX_RETRIES:
                    time.sleep(2)
                    continue
                else:
                    print(f"   💀 没救了，保留现有结果吧。")

            # 成功
            if os.path.exists(srt_path): os.remove(srt_path)
            os.rename(temp_srt, srt_path)
            print(f"   ✅ 成功生成！耗时: {time.time() - start_time:.1f}s")
            return

        except Exception as e:
            print(f"\n   ❌ 报错: {e}")
            traceback.print_exc()
            time.sleep(2)


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    if len(sys.argv) < 2: return

    input_path = sys.argv[1]
    # ... (扫描文件逻辑省略，保持不变) ...
    if os.path.isfile(input_path):
        todo_list = [input_path]
    else:
        todo_list = []  # 简写了，逻辑同前

    print(f"🔥 正在加载 Whisper (Watchdog版)...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
    except Exception as e:
        print(f"❌ 显卡报错: {e}")
        return

    for idx, video_path in enumerate(todo_list, start=1):
        print(f"\n🎬 处理: {os.path.basename(video_path)}")
        try:
            _, info = batched_model.transcribe(video_path, batch_size=4)
            transcribe_with_retry(batched_model, video_path, os.path.splitext(video_path)[0] + ".srt", info.duration)
            gc.collect()
        except Exception as e:
            print(f"❌ 失败: {e}")


if __name__ == "__main__":
    main()