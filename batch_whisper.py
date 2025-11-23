import sys
import os
import time
import shutil
import traceback
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= ❄️ RTX 5080 取暖配置 ❄️ =================
# 模型：Turbo (速度快，精度高)
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# 显存够大，Batch Size 设为 16
BATCH_SIZE = 16

# 【功能开关】是否开启长句智能切分
# True: 超过 MAX_CHARS_PER_LINE 字自动切断（适合直接看视频字幕）
# False: 保持原句不切断（适合搜索关键词、做文档归档）
ENABLE_SMART_SPLIT = False

# 单行最大字数限制 (仅当 ENABLE_SMART_SPLIT = True 时生效)
MAX_CHARS_PER_LINE = 18

# 支持的视频后缀
VIDEO_EXTS = {'.mp4', '.flv', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m4v', '.m4a'}


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


# --- ✂️ 智能切分算法 ✂️ ---
def smart_split_segment(segment, max_chars=18):
    """
    如果一句话太长，利用单词时间戳把它切成多句短字幕。
    """
    # 如果本来就很短，或者没有单词信息，直接返回原样
    if len(segment.text) <= max_chars or not segment.words:
        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        return

    # 开始切分逻辑
    current_words = []
    current_len = 0
    segment_start = segment.words[0].start

    for word in segment.words:
        word_text = word.word
        word_len = len(word_text)

        # 如果加上当前词会超长，且缓存里已经有词了 -> 立即结算上一句
        if current_len + word_len > max_chars and current_words:
            yield {
                "start": segment_start,
                "end": current_words[-1].end,
                "text": "".join([w.word for w in current_words]).strip()
            }
            # 重置下一句
            current_words = []
            current_len = 0
            segment_start = word.start

        current_words.append(word)
        current_len += word_len

    # 结算剩下的尾巴
    if current_words:
        yield {
            "start": segment_start,
            "end": current_words[-1].end,
            "text": "".join([w.word for w in current_words]).strip()
        }


# -----------------------------

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
        vad_params = {
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 1500,
        }

        # 1. 快速分析时长
        print("   🔍 分析视频时长...", end="", flush=True)
        _, info = batched_model.transcribe(video_path, batch_size=BATCH_SIZE)
        total_duration = info.duration
        print(f" -> {format_timestamp(total_duration)}")

        # 2. 开始转写
        start_time = time.time()

        magic_prompt = "饼干岁们好，我是岁己。今天直播玩游戏，杂谈唱歌。哎呀，这个好难啊？没关系，我们可以的。请多关照。"

        # 这里做一个判断：如果需要切分，必须开启 word_timestamps
        # 如果不需要切分，开启它可以提高精度，但关闭它可能会快一丢丢。
        # 为了保证时间轴质量，建议始终开启。
        segments, _ = batched_model.transcribe(
            video_path,
            batch_size=BATCH_SIZE,
            language="zh",
            initial_prompt=magic_prompt,
            vad_filter=True,
            vad_parameters=vad_params,
            word_timestamps=True
        )

        # 准备进度条
        term_width = shutil.get_terminal_size().columns
        bar_width = max(20, term_width - 50)

        line_count = 0

        with open(srt_path, "w", encoding="utf-8") as f:
            for raw_segment in segments:

                # --- 根据开关决定处理方式 ---
                if ENABLE_SMART_SPLIT:
                    # 使用智能切分
                    sub_segments = smart_split_segment(raw_segment, MAX_CHARS_PER_LINE)
                else:
                    # 不切分，直接包装成列表，方便下面统一处理
                    sub_segments = [{
                        "start": raw_segment.start,
                        "end": raw_segment.end,
                        "text": raw_segment.text.strip()
                    }]
                # -------------------------

                for split_seg in sub_segments:
                    line_count += 1

                    # 进度条逻辑
                    current_time = split_seg['end']
                    percent = (current_time / total_duration) * 100
                    if percent > 100: percent = 100

                    elapsed = time.time() - start_time
                    speed = current_time / elapsed if elapsed > 0 else 0
                    eta = (total_duration - current_time) / speed if speed > 0 else 0

                    filled_len = int(bar_width * percent / 100)
                    bar = '█' * filled_len + '-' * (bar_width - filled_len)

                    sys.stdout.write(f"\r   🚀 {percent:5.1f}% [{bar}] ETA:{int(eta)}s | {speed:.0f}x")
                    sys.stdout.flush()

                    start_str = format_timestamp(split_seg['start'])
                    end_str = format_timestamp(split_seg['end'])
                    text = split_seg['text']

                    f.write(f"{line_count}\n{start_str} --> {end_str}\n{text}\n\n")

                f.flush()

        total_time = time.time() - start_time
        print(f"\n   ✅ 完成！耗时: {total_time:.1f}s")

    except Exception as e:
        print(f"\n   ❌ 处理失败: {filename}")
        print(f"   错误信息: {e}")
        traceback.print_exc()


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
        if is_video_file(input_path):
            todo_list.append(input_path)
    else:
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
    print(f"🔧 智能切分状态: {'✅ 开启' if ENABLE_SMART_SPLIT else '⛔ 关闭 (保留长句)'}")
    print("=" * 60)

    # 2. 初始化模型
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
    print(f"⏱️  总耗时: {int(duration // 3600)}小时 {int((duration % 3600) // 60)}分")
    print("🛌 祝你好梦！")


if __name__ == "__main__":
    main()