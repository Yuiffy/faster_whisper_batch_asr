import sys
import os
import time
import shutil
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= 性能配置 =================
# 5080 显卡推荐配置
#MODEL_SIZE = "large-v3" 

#BATCH_SIZE = 16        # 并发数，越大越快，爆显存就改小 (8 或 4)
# 1. 改为 "small" (推荐) 或 "base" (极速但不太准)
#MODEL_SIZE = "small" 

# 2. 既然模型变小了，显存空出来了，我们可以把并发加大！
# 5080 显存巨大，跑 small 模型甚至可以开到 32 或 64
#BATCH_SIZE = 32

# 使用 HuggingFace 上的转换版 Turbo 模型
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Batch Size 保持适中
BATCH_SIZE = 16
# ===========================================

def main():
    # 0. 清屏，准备起飞
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
        print(f"⏳ 正在预热引擎 ({MODEL_SIZE}, Batch={BATCH_SIZE})...")
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)

        # 2. 预处理（获取时长）
        print("🔍 分析音频流...")
        # 这里我们做一个极其轻量的 dummy 调用来获取 info，或者直接用 batched_model
        # faster-whisper 的 transcribe 会返回 (segments, info)
        # segments 是生成器，info 包含时长
        segments, info = batched_model.transcribe(
            video_path, 
            batch_size=BATCH_SIZE,
            language="zh",
            initial_prompt="以下是四川口音的二次元虚拟主播直播录像，请使用简体中文。"
        )

        total_duration = info.duration
        print(f"✅ 视频总长: {format_timestamp(total_duration)} ({total_duration:.2f}秒)")
        print("🚀 竞速模式启动！(仅显示进度，不刷屏文字)")
        print("=" * 50)

        # 3. 跑分式进度条
        output_dir = os.path.dirname(video_path)
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, filename_no_ext + ".srt")
        
        start_time = time.time()
        
        # 获取终端宽度用于绘制进度条
        term_width = shutil.get_terminal_size().columns
        bar_width = max(20, term_width - 40) # 动态调整进度条长度

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                # --- 进度计算核心 ---
                current_time = segment.end
                percent = (current_time / total_duration) * 100
                if percent > 100: percent = 100
                
                # 计算剩余时间 (ETA)
                elapsed = time.time() - start_time
                speed = current_time / elapsed if elapsed > 0 else 0 # 这里的 speed 是 "x倍速"
                eta = (total_duration - current_time) / speed if speed > 0 else 0
                
                # 绘制进度条 [█████-----] 50%
                filled_len = int(bar_width * percent / 100)
                bar = '█' * filled_len + '-' * (bar_width - filled_len)
                
                # \r 让光标回到行首，实现单行刷新（不刷屏）
                sys.stdout.write(f"\r[{bar}] {percent:5.1f}% | ETA: {int(eta)}s | 倍速: {speed:.1f}x")
                sys.stdout.flush()

                # 写入文件
                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                text = segment.text.strip()
                f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
                
                # 强制落盘，防断电
                if i % 5 == 0: f.flush() 

        total_time = time.time() - start_time
        print("\n" + "=" * 50) # 换行，防止最后一行被覆盖
        print(f"🏆 任务完成！")
        print(f"⏱️  实际耗时: {total_time:.2f}秒")
        print(f"⚡ 平均倍速: {total_duration/total_time:.1f} 倍速")
        print(f"💾 字幕已保存: {srt_path}")

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