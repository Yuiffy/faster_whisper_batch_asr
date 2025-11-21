import sys
import os
import time
from faster_whisper import WhisperModel

# ================= 配置区域 =================
# 改回 medium，速度更快，精度对日常够用
# 如果想换回最强模型，改回 "large-v3" 即可
MODEL_SIZE = "medium" 
# ===========================================

def main():
    if len(sys.argv) < 2:
        print("❌ 请直接拖拽视频文件到 .bat 上")
        return

    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 找不到文件: {video_path}")
        return

    print(f"📂 正在加载视频: {os.path.basename(video_path)}")

    try:
        # 1. 加载模型
        print(f"⏳ 正在初始化 Faster-Whisper ({MODEL_SIZE})...")
        # compute_type="int8" 在 CPU 上也会快很多，精度损失很小
        # device="auto" 会优先尝试 GPU
        model = WhisperModel(MODEL_SIZE, device="auto", compute_type="int8")

        # 2. 预处理，获取视频总时长
        print("🔍 正在分析音频流...")
        segments_generator, info = model.transcribe(
            video_path, 
            beam_size=5, 
            language="zh",
            initial_prompt="以下是四川口音的二次元虚拟主播直播录像，请使用简体中文。"
        )

        total_duration = info.duration
        print(f"✅ 视频总时长: {format_timestamp(total_duration)} ({total_duration:.2f}秒)")
        print(f"🚀 开始转写 (按 Ctrl+C 可以随时中断并保存)")
        print("=" * 60)

        output_dir = os.path.dirname(video_path)
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, filename_no_ext + ".srt")

        start_time = time.time()
        
        # 标记是否是人为中断
        interrupted = False

        with open(srt_path, "w", encoding="utf-8") as f:
            try:
                # 遍历生成器
                for i, segment in enumerate(segments_generator, start=1):
                    # 计算进度
                    current_end = segment.end
                    percent = (current_end / total_duration) * 100
                    if percent > 100: percent = 100
                    
                    # 格式化
                    start_str = format_timestamp(segment.start)
                    end_str = format_timestamp(segment.end)
                    text = segment.text.strip()

                    # 估算剩余时间
                    elapsed = time.time() - start_time
                    speed = current_end / elapsed if elapsed > 0 else 0
                    eta = (total_duration - current_end) / speed if speed > 0 else 0
                    
                    # 打印进度
                    print(f"[{percent:5.1f}%] {text[:50]}... (ETA: {int(eta)}s)")

                    # 写入文件
                    f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
                    
                    # 【关键】强制刷新缓冲区，确保每一句都真正写到了硬盘里
                    f.flush() 

            except KeyboardInterrupt:
                interrupted = True
                print("\n" + "!" * 40)
                print("🛑 检测到用户中断！正在保存已生成的字幕...")
                print("!" * 40)
                # 此时退出循环，with 语句会自动安全关闭文件

        total_time = time.time() - start_time
        print("=" * 60)
        if interrupted:
            print(f"⚠️  任务已中断，但字幕文件是安全的。")
            print(f"📂 字幕只生成到了: {format_timestamp(time.time() - start_time)}")
        else:
            print(f"✅ 全部完成！耗时: {total_time:.1f}秒")
        
        print(f"📄 文件位置: {srt_path}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
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