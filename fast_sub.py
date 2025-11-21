import sys
import os
# 注意这里导入的是 faster_whisper
from faster_whisper import WhisperModel 

# 既然是 5080，直接上 large-v3，且使用 float16 精度
MODEL_SIZE = "large-v3" 

def main():
    if len(sys.argv) < 2:
        print("❌ 请直接拖拽视频文件到 .bat 上")
        return

    video_path = sys.argv[1]
    print(f"📂 处理文件: {os.path.basename(video_path)}")

    try:
        print(f"⏳ 正在加载 Faster-Whisper 模型 ({MODEL_SIZE})...")
        # device="auto" 会自动尝试调用 GPU，如果 CTranslate2 支持 5080，这里就能直接跑
        # 如果还是报错，把 device="auto" 改成 device="cpu"
        model = WhisperModel(MODEL_SIZE, device="auto", compute_type="float16")

        print("🎙️  开始极速转写...")
        segments, info = model.transcribe(
            video_path, 
            beam_size=5, 
            language="zh",
            initial_prompt="以下是四川口音的二次元虚拟主播直播录像，请使用简体中文。"
        )

        # 准备写入文件
        output_dir = os.path.dirname(video_path)
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, filename_no_ext + ".srt")

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                # 格式化时间戳
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip()
                
                # 实时打印进度
                print(f"[{start} --> {end}] {text}")
                
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        print(f"\n✅ 完成！文件已保存: {srt_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def format_timestamp(seconds):
    # 简单的辅助函数把秒数转为 00:00:00,000 格式
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    main()