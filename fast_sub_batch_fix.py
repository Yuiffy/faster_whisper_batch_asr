import sys
import os
import time
import shutil
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ================= 性能配置 =================
# 使用 HuggingFace 上的转换版 Turbo 模型 (速度接近 Medium，精度接近 Large)
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
# MODEL_SIZE = "large-v3"

# Batch Size 保持适中，16 或 32 都可以
BATCH_SIZE = 16
# ===========================================

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
        
        batched_model = BatchedInferencePipeline(model=model)

        # 2. 准备 VAD 参数（防止烂尾）
        print("🔧 配置 VAD 参数以防止吞字...")
        vad_params = {
            "min_silence_duration_ms": 2000, 
            "speech_pad_ms": 1500,           
        }

        # 3. 预处理获取时长
        print("🔍 分析音频流...")
        # 这里的 batch_size 仅用于快速探测，不影响后续
        dummy_gen, info = batched_model.transcribe(video_path, batch_size=BATCH_SIZE)
        total_duration = info.duration
        
        print(f"✅ 视频总长: {format_timestamp(total_duration)}")
        print("🚀 竞速模式启动 (Turbo模型 + 防烂尾 + 防覆盖)...")
        print("=" * 50)

        # --- 🛡️ 核心修改：智能防覆盖逻辑 🛡️ ---
        output_dir = os.path.dirname(video_path)
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, filename_no_ext + ".srt")
        
        # 循环检测：如果文件存在，就加后缀 _1, _2, _3...
        counter = 1
        original_srt_path = srt_path # 记录一下原本想存的名字
        while os.path.exists(srt_path):
            new_filename = f"{filename_no_ext}_{counter}.srt"
            srt_path = os.path.join(output_dir, new_filename)
            counter += 1
            
        if counter > 1:
            print(f"⚠️  检测到同名文件: {os.path.basename(original_srt_path)}")
            print(f"✨ 自动重命名为: {os.path.basename(srt_path)}")
        else:
            print(f"💾 准备保存为: {os.path.basename(srt_path)}")
        # ---------------------------------------
        
        start_time = time.time()
        
        # 4. 开始转写
        segments, _ = batched_model.transcribe(
            video_path, 
            batch_size=BATCH_SIZE,
            language="zh",
            initial_prompt="饼干岁们好，我是岁己。今天直播玩游戏，杂谈唱歌。哎呀，这个好难啊？没关系，我们可以的。请多关照。",
            vad_filter=True,            
            vad_parameters=vad_params   
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