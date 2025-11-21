import sys
import os
import whisper
import torch
from whisper.utils import get_writer

# ================= 配置区域 =================
# 模型大小：tiny, base, small, medium, large
# 推荐 medium (平衡) 或 large (最准但慢)
MODEL_SIZE = "medium" 
# ===========================================

def main():
    # 1. 检查是否有文件被拖入
    if len(sys.argv) < 2:
        print("❌ 错误：请直接把视频文件拖拽到 .bat 文件上！")
        return

    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 错误：找不到文件: {video_path}")
        return

    print(f"📂 正在处理文件: {os.path.basename(video_path)}")

    # 2. 检查硬件加速 (GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 运行设备: {device.upper()}")
    if device == "cpu":
        print("⚠️  警告：未检测到 N卡 或 CUDA 环境，使用 CPU 速度会非常慢！")

    try:
        # 3. 加载模型
        print(f"⏳ 正在加载 Whisper 模型 ({MODEL_SIZE})...")
        model = whisper.load_model(MODEL_SIZE, device=device)

        # 4. 开始识别 (Transcribing)
        print("🎙️  正在识别中，请耐心等待 (大文件可能需要几分钟到几十分钟)...")
        # initial_prompt 可以用来引导模型，比如加标点或特定术语，这里先留空
        result = model.transcribe(video_path, language="zh", verbose=True)

        # 5. 保存字幕文件 (.srt)
        output_dir = os.path.dirname(video_path)
        # 获取不带后缀的文件名
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        
        # 实例化 srt writer
        srt_writer = get_writer("srt", output_dir)
        
        # 写入文件
        srt_writer(result, filename_no_ext)

        print("\n" + "="*30)
        print(f"✅ 成功！字幕已生成在原目录：")
        print(f"📄 {os.path.join(output_dir, filename_no_ext + '.srt')}")
        print("="*30 + "\n")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        # 提示常见错误
        if "ffmpeg" in str(e).lower():
            print("💡 提示：似乎是找不到 FFmpeg，请确认它已安装并添加到了系统环境变量 Path 中。")

if __name__ == "__main__":
    main()