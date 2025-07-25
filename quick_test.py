import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip
import warnings
from src.subtitle_processor import SubtitleProcessor, SubtitleRenderer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import moviepy as mp
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip
import warnings

def load_config():
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_captions():
    """加载字幕文件"""
    with open('assets/caption.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_subtitle_overlay_from_rst(frame, rst_renderer, current_time, stroke_width=2):
    """从RST渲染器创建字幕叠加"""
    height, width = frame.shape[:2]
    
    # 获取当前时间的字幕文本
    subtitle_text = rst_renderer.get_subtitle_at_time(current_time)
    
    if not subtitle_text.strip():
        return frame
    
    # 创建PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 获取样式配置
    style_config = rst_renderer.get_style_config()
    font_path = style_config['font_family']
    font_size = style_config['font_size']
    
    try:
        # 使用微软雅黑字体
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"Warning: Cannot load font {font_path}, using default font")
        font = ImageFont.load_default()
    
    # 获取文本尺寸
    bbox = draw.textbbox((0, 0), subtitle_text, font=font)
    text_width = bbox[2] - bbox[0]
    
    # 计算字幕位置（屏幕下方1/3区域的中心）
    subtitle_area_start = height * 2 // 3
    
    # 计算位置（居中对齐）
    x = (width - text_width) // 2
    y = subtitle_area_start
    
    # 绘制描边
    stroke_width = style_config['stroke_width']
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), subtitle_text, font=font, fill=(0, 0, 0))
    
    # 绘制主文字
    draw.text((x, y), subtitle_text, font=font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_audio_and_music(video_clip, voice_audio_path, bg_music_path, bg_volume=0.3):
    """为视频添加配音音频和背景音乐"""
    if not os.path.exists(voice_audio_path):
        print(f"Warning: Voice audio file {voice_audio_path} not found")
        if os.path.exists(bg_music_path):
            # 只有背景音乐
            bg_audio = AudioFileClip(bg_music_path).with_volume_scaled(bg_volume)
            if bg_audio.duration > video_clip.duration:
                bg_audio = bg_audio.subclipped(0, video_clip.duration)
            return video_clip.with_audio(bg_audio)
        return video_clip
    
    try:
        # 加载配音音频
        voice_audio = AudioFileClip(voice_audio_path)
        
        # 调整视频长度以匹配配音长度
        if voice_audio.duration < video_clip.duration:
            video_clip = video_clip.subclipped(0, voice_audio.duration)
        elif voice_audio.duration > video_clip.duration:
            voice_audio = voice_audio.subclipped(0, video_clip.duration)
        
        # 如果有背景音乐，混合音频
        if os.path.exists(bg_music_path):
            bg_audio = AudioFileClip(bg_music_path).with_volume_scaled(bg_volume)
            
            # 调整背景音乐长度
            if bg_audio.duration > video_clip.duration:
                bg_audio = bg_audio.subclipped(0, video_clip.duration)
            elif bg_audio.duration < video_clip.duration:
                # 循环背景音乐
                loops_needed = int(np.ceil(video_clip.duration / bg_audio.duration))
                bg_clips = [bg_audio] * loops_needed
                bg_audio = concatenate_audioclips(bg_clips)
                bg_audio = bg_audio.subclipped(0, video_clip.duration)
            
            # 混合配音和背景音乐
            final_audio = CompositeAudioClip([voice_audio, bg_audio])
        else:
            final_audio = voice_audio
        
        # 为视频设置混合音频
        final_video = video_clip.with_audio(final_audio)
        
        return final_video
        
    except Exception as e:
        print(f"Error adding audio: {e}")
        return video_clip

def create_video_with_ken_burns():
    """创建带有Ken Burns效果的视频，第一个shot测试：使用第一条字幕滚动显示，与音频同步"""
    
    config = load_config()
    captions = load_captions()
    
    # 使用新的字幕处理系统
    print("初始化字幕处理系统...")
    
    # 处理第一条字幕：生成RST文件
    rst_output_path = "assets/subtitles/shot_01.rst"
    os.makedirs(os.path.dirname(rst_output_path), exist_ok=True)
    
    if captions and len(captions) > 0:
        first_caption = captions[0]
        audio_path = "assets/audios/01.wav"
        
        # 初始化字幕处理器
        processor = SubtitleProcessor("config.json")
        timing_info = processor.process_subtitle(first_caption, audio_path, rst_output_path)
        
        # 初始化字幕渲染器
        rst_renderer = SubtitleRenderer(rst_output_path)
        print(f"字幕处理完成，共 {len(timing_info)} 个片段")
    else:
        print("警告：没有找到字幕内容")
        rst_renderer = None
    
    # 视频参数
    fps = config['video_settings']['fps']
    duration = 12.24  # 使用固定的12.24秒
    width = 1440  # 使用固定的1440x1920分辨率
    height = 1920
    
    # 字幕参数，使用微软雅黑字体
    font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径
    font_size = 70
    
    # 计算总帧数
    total_frames = int(duration * fps)
    print(f"Creating video: {duration}s, {fps}fps, Total frames: {total_frames}")
    
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/quick_test.mp4', fourcc, fps, (width, height))
    
    # Ken Burns效果参数
    zoom_factor_start = 1.0
    zoom_factor_end = 1.05  # 5%缩放
    
    # 加载第一张图片
    img_path = 'assets/images/Tomato_00001_.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (width, height))
    
    for frame_idx in range(total_frames):
        current_time = frame_idx / fps
        
        # 计算Ken Burns缩放
        progress = frame_idx / total_frames
        current_zoom = zoom_factor_start + (zoom_factor_end - zoom_factor_start) * progress
        
        # 应用缩放
        zoom_height = int(height * current_zoom)
        zoom_width = int(width * current_zoom)
        zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
        
        # 从中心裁剪
        start_y = (zoom_height - height) // 2
        start_x = (zoom_width - width) // 2
        frame = zoomed_img[start_y:start_y+height, start_x:start_x+width]
        
        # 添加字幕 - 使用RST渲染器渲染字幕
        if rst_renderer:
            frame = create_subtitle_overlay_from_rst(
                frame, rst_renderer, current_time, stroke_width=3
            )
        
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
    
    out.release()
    print("Video created: videos/quick_test.mp4")
    
    # 添加配音和背景音乐
    print("Adding voice audio and background music...")
    video_clip = VideoFileClip("videos/quick_test.mp4")
    
    # 添加配音音频和背景音乐
    final_video = add_audio_and_music(video_clip, "assets/audios/01.wav", "assets/pianai.mp3", bg_volume=0.3)
    
    # 保存最终视频
    final_video.write_videofile(
        "videos/quick_test_with_audio.mp4",
        codec='libx264',
        audio_codec='aac',
        fps=fps
    )
    
    # 清理资源
    video_clip.close()
    final_video.close()
    
    print("Final video with audio created: videos/quick_test_with_audio.mp4")
    print("Font: Microsoft YaHei (微软雅黑)")
    print("Voice Audio: assets/audios/01.wav")
    print("Background Music: assets/pianai.mp3 (30% volume)")
    print("Subtitles: Generated from RST file, punctuation removed, sentence-based display, synced with voice")
    print(f"RST File: {rst_output_path}")

if __name__ == "__main__":
    # 忽略UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("Creating first shot test video with RST-based subtitle system...")
    create_video_with_ken_burns()