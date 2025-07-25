"""
为已渲染的视频添加音频
专门处理 videos/shot_XX.mp4 文件，为它们添加对应的音频和背景音乐
"""

import os
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip, concatenate_videoclips
import numpy as np


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


def process_single_video(shot_number):
    """为单个视频添加音频"""
    print(f"\n=== 处理 Shot {shot_number:02d} 音频 ===")
    
    # 文件路径
    video_path = f"videos/shot_{shot_number:02d}.mp4"
    audio_path = f"assets/audios/{shot_number:02d}.wav"
    bg_music_path = "assets/pianai.mp3"
    output_path = f"videos/shot_{shot_number:02d}_with_audio.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None
        
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return None
    
    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"输出文件已存在，跳过: {output_path}")
        return output_path
    
    try:
        print(f"加载视频: {video_path}")
        video_clip = VideoFileClip(video_path)
        
        print(f"添加音频: {audio_path}")
        print(f"背景音乐: {bg_music_path}")
        
        # 添加音频和背景音乐
        final_video = add_audio_and_music(
            video_clip, 
            audio_path, 
            bg_music_path, 
            bg_volume=0.3
        )
        
        print(f"保存到: {output_path}")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30
        )
        
        # 清理资源
        video_clip.close()
        final_video.close()
        
        print(f"完成: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"处理 Shot {shot_number:02d} 时出错: {e}")
        return None


def process_all_videos():
    """为所有视频添加音频"""
    print("=== 开始为所有视频添加音频 ===")
    
    successful_videos = []
    
    # 处理 Shot 01-15
    for shot_num in range(1, 16):
        result = process_single_video(shot_num)
        if result:
            successful_videos.append(result)
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理 {len(successful_videos)} 个视频")
    
    # 连接所有视频
    if len(successful_videos) >= 2:
        print(f"\n=== 连接所有视频 ===")
        try:
            print("加载所有视频片段...")
            clips = []
            for path in successful_videos:
                print(f"  加载: {path}")
                clip = VideoFileClip(path)
                clips.append(clip)
            
            print("连接视频...")
            final_video = concatenate_videoclips(clips, method="compose")
            
            output_path = "videos/complete_video_with_audio.mp4"
            print(f"保存完整视频: {output_path}")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=30
            )
            
            # 清理资源
            for clip in clips:
                clip.close()
            final_video.close()
            
            # 计算总时长
            total_duration = sum(clip.duration for clip in clips)
            print(f"完整视频创建成功: {output_path}")
            print(f"总时长: {total_duration:.2f} 秒")
            
        except Exception as e:
            print(f"连接视频时出错: {e}")
    
    return successful_videos


if __name__ == "__main__":
    process_all_videos()
