"""
完整视频渲染器（多线程优化版本）
直接渲染完整视频，不生成单独的shot片段
背景音乐连续播放，根据视频总时长裁剪
使用多线程并发处理提高渲染效率
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
import warnings
from src.subtitle_processor import SubtitleProcessor, SubtitleRenderer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import lru_cache
import time
import random
import math
from typing import Dict, List, Tuple, Optional


# 全局缓存，避免重复加载相同资源
image_cache: Dict[str, np.ndarray] = {}
image_cache_lock = threading.Lock()

# 字体缓存
font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
font_cache_lock = threading.Lock()

# Ken Burns特效类型定义
class KenBurnsEffect:
    """Ken Burns特效类型"""
    ZOOM_IN = "zoom_in"           # 缩放放大
    ZOOM_OUT = "zoom_out"         # 缩放缩小
    PAN_LEFT = "pan_left"         # 向左平移
    PAN_RIGHT = "pan_right"       # 向右平移
    PAN_UP = "pan_up"             # 向上平移
    PAN_DOWN = "pan_down"         # 向下平移
    ZOOM_PAN_LEFT = "zoom_pan_left"   # 缩放+左移
    ZOOM_PAN_RIGHT = "zoom_pan_right" # 缩放+右移
    ZOOM_PAN_UP = "zoom_pan_up"       # 缩放+上移
    ZOOM_PAN_DOWN = "zoom_pan_down"   # 缩放+下移
    ROTATE_ZOOM = "rotate_zoom"       # 旋转+缩放
    SPIRAL = "spiral"                 # 螺旋效果

# 预定义的特效序列（确保视觉多样性）
EFFECT_SEQUENCE = [
    KenBurnsEffect.ZOOM_IN,
    KenBurnsEffect.PAN_LEFT,
    KenBurnsEffect.ZOOM_PAN_RIGHT,
    KenBurnsEffect.PAN_UP,
    KenBurnsEffect.ZOOM_OUT,
    KenBurnsEffect.PAN_DOWN,
    KenBurnsEffect.ZOOM_PAN_LEFT,
    KenBurnsEffect.SPIRAL,
    KenBurnsEffect.PAN_RIGHT,
    KenBurnsEffect.ZOOM_PAN_UP,
    KenBurnsEffect.ROTATE_ZOOM,
    KenBurnsEffect.ZOOM_IN,
    KenBurnsEffect.PAN_LEFT,
    KenBurnsEffect.ZOOM_PAN_DOWN,
    KenBurnsEffect.PAN_UP
]


@lru_cache(maxsize=32)
def load_config():
    """加载配置文件（带缓存）"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache(maxsize=32)
def load_captions(shot="shot_01"):
    """加载字幕文件（带缓存）"""
    caption_path = f'assets/{shot}/subtitles/{shot}_caption.json'
    with open(caption_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_cached_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """获取缓存的字体"""
    cache_key = (font_path, font_size)
    
    with font_cache_lock:
        if cache_key in font_cache:
            return font_cache[cache_key]
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            font_cache[cache_key] = font
            return font
        except:
            font = ImageFont.load_default()
            font_cache[cache_key] = font
            return font


def preload_images(shot, scene_count: int, width: int, height: int) -> None:
    """预加载所有图像到缓存"""
    def load_single_image(scene_number: int):
        # 根据新的文件结构查找图像文件
        image_path = f"assets/{shot}/images/{shot}_{scene_number}.png"
        
        with image_cache_lock:
            if image_path in image_cache:
                return
        
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            print(f"警告：无法加载图像 {image_path}")
        else:
            img = cv2.resize(img, (width, height))
        
        with image_cache_lock:
            image_cache[image_path] = img
    
    # 使用线程池并发加载图像
    with ThreadPoolExecutor(max_workers=min(16, scene_count)) as executor:
        futures = [executor.submit(load_single_image, i) for i in range(1, scene_count + 1)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                pass  # 忽略单个图像加载错误
    
    print(f"图像缓存完成: {len(image_cache)}张")


def apply_ken_burns_effect(img: np.ndarray, progress: float, effect_type: str, 
                          width: int, height: int) -> np.ndarray:
    """
    应用Ken Burns特效，特效持续时间与场景持续时间匹配
    
    Args:
        img: 输入图像
        progress: 进度 (0.0 到 1.0) - 基于场景持续时间计算
        effect_type: 特效类型
        width: 目标宽度
        height: 目标高度
    
    Returns:
        处理后的图像
    """
    # 确保进度在有效范围内
    progress = max(0.0, min(1.0, progress))
    
    # 缓动函数（使运动更自然），不加速，与场景持续时间自然匹配
    eased_progress = easing_function(progress)
    
    if effect_type == KenBurnsEffect.ZOOM_IN:
        return apply_zoom_in(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ZOOM_OUT:
        return apply_zoom_out(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.PAN_LEFT:
        return apply_pan_left(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.PAN_RIGHT:
        return apply_pan_right(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.PAN_UP:
        return apply_pan_up(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.PAN_DOWN:
        return apply_pan_down(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ZOOM_PAN_LEFT:
        return apply_zoom_pan_left(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ZOOM_PAN_RIGHT:
        return apply_zoom_pan_right(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ZOOM_PAN_UP:
        return apply_zoom_pan_up(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ZOOM_PAN_DOWN:
        return apply_zoom_pan_down(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.ROTATE_ZOOM:
        return apply_rotate_zoom(img, eased_progress, width, height)
    elif effect_type == KenBurnsEffect.SPIRAL:
        return apply_spiral(img, eased_progress, width, height)
    else:
        # 默认缩放效果
        return apply_zoom_in(img, eased_progress, width, height)


def easing_function(t: float) -> float:
    """缓动函数，使运动更自然"""
    # 使用ease-in-out cubic函数
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def apply_zoom_in(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放放大效果"""
    zoom_start = 1.0
    zoom_end = 1.3  # 增加缩放幅度
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    zoom_height = int(height * current_zoom)
    zoom_width = int(width * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    # 从中心裁剪
    start_y = (zoom_height - height) // 2
    start_x = (zoom_width - width) // 2
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_zoom_out(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放缩小效果"""
    zoom_start = 1.3
    zoom_end = 1.0
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    zoom_height = int(height * current_zoom)
    zoom_width = int(width * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_y = (zoom_height - height) // 2
    start_x = (zoom_width - width) // 2
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_pan_left(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """向左平移效果"""
    pan_distance = int(width * 0.3)  # 30%的宽度
    zoom_factor = 1.2  # 轻微放大以确保覆盖
    
    zoom_width = int(width * zoom_factor)
    zoom_height = int(height * zoom_factor)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    # 计算起始位置（从右侧开始）
    start_x = int((zoom_width - width) - progress * pan_distance)
    start_y = (zoom_height - height) // 2
    
    start_x = max(0, min(start_x, zoom_width - width))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_pan_right(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """向右平移效果"""
    pan_distance = int(width * 0.3)
    zoom_factor = 1.2
    
    zoom_width = int(width * zoom_factor)
    zoom_height = int(height * zoom_factor)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = int(progress * pan_distance)
    start_y = (zoom_height - height) // 2
    
    start_x = max(0, min(start_x, zoom_width - width))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_pan_up(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """向上平移效果"""
    pan_distance = int(height * 0.3)
    zoom_factor = 1.2
    
    zoom_width = int(width * zoom_factor)
    zoom_height = int(height * zoom_factor)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = (zoom_width - width) // 2
    start_y = int((zoom_height - height) - progress * pan_distance)
    
    start_y = max(0, min(start_y, zoom_height - height))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_pan_down(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """向下平移效果"""
    pan_distance = int(height * 0.3)
    zoom_factor = 1.2
    
    zoom_width = int(width * zoom_factor)
    zoom_height = int(height * zoom_factor)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = (zoom_width - width) // 2
    start_y = int(progress * pan_distance)
    
    start_y = max(0, min(start_y, zoom_height - height))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_zoom_pan_left(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放+左移组合效果"""
    zoom_start = 1.0
    zoom_end = 1.25
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    pan_distance = int(width * 0.2)
    
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = int((zoom_width - width) / 2 - progress * pan_distance)
    start_y = (zoom_height - height) // 2
    
    start_x = max(0, min(start_x, zoom_width - width))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_zoom_pan_right(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放+右移组合效果"""
    zoom_start = 1.0
    zoom_end = 1.25
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    pan_distance = int(width * 0.2)
    
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = int((zoom_width - width) / 2 + progress * pan_distance)
    start_y = (zoom_height - height) // 2
    
    start_x = max(0, min(start_x, zoom_width - width))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_zoom_pan_up(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放+上移组合效果"""
    zoom_start = 1.0
    zoom_end = 1.25
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    pan_distance = int(height * 0.2)
    
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = (zoom_width - width) // 2
    start_y = int((zoom_height - height) / 2 - progress * pan_distance)
    
    start_y = max(0, min(start_y, zoom_height - height))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_zoom_pan_down(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """缩放+下移组合效果"""
    zoom_start = 1.0
    zoom_end = 1.25
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    pan_distance = int(height * 0.2)
    
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    start_x = (zoom_width - width) // 2
    start_y = int((zoom_height - height) / 2 + progress * pan_distance)
    
    start_y = max(0, min(start_y, zoom_height - height))
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]


def apply_rotate_zoom(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """旋转+缩放效果"""
    zoom_start = 1.1
    zoom_end = 1.3
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    # 旋转角度（±3度）
    rotation_angle = math.sin(progress * math.pi) * 3
    
    # 创建更大的画布进行旋转
    canvas_size = int(max(width, height) * current_zoom * 1.5)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # 缩放图像
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    # 将图像放置在画布中心
    start_y = (canvas_size - zoom_height) // 2
    start_x = (canvas_size - zoom_width) // 2
    canvas[start_y:start_y+zoom_height, start_x:start_x+zoom_width] = zoomed_img
    
    # 应用旋转
    center = (canvas_size // 2, canvas_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(canvas, rotation_matrix, (canvas_size, canvas_size))
    
    # 从中心裁剪目标尺寸
    final_start_y = (canvas_size - height) // 2
    final_start_x = (canvas_size - width) // 2
    return rotated[final_start_y:final_start_y+height, final_start_x:final_start_x+width]


def apply_spiral(img: np.ndarray, progress: float, width: int, height: int) -> np.ndarray:
    """螺旋效果（缩放+旋转+轻微平移）"""
    zoom_start = 1.0
    zoom_end = 1.4
    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    # 螺旋运动
    angle = progress * math.pi * 2  # 完整圆周
    spiral_radius = min(width, height) * 0.1 * progress  # 螺旋半径
    
    pan_x = int(math.cos(angle) * spiral_radius)
    pan_y = int(math.sin(angle) * spiral_radius)
    
    # 缩放
    zoom_width = int(width * current_zoom)
    zoom_height = int(height * current_zoom)
    zoomed_img = cv2.resize(img, (zoom_width, zoom_height))
    
    # 计算起始位置（加上螺旋偏移）
    start_x = (zoom_width - width) // 2 + pan_x
    start_y = (zoom_height - height) // 2 + pan_y
    
    # 确保不越界
    start_x = max(0, min(start_x, zoom_width - width))
    start_y = max(0, min(start_y, zoom_height - height))
    
    return zoomed_img[start_y:start_y+height, start_x:start_x+width]
    """获取缓存的图像"""
    with image_cache_lock:
        if image_path in image_cache:
            return image_cache[image_path].copy()  # 返回副本避免多线程冲突
    
    # 如果缓存中没有，现场加载
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (width, height))
    
    with image_cache_lock:
        image_cache[image_path] = img
    
    return img.copy()


def get_cached_image(shot: str, scene_number: int, width: int, height: int) -> np.ndarray:
    """获取缓存的图像"""
    # 根据新的文件结构查找图像文件
    image_path = f"assets/{shot}/images/{shot}_{scene_number}.png"
    
    with image_cache_lock:
        if image_path in image_cache:
            return image_cache[image_path].copy()  # 返回副本避免多线程冲突
    
    # 如果缓存中没有，现场加载
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        print(f"警告：无法加载图像 {image_path}")
    else:
        img = cv2.resize(img, (width, height))
    
    with image_cache_lock:
        image_cache[image_path] = img
    
    return img.copy()


def create_subtitle_overlay_from_rst(frame, rst_renderer, current_time, stroke_width=2):
    """从RST渲染器创建字幕叠加（优化版本）"""
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
    
    # 使用缓存的字体
    font = get_cached_font(font_path, font_size)
    
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


def get_audio_duration(audio_path):
    """获取音频文件时长"""
    if not os.path.exists(audio_path):
        return 3.0  # 默认时长
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration
    except:
        return 3.0


def render_frame_batch(frame_batch_info: List[Tuple[int, float]], 
                      all_segments: List[Dict], 
                      complete_rst_renderer: SubtitleRenderer,
                      width: int, height: int, fps: int,
                      shot: str,
                      zoom_factor_start: float = 1.0, 
                      zoom_factor_end: float = 1.05) -> List[Tuple[int, np.ndarray]]:
    """渲染一批帧（线程安全）- 使用多样化Ken Burns特效"""
    rendered_frames = []
    
    for frame_idx, current_time in frame_batch_info:
        # 确定当前时间对应的场景和场景信息
        scene_number = 1
        scene_start_time = 0
        scene_end_time = 0
        
        # 找到当前场景的完整信息
        for i, segment in enumerate(all_segments):
            if segment['start_time'] <= current_time < segment['end_time']:
                scene_number = segment['scene']
                # 找到这个场景的开始和结束时间
                scene_segments = [seg for seg in all_segments if seg['scene'] == scene_number]
                scene_start_time = min(seg['start_time'] for seg in scene_segments)
                scene_end_time = max(seg['end_time'] for seg in scene_segments)
                break
        
        # 如果没有找到匹配的段，使用默认值
        if scene_end_time == 0:
            scene_end_time = scene_start_time + 3.0
        
        # 加载对应场景的图片（使用缓存）
        img = get_cached_image(shot, scene_number, width, height)
        
        # 计算场景内的Ken Burns效果进度（从场景开始到场景结束）
        scene_duration = scene_end_time - scene_start_time
        if scene_duration <= 0:
            scene_duration = 1.0
        
        # 计算当前时间在整个场景中的进度（0.0 到 1.0）
        scene_progress = (current_time - scene_start_time) / scene_duration
        scene_progress = max(0, min(1, scene_progress))  # 限制在0-1之间
        
        # 根据场景选择Ken Burns特效类型（循环使用预定义序列）
        effect_type = EFFECT_SEQUENCE[(scene_number - 1) % len(EFFECT_SEQUENCE)]
        
        # 应用Ken Burns特效（持续时间与场景匹配，从场景开始立即执行）
        frame = apply_ken_burns_effect(
            img, scene_progress, effect_type, width, height
        )
        
        # 添加字幕（使用完整的RST渲染器）
        frame = create_subtitle_overlay_from_rst(
            frame, complete_rst_renderer, current_time, stroke_width=3
        )
        
        rendered_frames.append((frame_idx, frame))
    
    return rendered_frames


def load_audio_clips_concurrent(shot: str, scene_durations: Dict[int, float]) -> List[AudioFileClip]:
    """并发加载音频文件"""
    print("并发加载音频文件...")
    
    def load_single_audio(scene_audio_info):
        scene, audio_path, start_time = scene_audio_info
        if os.path.exists(audio_path):
            try:
                voice_clip = AudioFileClip(audio_path)
                voice_clip = voice_clip.with_start(start_time)
                return voice_clip
            except Exception as e:
                print(f"加载音频失败: {audio_path}, 错误: {e}")
                return None
        else:
            print(f"音频文件不存在: {audio_path}")
            return None
    
    # 准备音频加载任务
    audio_tasks = []
    current_time = 0
    
    # 使用新的文件结构：assets/{shot}/audios/{shot}_{scene}.wav
    for scene in sorted(scene_durations.keys()):
        audio_path = f"assets/{shot}/audios/{shot}_{scene}.wav"
        audio_tasks.append((scene, audio_path, current_time))
        current_time = scene_durations[scene]
    
    # 并发加载音频
    voice_clips = []
    with ThreadPoolExecutor(max_workers=min(8, len(audio_tasks))) as executor:
        futures = [executor.submit(load_single_audio, task) for task in audio_tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    voice_clips.append(result)
            except Exception as e:
                print(f"音频加载任务失败: {e}")
    
    print(f"音频加载完成: {len(voice_clips)}个")
    return voice_clips


def create_complete_video(shot="shot_02"):
    """创建完整视频（多线程优化版本 - 直接输出最终视频）"""
    print(f"=== 开始创建完整视频（{shot} - 多线程版本） ===")
    start_time = time.time()
    
    # 加载配置和字幕
    config = load_config()
    captions = load_captions(shot)
    
    if not captions:
        print("错误：没有找到字幕内容")
        return
    
    print(f"找到 {len(captions)} 条字幕")
    
    # 生成完整的RST字幕文件
    audio_files = []
    
    # 使用新的文件结构：assets/{shot}/audios/{shot}_{scene}.wav
    for i in range(1, len(captions) + 1):
        audio_path = f"assets/{shot}/audios/{shot}_{i}.wav"
        audio_files.append(audio_path)
    
    complete_rst_path = f"assets/{shot}/subtitles/{shot}_complete_video.rst"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(complete_rst_path), exist_ok=True)
    
    # 生成字幕文件
    processor = SubtitleProcessor("config.json")
    all_segments = processor.generate_complete_rst_file(captions, audio_files, complete_rst_path)
    
    # 初始化完整字幕渲染器
    complete_rst_renderer = SubtitleRenderer(complete_rst_path)
    
    # 计算总时长
    total_duration = max(seg['end_time'] for seg in all_segments) if all_segments else 0
    scene_count = len(set(seg['scene'] for seg in all_segments))
    print(f"视频总时长: {total_duration:.2f}s，字幕片段: {len(all_segments)}个")
    print(f"Ken Burns特效: {scene_count}个场景，2倍速度，多样化效果")
    
    # 视频参数
    fps = config['video_settings']['fps']
    width = 1440
    height = 1920
    
    # 计算总帧数
    total_frames = int(total_duration * fps)
    print(f"总帧数: {total_frames}")
    
    # 预加载图像资源
    scene_count = len(set(seg['scene'] for seg in all_segments))
    preload_images(shot, scene_count, width, height)
    
    # Ken Burns效果参数
    zoom_factor_start = 1.0
    zoom_factor_end = 1.05  # 5%缩放
    
    print("开始多线程渲染视频帧...")
    
    # 确定线程数（基于CPU核心数，但不超过合理上限）
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, 8)  # 限制最大线程数为8，避免过度竞争
    batch_size = max(30, total_frames // (max_workers * 4))  # 动态批次大小
    
    # 准备帧批次
    frame_batches = []
    for i in range(0, total_frames, batch_size):
        batch_end = min(i + batch_size, total_frames)
        batch_info = [(j, j / fps) for j in range(i, batch_end)]
        frame_batches.append(batch_info)
    
    print(f"分成 {len(frame_batches)} 个批次进行渲染")
    
    # 存储所有渲染好的帧
    all_rendered_frames = {}
    
    # 使用ThreadPoolExecutor进行多线程渲染
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {
            executor.submit(
                render_frame_batch, 
                batch_info, 
                all_segments, 
                complete_rst_renderer,
                width, height, fps,
                shot,  # 传递shot参数
                zoom_factor_start, zoom_factor_end
            ): i for i, batch_info in enumerate(frame_batches)
        }
        
        # 收集结果
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_frames = future.result()
                # 将批次结果存储到字典中
                for frame_idx, frame in batch_frames:
                    all_rendered_frames[frame_idx] = frame
                
                completed_batches += 1
                progress = completed_batches / len(frame_batches) * 100
                print(f"  渲染进度: {progress:.1f}% (完成批次 {completed_batches}/{len(frame_batches)})")
                
            except Exception as e:
                print(f"批次 {batch_idx} 渲染失败: {e}")
    
    # 创建VideoClip从渲染的帧
    print("创建视频片段...")
    
    def make_frame(t):
        """根据时间返回对应的帧"""
        frame_idx = int(t * fps)
        if frame_idx in all_rendered_frames:
            frame = all_rendered_frames[frame_idx]
            # 转换BGR到RGB（MoviePy需要RGB格式）
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # 如果帧不存在，返回黑色帧
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建视频片段
    from moviepy import VideoClip
    video_clip = VideoClip(make_frame, duration=total_duration)
    
    # 并发处理音频
    print("开始并发处理音频...")
    
    # 计算场景时长
    scene_durations = {}
    for segment in all_segments:
        scene = segment['scene']
        if scene not in scene_durations:
            scene_durations[scene] = segment['end_time']
        else:
            scene_durations[scene] = max(scene_durations[scene], segment['end_time'])
    
    # 并发加载配音
    voice_clips = load_audio_clips_concurrent(shot, scene_durations)
    
    # 处理背景音乐（在单独线程中）
    def process_background_music():
        bg_music_path = "assets/pianai.mp3"
        if os.path.exists(bg_music_path):
            try:
                bg_music_clip = AudioFileClip(bg_music_path)
                print(f"背景音乐时长: {bg_music_clip.duration:.2f}s, 需要时长: {total_duration:.2f}s")
                
                if bg_music_clip.duration < total_duration:
                    # 循环背景音乐
                    loops_needed = int(np.ceil(total_duration / bg_music_clip.duration))
                    print(f"需要循环背景音乐 {loops_needed} 次")
                    bg_clips = [bg_music_clip] * loops_needed
                    bg_music = concatenate_audioclips(bg_clips)
                else:
                    bg_music = bg_music_clip
                
                # 裁剪到准确时长
                bg_music = bg_music.subclipped(0, total_duration).with_volume_scaled(0.3)
                print("背景音乐处理完成")
                return bg_music
                
            except Exception as e:
                print(f"处理背景音乐失败: {e}")
                return None
        return None
    
    # 在单独线程中处理背景音乐
    with ThreadPoolExecutor(max_workers=1) as executor:
        bg_music_future = executor.submit(process_background_music)
        bg_music = bg_music_future.result()
    
    # 合成最终音频并附加到视频
    if voice_clips or bg_music:
        try:
            final_audio_clips = []
            
            if voice_clips:
                # 合成所有配音
                voice_audio = CompositeAudioClip(voice_clips)
                final_audio_clips.append(voice_audio)
                print("配音合成完成")
            
            if bg_music:
                final_audio_clips.append(bg_music)
                print("添加背景音乐")
            
            # 最终音频合成
            if len(final_audio_clips) > 1:
                final_audio = CompositeAudioClip(final_audio_clips)
            else:
                final_audio = final_audio_clips[0]
            
            # 将音频附加到视频
            final_video = video_clip.with_audio(final_audio)
            
            # 直接保存最终视频
            final_output_path = f"videos/{shot}_complete_video_with_audio.mp4"
            os.makedirs("videos", exist_ok=True)
            
            print("开始写入最终视频文件...")
            final_video.write_videofile(
                final_output_path,
                codec='libx264',
                audio_codec='aac',
                fps=fps,
                threads=max_workers  # 使用多线程编码
            )
            
            # 清理资源
            video_clip.close()
            final_video.close()
            if bg_music:
                bg_music.close()
            for clip in voice_clips:
                clip.close()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n=== {shot} 完整视频创建成功（多线程优化版本） ===")
            print(f"输出文件: {final_output_path}")
            print(f"字幕文件: {complete_rst_path}")
            print(f"总时长: {total_duration:.2f} 秒")
            print(f"分辨率: {width}x{height}")
            print(f"帧率: {fps} fps")
            print(f"包含 {len(set(seg['scene'] for seg in all_segments))} 个场景")
            print(f"字幕片段: {len(all_segments)} 个")
            print(f"渲染时间: {total_time:.2f} 秒")
            print(f"渲染效率: {total_frames/total_time:.1f} 帧/秒")
            print(f"使用线程数: {max_workers}")
            
        except Exception as e:
            print(f"视频合成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有音频内容，创建无声视频")
        final_output_path = f"videos/{shot}_complete_video.mp4"
        os.makedirs("videos", exist_ok=True)
        
        video_clip.write_videofile(
            final_output_path,
            codec='libx264',
            fps=fps,
            threads=max_workers
        )
        
        video_clip.close()
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"输出文件: {final_output_path}")
        print(f"渲染时间: {total_time:.2f} 秒")
        print(f"渲染效率: {total_frames/total_time:.1f} 帧/秒")


if __name__ == "__main__":
    # 忽略警告
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 默认渲染shot_02，可以通过命令行参数指定shot
    import sys
    shot = "shot_02"
    if len(sys.argv) > 1:
        shot = sys.argv[1]
    
    create_complete_video(shot)
