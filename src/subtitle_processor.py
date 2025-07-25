"""
字幕处理系统
支持字幕预处理、RST文件生成和样式渲染
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import wave
from moviepy import AudioFileClip


class SubtitleProcessor:
    """字幕处理核心类"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化字幕处理器"""
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
    def remove_punctuation(self, text: str) -> str:
        """去除标点符号"""
        # 定义中英文标点符号模式
        punctuation_pattern = r'[，。！？：；""''「」『』（）【】《》〈〉、,.!?:;"\'()\\[\\]{}<>/|~`@#$%^&*+=_-]'
        # 去除标点符号，用空格替换
        clean_text = re.sub(punctuation_pattern, ' ', text)
        # 去除多余的空格
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    def split_into_chunks(self, text: str, max_chars_per_chunk: int = 15) -> List[str]:
        """将文本分块，每块不超过指定字符数"""
        # 先按空格分割成词
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            # 检查添加这个词后是否会超过限制
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(test_chunk) <= max_chars_per_chunk:
                # 不超过限制，添加这个词
                current_chunk = test_chunk
            else:
                # 超过限制，保存当前块并开始新块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = word
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频文件的时长（秒）"""
        try:
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0
    
    def calculate_timing(self, chunks: List[str], total_duration: float) -> List[Dict]:
        """根据音频时长计算每个块的时间"""
        if not chunks or total_duration <= 0:
            return []
            
        chunk_duration = total_duration / len(chunks)
        timing_info = []
        
        for i, chunk in enumerate(chunks):
            start_time = i * chunk_duration
            end_time = (i + 1) * chunk_duration
            
            timing_info.append({
                'text': chunk,
                'start_time': start_time,
                'end_time': end_time,
                'duration': chunk_duration
            })
            
        return timing_info
    
    def generate_rst_content(self, timing_info: List[Dict], title: str = "字幕文件") -> str:
        """生成RST格式的字幕文件"""
        rst_content = f"""
{title}
{"=" * len(title)}

:作者: 自动生成
:日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
:版本: 1.0

字幕信息
--------

本文件包含视频字幕的时间和内容信息。

.. contents:: 目录
   :local:

字幕片段
--------

"""
        
        for i, info in enumerate(timing_info, 1):
            section_title = f"片段 {i}"
            rst_content += f"""
{section_title}
{'^' * len(section_title)}

:开始时间: {info['start_time']:.2f}秒
:结束时间: {info['end_time']:.2f}秒
:持续时间: {info['duration']:.2f}秒
:内容: {info['text']}

.. parsed-literal::

   时间: {info['start_time']:.2f}s - {info['end_time']:.2f}s
   内容: "{info['text']}"

"""
        
        # 添加样式信息
        rst_content += """
样式设置
--------

字体样式
^^^^^^^^

:字体: 微软雅黑 (Microsoft YaHei)
:字体大小: 70px
:字体颜色: 白色 (#FFFFFF)
:描边颜色: 黑色 (#000000)
:描边宽度: 3px

位置设置
^^^^^^^^

:水平位置: 居中对齐
:垂直位置: 屏幕下方1/3区域
:行间距: 80px (字体大小 + 10px)

渲染配置
^^^^^^^^

.. code-block:: json

   {
     "font_family": "Microsoft YaHei",
     "font_size": 70,
     "font_color": "#FFFFFF",
     "stroke_color": "#000000",
     "stroke_width": 3,
     "position": {
       "horizontal": "center",
       "vertical": "bottom_third"
     },
     "line_spacing": 80
   }

"""
        
        return rst_content
    
    def save_rst_file(self, rst_content: str, output_path: str):
        """保存RST文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rst_content)
        print(f"RST文件已保存到: {output_path}")
    
    def process_subtitle(self, text: str, audio_path: str, output_rst_path: str) -> List[Dict]:
        """完整的字幕处理流程"""
        # 1. 去除标点符号
        clean_text = self.remove_punctuation(text)
        
        # 2. 分块
        max_chars = self.config.get('subtitle_settings', {}).get('max_chars_per_line', 15)
        chunks = self.split_into_chunks(clean_text, max_chars)
        
        # 3. 获取音频时长
        audio_duration = self.get_audio_duration(audio_path)
        
        # 4. 计算时间
        timing_info = self.calculate_timing(chunks, audio_duration)
        
        # 5. 生成RST内容
        rst_content = self.generate_rst_content(timing_info, f"字幕文件 - {os.path.basename(audio_path)}")
        
        # 6. 保存RST文件
        self.save_rst_file(rst_content, output_rst_path)
        
        return timing_info
    
    def generate_complete_rst_file(self, captions_list, audio_files_list, output_path):
        """
        生成完整的RST字幕文件，包含所有场景的字幕
        
        Args:
            captions_list: 字幕文本列表
            audio_files_list: 音频文件路径列表
            output_path: 输出RST文件路径
        
        Returns:
            list: 包含所有片段时间信息的列表
        """
        print("开始生成完整RST字幕文件...")
        
        all_segments = []
        current_start_time = 0.0
        
        # 处理每个场景
        for i, (caption_text, audio_path) in enumerate(zip(captions_list, audio_files_list), 1):
            # 获取音频时长
            if os.path.exists(audio_path):
                try:
                    from moviepy import AudioFileClip
                    audio_clip = AudioFileClip(audio_path)
                    scene_duration = audio_clip.duration
                    audio_clip.close()
                except Exception as e:
                    scene_duration = 3.0
            else:
                print(f"  警告：音频文件不存在 {audio_path}")
                scene_duration = 3.0
            
            # 预处理字幕文本
            clean_text = self.remove_punctuation(caption_text)
            
            # 文本分块
            chunks = self.split_into_chunks(clean_text)
            
            # 计算每个块的时间分配
            chunk_duration = scene_duration / len(chunks) if chunks else scene_duration
            
            # 为每个块创建时间段
            for j, chunk in enumerate(chunks):
                segment_start = current_start_time + j * chunk_duration
                segment_end = current_start_time + (j + 1) * chunk_duration
                
                segment_info = {
                    'scene': i,
                    'chunk': j + 1,
                    'text': chunk,
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': chunk_duration
                }
                all_segments.append(segment_info)
            
            # 更新当前开始时间
            current_start_time += scene_duration
        
        # 生成完整的RST内容
        total_duration = current_start_time
        rst_content = self.generate_complete_rst_content(all_segments, total_duration)
        
        # 保存RST文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rst_content)
        
        print(f"完整RST文件已保存到: {output_path}")
        print(f"总时长: {total_duration:.2f}s, 总片段数: {len(all_segments)}")
        
        return all_segments
    
    def generate_complete_rst_content(self, segments, total_duration):
        """
        生成完整的RST内容
        
        Args:
            segments: 所有字幕片段信息
            total_duration: 视频总时长
        
        Returns:
            str: RST格式的内容
        """
        from datetime import datetime
        
        rst_content = f"""完整视频字幕文件
==================

:创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
:总时长: {total_duration:.2f}秒
:片段数量: {len(segments)}
:字体: {self.config['subtitle_settings']['font_family']}
:字体大小: {self.config['subtitle_settings']['font_size']}
:描边宽度: {self.config['subtitle_settings']['stroke_width']}

字幕内容
--------

"""
        
        for segment in segments:
            rst_content += f"""
.. subtitle:: 场景{segment['scene']:02d}_片段{segment['chunk']:02d}
   :start_time: {segment['start_time']:.2f}
   :end_time: {segment['end_time']:.2f}
   :duration: {segment['duration']:.2f}
   :text: {segment['text']}

"""
        
        rst_content += f"""

样式配置
--------

.. style_config::
   :font_family: {self.config['subtitle_settings']['font_family']}
   :font_size: {self.config['subtitle_settings']['font_size']}
   :font_color: {self.config['subtitle_settings']['font_color']}
   :stroke_width: {self.config['subtitle_settings']['stroke_width']}
   :stroke_color: {self.config['subtitle_settings']['stroke_color']}
   :position: {self.config['subtitle_settings']['position']}

"""
        
        return rst_content


class SubtitleRenderer:
    """从RST文件渲染字幕到视频的类"""
    
    def __init__(self, rst_path: str):
        """初始化渲染器"""
        self.rst_path = rst_path
        self.timing_info = []
        self.style_config = {}
        self.parse_rst_file()
    
    def parse_rst_file(self):
        """解析RST文件提取字幕信息"""
        try:
            with open(self.rst_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析字幕片段信息 - 使用正确的RST指令格式
            # 匹配 .. subtitle:: 指令块
            pattern = r'.. subtitle::\s*(.+?)\s+:start_time:\s*([0-9.]+)\s+:end_time:\s*([0-9.]+)\s+:duration:\s*([0-9.]+)\s+:text:\s*(.+?)(?=\n\n|\n.. subtitle::|\Z)'
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                scene_name = match[0].strip()
                start_time = float(match[1])
                end_time = float(match[2])
                duration = float(match[3])
                text = match[4].strip()
                
                self.timing_info.append({
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'scene': scene_name
                })
            
            # 如果解析失败，尝试备用模式
            if not self.timing_info:
                # 逐行解析模式
                lines = content.split('\n')
                current_subtitle = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('.. subtitle::'):
                        # 新的字幕块开始
                        if current_subtitle and 'text' in current_subtitle:
                            self.timing_info.append(current_subtitle)
                        current_subtitle = {'scene': line.replace('.. subtitle::', '').strip()}
                    elif line.startswith(':start_time:'):
                        current_subtitle['start_time'] = float(line.split(':')[-1].strip())
                    elif line.startswith(':end_time:'):
                        current_subtitle['end_time'] = float(line.split(':')[-1].strip())
                    elif line.startswith(':duration:'):
                        current_subtitle['duration'] = float(line.split(':')[-1].strip())
                    elif line.startswith(':text:'):
                        current_subtitle['text'] = line.split(':', 2)[-1].strip()
                
                # 添加最后一个字幕
                if current_subtitle and 'text' in current_subtitle:
                    self.timing_info.append(current_subtitle)
            
            # 设置默认样式配置
            self.style_config = {
                'font_family': 'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                'font_size': 70,
                'font_color': '#FFFFFF',
                'stroke_color': '#000000',
                'stroke_width': 3,
                'position': {
                    'horizontal': 'center',
                    'vertical': 'bottom_third'
                },
                'line_spacing': 80
            }
            
        except Exception as e:
            print(f"解析RST文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def get_subtitle_at_time(self, current_time: float) -> str:
        """根据时间获取对应的字幕文本"""
        for info in self.timing_info:
            if info['start_time'] <= current_time <= info['end_time']:
                return info['text']
        return ""
    
    def get_timing_info(self) -> List[Dict]:
        """获取所有时间信息"""
        return self.timing_info
    
    def get_style_config(self) -> Dict:
        """获取样式配置"""
        return self.style_config


def main():
    """主函数"""
    # 创建字幕处理器实例
    processor = SubtitleProcessor("config.json")
    
    # 加载字幕数据
    with open('assets/caption.json', 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    # 处理第一条字幕
    if captions and len(captions) > 0:
        first_caption = captions[0]
        audio_path = "assets/audios/01.wav"
        output_rst = "assets/subtitles/shot_01.rst"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_rst), exist_ok=True)
        
        # 处理字幕
        timing_info = processor.process_subtitle(first_caption, audio_path, output_rst)
        
        print("\n" + "="*50)
        print("字幕处理结果:")
        print("="*50)
        for i, info in enumerate(timing_info, 1):
            print(f"片段{i}: {info['start_time']:.2f}s-{info['end_time']:.2f}s | {info['text']}")
        
        print(f"\nRST文件已生成: {output_rst}")
        print("现在可以使用 SubtitleRenderer 来渲染字幕到视频中")


if __name__ == "__main__":
    main()
