#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Storyboard Processor

处理 storyboard.json 文件，生成三个输出：
1. enhanced_flux_image_prompts.json: LLM直接生成的高质量图像prompts
2. storyline.json: 故事线内容的JSON数组
3. 使用Azure AI Speech Service生成storyline的配音文件（默认启用）

Usage:
    python storyboard_processor.py [--shot shot_name] [--no-audio] [--no-enhance] [--assets-dir path]

Examples:
    python storyboard_processor.py                              # 处理所有故事板，自动生成音频和LLM prompts
    python storyboard_processor.py --shot shot_01               # 处理指定故事板，自动生成音频和LLM prompts
    python storyboard_processor.py --no-audio                   # 跳过音频生成
    python storyboard_processor.py --no-enhance                 # 跳过LLM生成，使用基础翻译
    python storyboard_processor.py --shot shot_01 --no-audio --no-enhance  # 只处理基础prompts和字幕
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import re
from datetime import datetime

# 加载环境变量
load_dotenv()

# Azure AI Speech Service 语音合成工具
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    print("警告: azure-cognitiveservices-speech 未安装，语音合成功能将不可用")

# OpenAI LLM API for prompt enhancement
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 未安装，prompt增强功能将不可用")

from enum import Enum

class VoiceType(Enum):
    """支持的中文音色枚举"""
    YUNXI = "zh-CN-YunxiNeural"      # 年轻男性，温和友好
    YUNYE = "zh-CN-YunyeNeural"      # 成熟男性，稳重专业  
    XIAOXIAO = "zh-CN-XiaoxiaoNeural" # 甜美女性，活泼可爱

class AzureSpeechSynthesizer:
    """Azure语音合成器类"""
    
    def __init__(self, speech_key: str, region: str):
        """初始化语音合成器
        
        Args:
            speech_key: Azure Speech Service密钥
            region: 服务区域
        """
        if not AZURE_SPEECH_AVAILABLE:
            raise ImportError("azure-cognitiveservices-speech 未安装，请运行: pip install azure-cognitiveservices-speech")
            
        self.speech_key = speech_key
        self.region = region
        self.base_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
        self.base_config.speech_synthesis_language = "zh-CN"
    
    def synthesize_text(self, text: str, voice: VoiceType, output_file: str = None) -> bytes:
        """合成文本为语音
        
        Args:
            text: 要合成的文本
            voice: 音色类型 (VoiceType枚举)
            output_file: 输出文件名，如果为None则不保存文件
            
        Returns:
            bytes: 音频数据
            
        Raises:
            Exception: 合成失败时抛出异常
        """
        # 创建语音配置
        config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.region)
        config.speech_synthesis_language = "zh-CN"
        config.speech_synthesis_voice_name = voice.value
        
        # 创建合成器（不设置音频输出设备，只生成音频数据）
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        
        # 合成语音
        result = synthesizer.speak_text_async(text).get()
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # 保存文件（如果指定了输出文件）
            if output_file:
                with open(output_file, "wb") as f:
                    f.write(result.audio_data)
            
            return result.audio_data
            
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_msg = f"语音合成失败: {cancellation_details.reason}"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                error_msg += f"\n错误详情: {cancellation_details.error_details}"
            raise Exception(error_msg)
        else:
            raise Exception(f"未知错误: {result.reason}")


class PromptEnhancer:
    """使用LLM增强Flux图像生成prompt的类"""
    
    def __init__(self, api_key: str = None):
        """
        初始化prompt增强器
        
        Args:
            api_key: ARK API密钥，如果为None则从环境变量ARK_API_KEY读取
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 未安装，请运行: pip install --upgrade \"openai>=1.0\"")
        
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        if not self.api_key:
            raise ValueError("未找到 ARK_API_KEY，请在.env文件中添加: ARK_API_KEY=your_api_key_here")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key,
        )
    
    def enhance_flux_prompts_batch(self, visual_descriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        使用LLM基于中文描述直接生成高质量的Flux图像生成prompts
        
        Args:
            visual_descriptions: 包含中文描述的视觉描述列表
            
        Returns:
            List[Dict[str, str]]: 生成完整Flux prompts的视觉描述列表
        """
        try:
            # 构建系统prompt
            system_prompt = """你是一个专业的AI图像生成prompt专家，专门为Flux AI图像生成模型创建高质量的英文prompt。

你的任务是根据给定的中文视觉描述，直接生成专业的英文Flux prompt，要求：
1. 详细准确地描述画面内容
2. 包含专业的艺术风格描述
3. 添加适当的光影效果描述
4. 增强画面构图和视觉效果
5. 保持场景之间的连贯性和风格统一性
6. 使用专业的英文艺术术语

请返回相同JSON格式，为每个场景添加flux_prompt字段，保持其他字段不变。
生成的prompt应该适合Flux AI图像生成模型，风格统一、详细专业。"""

            # 将visual_descriptions转换为JSON字符串
            input_json = json.dumps(visual_descriptions, ensure_ascii=False, indent=2)
            
            # 构建用户prompt
            user_content = f"""请根据以下JSON中每个场景的chinese_description，生成对应的flux_prompt字段，保持JSON格式不变：

{input_json}

要求：
1. 为每个场景添加flux_prompt字段（基于chinese_description生成）
2. 保持scene_number和chinese_description字段不变
3. 确保所有场景的风格保持一致，适合连续的视频场景
4. flux_prompt要详细、专业，适合Flux AI图像生成
5. 包含适当的画质、光影、构图描述
6. 返回完整的JSON格式

示例flux_prompt格式：
"Cinematic shot, high quality, detailed rendering, [具体场景描述], dramatic lighting, masterpiece, best quality, ultra detailed, 8k resolution"
"""
            
            # 调用LLM API
            response = self.client.chat.completions.create(
                model="doubao-1-5-pro-32k-250115",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=8000  # 增加token限制以处理更多内容
            )
            
            enhanced_content = response.choices[0].message.content.strip()
            
            # 尝试解析返回的JSON
            try:
                # 移除可能的markdown代码块标记
                if enhanced_content.startswith("```json"):
                    enhanced_content = enhanced_content[7:]
                if enhanced_content.startswith("```"):
                    enhanced_content = enhanced_content[3:]
                if enhanced_content.endswith("```"):
                    enhanced_content = enhanced_content[:-3]
                
                enhanced_descriptions = json.loads(enhanced_content.strip())
                
                # 验证返回的数据结构
                if isinstance(enhanced_descriptions, list) and len(enhanced_descriptions) == len(visual_descriptions):
                    # 验证每个描述都有flux_prompt字段
                    for i, desc in enumerate(enhanced_descriptions):
                        if not desc.get('flux_prompt'):
                            print(f"⚠️  场景 {desc.get('scene_number', i+1)} 的flux_prompt为空，生成默认prompt")
                            chinese_desc = desc.get('chinese_description', '')
                            desc['flux_prompt'] = f"Cinematic shot, high quality, detailed rendering, {chinese_desc}, dramatic lighting, masterpiece, best quality, ultra detailed, 8k resolution"
                    
                    return enhanced_descriptions
                else:
                    print(f"⚠️  LLM返回的数据结构不正确")
                    return self._generate_fallback_prompts(visual_descriptions)
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  LLM返回的内容无法解析为JSON: {e}")
                print(f"返回内容: {enhanced_content[:500]}...")
                return self._generate_fallback_prompts(visual_descriptions)
            
        except Exception as e:
            print(f"⚠️  LLM prompt生成失败: {e}")
            return self._generate_fallback_prompts(visual_descriptions)
    
    def _generate_fallback_prompts(self, visual_descriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        生成备用的基础Flux prompts
        
        Args:
            visual_descriptions: 中文视觉描述列表
            
        Returns:
            添加了基础flux_prompt的描述列表
        """
        fallback_descriptions = []
        for desc in visual_descriptions:
            chinese_desc = desc.get('chinese_description', '')
            # 使用原有的translate_to_flux_prompt方法作为备用
            fallback_prompt = self.translate_to_flux_prompt(chinese_desc)
            
            fallback_desc = desc.copy()
            fallback_desc['flux_prompt'] = fallback_prompt
            fallback_descriptions.append(fallback_desc)
        
        return fallback_descriptions
    
    def generate_flux_prompts_batch(self, visual_descriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        生成visual descriptions中的flux prompts（直接调用LLM生成方法）
        
        Args:
            visual_descriptions: 包含中文描述的视觉描述列表
            
        Returns:
            List[Dict[str, str]]: 生成flux_prompt后的视觉描述列表
        """
        print(f"🤖 开始使用LLM直接生成 {len(visual_descriptions)} 个图像prompt...")
        
        enhanced_descriptions = self.enhance_flux_prompts_batch(visual_descriptions)
        
        if enhanced_descriptions == visual_descriptions:
            print(f"⚠️  LLM生成失败，使用备用方法")
        else:
            print(f"🎯 所有prompt生成完成！")
        
        return enhanced_descriptions


class StoryboardProcessor:
    """故事板处理器"""
    
    def __init__(self, shot_name: str = None, generate_audio: bool = True, storyboard_dir: str = "storyboard", assets_dir: str = "assets", enhance_prompts: bool = True):
        """
        初始化处理器
        
        Args:
            shot_name: 分集名称，如 'shot_01'，如果为None则处理所有分集
            generate_audio: 是否生成配音文件（默认为True）
            storyboard_dir: storyboard文件目录
            assets_dir: 资源文件目录
            enhance_prompts: 是否使用LLM增强图像prompts（默认为True）
        """
        self.shot_name = shot_name
        self.generate_audio = generate_audio
        self.enhance_prompts = enhance_prompts
        self.storyboard_dir = Path(storyboard_dir)
        self.assets_dir = Path(assets_dir)
        
        # 确保目录存在
        self.assets_dir.mkdir(exist_ok=True)
        
        # Azure Speech Service 配置 - 从环境变量加载
        self.speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'switzerlandnorth')
        
        if self.generate_audio and not self.speech_key:
            print("❌ 警告: 未找到 AZURE_SPEECH_KEY 环境变量，请检查 .env 文件")
            self.generate_audio = False
        
        # 初始化语音合成器
        if self.generate_audio and AZURE_SPEECH_AVAILABLE:
            try:
                self.tts = AzureSpeechSynthesizer(self.speech_key, self.region)
                print("✅ Azure语音合成器初始化成功")
            except Exception as e:
                print(f"❌ Azure语音合成器初始化失败: {e}")
                self.generate_audio = False
        
        # 初始化prompt增强器
        self.prompt_enhancer = None
        if self.enhance_prompts and OPENAI_AVAILABLE:
            try:
                self.prompt_enhancer = PromptEnhancer()
                print("✅ LLM prompt增强器初始化成功")
            except Exception as e:
                print(f"❌ LLM prompt增强器初始化失败: {e}")
                print(f"将跳过prompt增强功能")
                self.enhance_prompts = False
    
    def get_storyboard_files(self) -> List[Path]:
        """
        获取要处理的故事板文件列表
        
        Returns:
            list: 故事板文件路径列表
        """
        storyboard_files = []
        
        # 处理所有故事板文件
        pattern = "*_storyboard.json"
        for storyboard_path in self.storyboard_dir.glob(pattern):
            storyboard_files.append(storyboard_path)
        
        if not storyboard_files:
            print(f"❌ 在 {self.storyboard_dir} 中未找到任何故事板文件")
        
        return storyboard_files
    
    def load_storyboard(self, shot_name: str) -> Dict[str, Any]:
        """
        加载指定分集的 storyboard.json 文件
        
        Args:
            shot_name: 分集名称，如 'shot_01'
            
        Returns:
            故事板数据字典
        """
        storyboard_path = self.storyboard_dir / f"{shot_name}_storyboard.json"
        if not storyboard_path.exists():
            raise FileNotFoundError(f"Storyboard file not found: {storyboard_path}")
        
        with open(storyboard_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def translate_to_flux_prompt(self, chinese_desc: str) -> str:
        """
        将中文视觉描述转换为基础的Flux风格英文prompt（备用方案）
        
        Args:
            chinese_desc: 中文视觉描述
            
        Returns:
            英文Flux风格prompt
        """
        # 简化的备用方案：直接使用中文描述生成基础prompt
        # 这个方法主要在LLM不可用时作为最后备用
        
        # 基本清理：移除一些明显的中文标点
        cleaned_desc = chinese_desc.replace("，", ", ").replace("。", ". ")
        
        # 生成基础的Flux风格prompt
        flux_style_prefix = "Cinematic shot, high quality, detailed rendering, "
        flux_style_suffix = ", dramatic lighting, masterpiece, best quality, ultra detailed, 8k resolution"
        
        # 构建最终的基础Flux prompt
        flux_prompt = f"{flux_style_prefix}{cleaned_desc}{flux_style_suffix}"
        
        return flux_prompt
    
    def generate_comfyui_prompts(self, enhanced_descriptions: List[Dict[str, str]], images_folder: Path) -> None:
        """
        从增强的描述中提取flux_prompt字段，生成ComfyUI专用的prompts数组
        
        Args:
            enhanced_descriptions: 包含flux_prompt的描述列表
            images_folder: 图像文件夹路径
        """
        # 提取所有的flux_prompt
        comfyui_prompts = []
        for desc in enhanced_descriptions:
            flux_prompt = desc.get('flux_prompt', '')
            if flux_prompt:
                comfyui_prompts.append(flux_prompt)
        
        # 保存ComfyUI prompts文件
        comfyui_prompts_path = images_folder / "comfyui_prompts.json"
        with open(comfyui_prompts_path, 'w', encoding='utf-8') as f:
            json.dump(comfyui_prompts, f, ensure_ascii=False, indent=2)
        
        print(f"已生成ComfyUI prompts文件: {comfyui_prompts_path}")
        print(f"包含 {len(comfyui_prompts)} 个图像生成prompt")
    
    def process_visual_descriptions(self, storyboard_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        处理视觉描述，生成包含中文原文的数组（不生成初始英文prompt）
        
        Args:
            storyboard_data: 故事板数据
            
        Returns:
            视觉描述列表
        """
        visual_descriptions = []
        
        # 处理 preview_scenes
        for scene in storyboard_data.get("preview_scenes", []):
            if "visual_description" in scene:
                chinese_desc = scene["visual_description"]
                
                visual_descriptions.append({
                    "scene_number": scene.get("scene_number"),
                    "chinese_description": chinese_desc
                })
        
        # 处理 main_scenes
        for scene in storyboard_data.get("main_scenes", []):
            if "visual_description" in scene:
                chinese_desc = scene["visual_description"]
                
                visual_descriptions.append({
                    "scene_number": scene.get("scene_number"),
                    "chinese_description": chinese_desc
                })
        
        return visual_descriptions
    
    def process_storylines(self, storyboard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理故事线，生成故事线数组
        
        Args:
            storyboard_data: 故事板数据
            
        Returns:
            故事线列表
        """
        storylines = []
        
        # 处理 preview_scenes
        for scene in storyboard_data.get("preview_scenes", []):
            if "storyline" in scene:
                storylines.append({
                    "scene_number": scene.get("scene_number"),
                    "storyline": scene["storyline"]
                })
        
        # 处理 main_scenes
        for scene in storyboard_data.get("main_scenes", []):
            if "storyline" in scene:
                storylines.append({
                    "scene_number": scene.get("scene_number"),
                    "storyline": scene["storyline"]
                })
        
        return storylines
    
    def generate_audio_files(self, storylines: List[Dict[str, Any]], shot_name: str) -> None:
        """
        为storylines生成配音文件
        
        Args:
            storylines: 故事线列表
            shot_name: 分集名称，如 'shot_01'
        """
        if not self.generate_audio or not AZURE_SPEECH_AVAILABLE:
            print("跳过音频生成")
            return
        
        print(f"开始生成 {shot_name} 的配音文件，使用音色: {VoiceType.YUNXI.value}")
        
        # 确保音频目录存在
        audio_folder = self.assets_dir / shot_name / "audios"
        audio_folder.mkdir(parents=True, exist_ok=True)
        
        generated_count = 0
        skipped_count = 0
        
        for storyline in storylines:
            scene_number = storyline["scene_number"]
            text = storyline["storyline"]
            
            # 生成目标音频文件名（符合项目命名规则）
            target_audio_filename = f"{shot_name}_{scene_number}.wav"
            target_audio_path = audio_folder / target_audio_filename
            
            # 检查目标文件是否已存在
            if target_audio_path.exists():
                print(f"⏭️  {shot_name} 场景 {scene_number} 音频已存在，跳过生成: {target_audio_filename}")
                skipped_count += 1
                continue
            
            try:
                print(f"🎵 正在生成 {shot_name} 场景 {scene_number} 的配音...")
                self.tts.synthesize_text(
                    text=text,
                    voice=VoiceType.YUNXI,
                    output_file=str(target_audio_path)
                )
                print(f"✅ 已生成: {target_audio_filename}")
                generated_count += 1
                
            except Exception as e:
                print(f"❌ {shot_name} 场景 {scene_number} 配音生成失败: {e}")
        
        print(f"{shot_name} 配音生成完成! 新生成: {generated_count} 个, 跳过: {skipped_count} 个")
        print(f"音频文件保存在: {audio_folder}")
    
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
            # 首选方案：使用wave库读取WAV文件
            import wave
            with wave.open(audio_path, 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"获取音频时长失败 (wave): {e}")
            try:
                # 备用方案：使用moviepy
                from moviepy.editor import AudioFileClip
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
                audio_clip.close()
                return duration
            except ImportError:
                print("MoviePy未安装，使用默认时长")
            except Exception as e:
                print(f"获取音频时长失败 (moviepy): {e}")
        
        return 3.0  # 默认3秒
    
    def _ensure_audio_files_exist(self, storylines: List[Dict[str, Any]], shot_name: str) -> bool:
        """
        确保所有音频文件都存在，如果不存在则生成
        
        Args:
            storylines: 故事线列表
            shot_name: 分集名称，如 'shot_01'
            
        Returns:
            bool: 所有音频文件是否最终都存在
        """
        print(f"🔍 检查 {shot_name} 的音频文件状态...")
        
        # 音频文件目录
        audio_folder = self.assets_dir / shot_name / "audios"
        audio_folder.mkdir(parents=True, exist_ok=True)
        
        # 检查所有音频文件是否存在
        missing_audio_files = []
        expected_audio_files = []
        
        for storyline in storylines:
            scene_number = storyline["scene_number"]
            audio_filename = f"{shot_name}_{scene_number}.wav"
            audio_path = audio_folder / audio_filename
            expected_audio_files.append(audio_filename)
            
            if not audio_path.exists():
                missing_audio_files.append(audio_filename)
        
        if not missing_audio_files:
            print(f"✅ 所有 {len(expected_audio_files)} 个音频文件都已存在")
            return True
        
        # 有缺失的音频文件，开始生成
        print(f"❌ 发现 {len(missing_audio_files)} 个音频文件缺失:")
        for missing_file in missing_audio_files:
            print(f"   - {missing_file}")
        
        if not self.tts:
            print(f"❌ Azure语音合成器未初始化，无法生成音频文件")
            return False
        
        print(f"🎵 开始生成缺失的音频文件...")
        
        try:
            # 生成音频文件
            self.generate_audio_files(storylines, shot_name)
            print(f"✅ 音频生成完成")
            
            # 验证所有音频文件是否都已生成
            print(f"🔍 验证音频文件生成结果...")
            final_missing_files = []
            for storyline in storylines:
                scene_number = storyline["scene_number"]
                audio_filename = f"{shot_name}_{scene_number}.wav"
                audio_path = audio_folder / audio_filename
                if not audio_path.exists():
                    final_missing_files.append(audio_filename)
            
            if final_missing_files:
                print(f"❌ 仍有 {len(final_missing_files)} 个音频文件生成失败:")
                for missing_file in final_missing_files:
                    print(f"   - {missing_file}")
                return False
            else:
                print(f"✅ 所有音频文件已成功生成并验证完成")
                return True
                
        except Exception as e:
            print(f"❌ 音频生成过程中发生错误: {e}")
            return False
    
    def generate_complete_rst_file(self, storylines: List[Dict[str, Any]], shot_name: str) -> List[Dict]:
        """
        根据storylines生成完整的RST字幕文件
        
        Args:
            storylines: 故事线列表
            shot_name: 分集名称，如 'shot_01'
            
        Returns:
            list: 包含所有片段时间信息的列表
        """
        print(f"📝 开始生成 {shot_name} 的RST字幕文件...")
        
        # 音频文件目录
        audio_folder = self.assets_dir / shot_name / "audios"
        
        # 检查音频文件状态（只做最终确认，不生成）
        missing_audio_files = []
        for storyline in storylines:
            scene_number = storyline["scene_number"]
            audio_filename = f"{shot_name}_{scene_number}.wav"
            audio_path = audio_folder / audio_filename
            if not audio_path.exists():
                missing_audio_files.append(audio_filename)
        
        if missing_audio_files:
            print(f"⚠️  发现 {len(missing_audio_files)} 个音频文件仍然缺失，将使用默认时长 3.0 秒")
        else:
            print(f"✅ 所有音频文件已就绪，使用实际音频时长")
        
        all_segments = []
        current_start_time = 0.0
        
        # 处理每个场景的storyline
        for i, storyline in enumerate(storylines, 1):
            scene_number = storyline["scene_number"]
            storyline_text = storyline["storyline"]
            
            # 构建对应的音频文件路径
            audio_filename = f"{shot_name}_{scene_number}.wav"
            audio_path = audio_folder / audio_filename
            
            # 获取音频时长
            if audio_path.exists():
                scene_duration = self.get_audio_duration(str(audio_path))
            else:
                # 使用默认时长（错误信息已在上面显示过）
                scene_duration = 3.0
            
            # 预处理字幕文本
            clean_text = self.remove_punctuation(storyline_text)
            
            # 文本分块
            chunks = self.split_into_chunks(clean_text)
            
            # 如果没有分块，使用原文
            if not chunks:
                chunks = [clean_text or storyline_text]
            
            # 计算每个块的时间分配
            chunk_duration = scene_duration / len(chunks) if chunks else scene_duration
            
            # 为每个块创建时间段
            for j, chunk in enumerate(chunks):
                segment_start = current_start_time + j * chunk_duration
                segment_end = current_start_time + (j + 1) * chunk_duration
                
                segment_info = {
                    'scene': scene_number,
                    'chunk': j + 1,
                    'text': chunk,
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': chunk_duration
                }
                all_segments.append(segment_info)
                print(f"  {shot_name} 场景{scene_number}_片段{j+1}: {segment_start:.2f}s-{segment_end:.2f}s | {chunk}")
            
            # 更新当前开始时间
            current_start_time += scene_duration
        
        # 生成完整的RST内容
        total_duration = current_start_time
        rst_content = self.generate_complete_rst_content(all_segments, total_duration)
        
        # 保存RST文件到subtitles目录
        subtitles_folder = self.assets_dir / shot_name / "subtitles"
        subtitles_folder.mkdir(parents=True, exist_ok=True)
        final_rst_path = subtitles_folder / f"{shot_name}.rst"
        with open(final_rst_path, 'w', encoding='utf-8') as f:
            f.write(rst_content)
        
        print(f"{shot_name} 完整RST文件已保存到: {final_rst_path}")
        print(f"总时长: {total_duration:.2f}s, 总片段数: {len(all_segments)}")
        
        return all_segments
    
    def generate_complete_rst_content(self, segments: List[Dict], total_duration: float) -> str:
        """
        生成完整的RST内容
        
        Args:
            segments: 所有字幕片段信息
            total_duration: 视频总时长
        
        Returns:
            str: RST格式的内容
        """
        rst_content = f"""完整视频字幕文件
==================

:创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
:总时长: {total_duration:.2f}秒
:片段数量: {len(segments)}
:字体: 微软雅黑
:字体大小: 70px
:描边宽度: 3px

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
   :font_family: C:/Windows/Fonts/msyh.ttc
   :font_size: 70
   :font_color: #FFFFFF
   :stroke_width: 3
   :stroke_color: #000000
   :position: bottom_third

生成信息
--------

:生成工具: storyboard_processor.py
:基于内容: storyline
:分块规则: 最大15字符每块
:时间分配: 基于音频文件时长均匀分配

"""
        
        return rst_content
    
    def process(self, shot_name: str = None) -> None:
        """
        执行完整的处理流程
        
        Args:
            shot_name: 特定分集名称，如 'shot_01'。如果为None，则处理所有分集
        """
        if shot_name:
            # 处理特定分集
            self.process_single_shot(shot_name)
        else:
            # 批量处理所有分集
            storyboard_files = self.get_storyboard_files()
            if not storyboard_files:
                print("未找到任何故事板文件")
                return
            
            print(f"发现 {len(storyboard_files)} 个故事板文件，开始批量处理...")
            for storyboard_file in storyboard_files:
                shot_name = storyboard_file.stem.replace('_storyboard', '')
                self.process_single_shot(shot_name)
                print("-" * 50)
    
    def process_single_shot(self, shot_name: str) -> None:
        """
        处理单个分集的故事板
        
        Args:
            shot_name: 分集名称，如 'shot_01'
        """
        print(f"正在处理故事板: {shot_name}")
        
        # 加载故事板数据
        storyboard_data = self.load_storyboard(shot_name)
        
        # 处理视觉描述（只提取中文描述）
        visual_descriptions = self.process_visual_descriptions(storyboard_data)
        images_folder = self.assets_dir / shot_name / "images"
        images_folder.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经存在增强后的prompts文件
        flux_prompts_path = images_folder / "flux_prompts.json"
        
        if flux_prompts_path.exists() and not self.enhance_prompts:
            # 如果文件存在但禁用了增强功能，直接加载现有文件
            print(f"📄 发现现有的flux prompts文件: {flux_prompts_path}")
            with open(flux_prompts_path, 'r', encoding='utf-8') as f:
                enhanced_descriptions = json.load(f)
            print(f"已加载现有的flux图像生成prompts")
            
            # 检查并生成ComfyUI prompts文件
            comfyui_prompts_path = images_folder / "comfyui_prompts.json"
            if not comfyui_prompts_path.exists():
                self.generate_comfyui_prompts(enhanced_descriptions, images_folder)
            
        elif flux_prompts_path.exists() and self.enhance_prompts:
            # 如果文件存在且启用了增强功能，询问是否跳过
            print(f"📄 发现现有的flux prompts文件: {flux_prompts_path}")
            print(f"🔄 跳过LLM调用，使用现有文件")
            with open(flux_prompts_path, 'r', encoding='utf-8') as f:
                enhanced_descriptions = json.load(f)
            print(f"已加载现有的flux图像生成prompts")
            
            # 检查并生成ComfyUI prompts文件
            comfyui_prompts_path = images_folder / "comfyui_prompts.json"
            if not comfyui_prompts_path.exists():
                self.generate_comfyui_prompts(enhanced_descriptions, images_folder)
            
        else:
            # 需要生成新的prompts
            if self.enhance_prompts and self.prompt_enhancer:
                # 使用LLM直接生成prompts
                enhanced_descriptions = self.prompt_enhancer.generate_flux_prompts_batch(visual_descriptions)
                
                # 保存生成后的prompts
                with open(flux_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_descriptions, f, ensure_ascii=False, indent=2)
                print(f"已保存LLM生成的图像prompts: {flux_prompts_path}")
                
                # 生成ComfyUI专用的prompts文件
                self.generate_comfyui_prompts(enhanced_descriptions, images_folder)
                
            else:
                # 如果禁用了增强或初始化失败，使用备用方法
                print("⚠️  LLM生成被禁用或初始化失败，使用备用翻译方法")
                enhanced_descriptions = []
                for desc in visual_descriptions:
                    chinese_desc = desc.get('chinese_description', '')
                    fallback_prompt = self.translate_to_flux_prompt(chinese_desc)
                    
                    enhanced_desc = desc.copy()
                    enhanced_desc['flux_prompt'] = fallback_prompt
                    enhanced_descriptions.append(enhanced_desc)
                
                # 保存备用prompts到基础文件
                basic_prompts_path = images_folder / "basic_flux_prompts.json"
                with open(basic_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_descriptions, f, ensure_ascii=False, indent=2)
                print(f"已保存基础图像生成prompts: {basic_prompts_path}")
                
                # 生成ComfyUI专用的prompts文件
                self.generate_comfyui_prompts(enhanced_descriptions, images_folder)
        
        # 更新visual_descriptions为最终版本
        visual_descriptions = enhanced_descriptions
        
        # 处理故事线并保存到subtitles目录
        storylines = self.process_storylines(storyboard_data)
        subtitles_folder = self.assets_dir / shot_name / "subtitles"
        subtitles_folder.mkdir(parents=True, exist_ok=True)
        storyline_path = subtitles_folder / f"{shot_name}_storylines.json"
        with open(storyline_path, 'w', encoding='utf-8') as f:
            json.dump(storylines, f, ensure_ascii=False, indent=2)
        print(f"已保存故事线到: {storyline_path}")
        
        # 步骤1：检查并生成音频文件
        if self.generate_audio:
            self._ensure_audio_files_exist(storylines, shot_name)
        
        # 步骤2：生成字幕RST文件（此时音频文件应该已完整）
        subtitle_segments = self.generate_complete_rst_file(storylines, shot_name)
        print(f"✅ 字幕文件生成完成: {len(subtitle_segments)} 个片段")
        
        print(f"{shot_name} 处理完成!")
        print(f"图像prompts已保存到: {images_folder}")
        print(f"字幕文件已保存到: {subtitles_folder}")
        print(f"音频文件目录: {self.assets_dir / shot_name / 'audios'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理故事板文件，生成视觉描述prompt和故事线')
    parser.add_argument('--shot', help='指定处理的分集，如 shot_01。如果不指定，则批量处理所有分集')
    parser.add_argument('--no-audio', action='store_true', 
                       help='跳过配音文件生成（默认会自动生成音频）')
    parser.add_argument('--no-enhance', action='store_true',
                       help='跳过LLM prompt增强（默认会使用LLM增强图像prompts）')
    parser.add_argument('--assets-dir', default='assets', 
                       help='资源目录路径（默认：assets）')
    
    args = parser.parse_args()
    
    # 默认生成音频，除非指定 --no-audio
    generate_audio = not args.no_audio
    # 默认增强prompt，除非指定 --no-enhance
    enhance_prompts = not args.no_enhance
    
    try:
        processor = StoryboardProcessor(
            shot_name=args.shot, 
            generate_audio=generate_audio, 
            storyboard_dir="storyboard",
            assets_dir=args.assets_dir,
            enhance_prompts=enhance_prompts
        )
        processor.process(args.shot)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
