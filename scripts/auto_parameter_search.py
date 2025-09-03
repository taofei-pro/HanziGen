#!/usr/bin/env python3
"""
自动化参数搜索脚本
自动执行训练流程，寻找最佳参数配置
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoParameterSearch:
    def __init__(self, target_font_path: str, max_iterations: int = 20):
        self.target_font_path = target_font_path
        self.target_font_name = Path(target_font_path).stem
        self.max_iterations = max_iterations
        self.search_history = []
        self.best_score = float('inf')
        self.best_config = None
        
        # 目标分数
        self.target_scores = {
            'PSNR': 20.0,
            'SSIM': 0.95,
            'LPIPS': 0.05,
            'FID': 5.0
        }
        
        # 当前最佳配置（基于批次#8）
        self.current_config = {
            'vqvae': {
                'encoder_channels': 112,
                'latent_dim': 4,
                'codebook_size': 192
            },
            'ldm': {
                'unet_channels': 96,
                'time_emb_dim': 1792,
                'time_steps': 1400
            },
            'training': {
                'vqvae_batch_size': 16,
                'vqvae_epochs': 200,
                'vqvae_lr': 6e-4,
                'ldm_batch_size': 32,
                'ldm_epochs': 500,
                'ldm_lr': 2e-4,
                'sample_steps': 120
            }
        }
    
    def generate_next_config(self, current_results: Dict) -> Dict:
        """基于当前结果生成下一组参数配置"""
        logger.info("生成下一组参数配置...")
        
        # 分析当前结果，决定调整策略
        psnr = current_results.get('PSNR', 0)
        ssim = current_results.get('SSIM', 0)
        lpips = current_results.get('LPIPS', 1)
        fid = current_results.get('FID', 100)
        
        # 计算综合分数（越低越好）
        current_score = self.calculate_composite_score(psnr, ssim, lpips, fid)
        
        # 如果当前结果更好，更新最佳配置
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_config = self.current_config.copy()
            logger.info(f"发现更好的配置！综合分数: {current_score:.4f}")
        
        # 基于当前结果决定调整方向
        new_config = self.current_config.copy()
        
        # 如果PSNR较低，增加模型容量
        if psnr < 15:
            new_config['vqvae']['encoder_channels'] = min(128, new_config['vqvae']['encoder_channels'] + 8)
            new_config['vqvae']['latent_dim'] = min(5, new_config['vqvae']['latent_dim'] + 1)
        
        # 如果SSIM较低，增加训练轮数
        if ssim < 0.85:
            new_config['training']['vqvae_epochs'] = min(300, new_config['training']['vqvae_epochs'] + 50)
            new_config['training']['ldm_epochs'] = min(700, new_config['training']['ldm_epochs'] + 100)
        
        # 如果LPIPS较高，优化学习率
        if lpips > 0.15:
            new_config['training']['vqvae_lr'] = max(3e-4, new_config['training']['vqvae_lr'] * 0.9)
            new_config['training']['ldm_lr'] = max(1e-4, new_config['training']['ldm_lr'] * 0.9)
        
        # 如果FID较高，增加采样步数
        if fid > 50:
            new_config['training']['sample_steps'] = min(200, new_config['training']['sample_steps'] + 20)
        
        # 更新当前配置
        self.current_config = new_config
        
        logger.info(f"新配置: VQ-VAE通道{new_config['vqvae']['encoder_channels']}, "
                   f"潜在维度{new_config['vqvae']['latent_dim']}, "
                   f"训练轮数{new_config['training']['vqvae_epochs']}")
        
        return new_config
    
    def calculate_composite_score(self, psnr: float, ssim: float, lpips: float, fid: float) -> float:
        """计算综合分数（越低越好）"""
        # 归一化各项指标到0-1范围
        psnr_norm = max(0, (psnr - 8) / (25 - 8))  # 8-25范围
        ssim_norm = max(0, (ssim - 0.5) / (0.98 - 0.5))  # 0.5-0.98范围
        lpips_norm = max(0, (lpips - 0.05) / (0.3 - 0.05))  # 0.05-0.3范围
        fid_norm = max(0, (fid - 5) / (100 - 5))  # 5-100范围
        
        # 综合分数（加权平均）
        composite_score = (
            0.3 * (1 - psnr_norm) +      # PSNR权重30%
            0.3 * (1 - ssim_norm) +      # SSIM权重30%
            0.2 * lpips_norm +           # LPIPS权重20%
            0.2 * fid_norm               # FID权重20%
        )
        
        return composite_score
    
    def update_config_files(self, config: Dict) -> None:
        """更新配置文件"""
        logger.info("更新配置文件...")
        
        # 更新VQ-VAE配置
        vqvae_config_path = "configs/vqvae_config.py"
        self.update_vqvae_config(vqvae_config_path, config['vqvae'])
        
        # 更新LDM配置
        ldm_config_path = "configs/ldm_config.py"
        self.update_ldm_config(ldm_config_path, config['ldm'])
        
        # 更新训练脚本
        self.update_training_scripts(config['training'])
        
        logger.info("配置文件更新完成")
    
    def update_vqvae_config(self, config_path: str, config: Dict) -> None:
        """更新VQ-VAE配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换配置值
        content = content.replace(
            f"encoder_base_channels: int = {self.current_config['vqvae']['encoder_channels']}",
            f"encoder_base_channels: int = {config['encoder_channels']}"
        )
        content = content.replace(
            f"latent_dim: int = {self.current_config['vqvae']['latent_dim']}",
            f"latent_dim: int = {config['latent_dim']}"
        )
        content = content.replace(
            f"codebook_size: int = {self.current_config['vqvae']['codebook_size']}",
            f"codebook_size: int = {config['codebook_size']}"
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def update_ldm_config(self, config_path: str, config: Dict) -> None:
        """更新LDM配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换配置值
        content = content.replace(
            f"unet_base_channels: int = {self.current_config['ldm']['unet_channels']}",
            f"unet_base_channels: int = {config['unet_channels']}"
        )
        content = content.replace(
            f"time_emb_dim: int = {self.current_config['ldm']['time_emb_dim']}",
            f"time_emb_dim: int = {config['time_emb_dim']}"
        )
        content = content.replace(
            f"time_steps: int = {self.current_config['ldm']['time_steps']}",
            f"time_steps: int = {config['time_steps']}"
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def update_training_scripts(self, config: Dict) -> None:
        """更新训练脚本"""
        # 更新VQ-VAE训练脚本
        vqvae_script = "scripts/train_vqvae.sh"
        with open(vqvae_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace(
            f"BATCH_SIZE={self.current_config['training']['vqvae_batch_size']}",
            f"BATCH_SIZE={config['vqvae_batch_size']}"
        )
        content = content.replace(
            f"NUM_EPOCHS={self.current_config['training']['vqvae_epochs']}",
            f"NUM_EPOCHS={config['vqvae_epochs']}"
        )
        content = content.replace(
            f"LEARNING_RATE={self.current_config['training']['vqvae_lr']}",
            f"LEARNING_RATE={config['vqvae_lr']}"
        )
        
        with open(vqvae_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 更新LDM训练脚本
        ldm_script = "scripts/train_ldm.sh"
        with open(ldm_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace(
            f"BATCH_SIZE={self.current_config['training']['ldm_batch_size']}",
            f"BATCH_SIZE={config['ldm_batch_size']}"
        )
        content = content.replace(
            f"NUM_EPOCHS={self.current_config['training']['ldm_epochs']}",
            f"NUM_EPOCHS={config['ldm_epochs']}"
        )
        content = content.replace(
            f"LEARNING_RATE={self.current_config['training']['ldm_lr']}",
            f"LEARNING_RATE={config['ldm_lr']}"
        )
        content = content.replace(
            f"SAMPLE_STEPS={self.current_config['training']['sample_steps']}",
            f"SAMPLE_STEPS={config['sample_steps']}"
        )
        
        with open(ldm_script, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def execute_training(self) -> Dict:
        """执行训练流程"""
        logger.info("开始执行训练流程...")
        
        start_time = time.time()
        
        try:
            # 执行完整训练流程，实时显示输出
            logger.info("开始执行训练流程，实时输出如下：")
            logger.info("=" * 60)
            
            # 使用Popen实时显示输出
            process = subprocess.Popen(
                ['bash', 'scripts/full_training_pipeline.sh'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取并显示输出
            output_lines = []
            start_time_training = time.time()
            timeout_seconds = 21600  # 6小时超时
            
            while True:
                # 检查超时
                if time.time() - start_time_training > timeout_seconds:
                    logger.error("训练超时（6小时），强制终止")
                    process.terminate()
                    process.wait()
                    return {}
                
                # 尝试读取输出，设置非阻塞读取
                try:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output = output.strip()
                        if output:  # 忽略空行
                            print(output)  # 实时显示在控制台
                            output_lines.append(output)
                            logger.info(output)  # 同时记录到日志
                except Exception as e:
                    logger.error(f"读取输出时出错: {e}")
                    break
            
            # 等待进程完成
            return_code = process.poll()
            
            if return_code != 0:
                logger.error(f"训练失败，返回码: {return_code}")
                return {}
            
            logger.info("=" * 60)
            logger.info("训练流程执行完成")
            
            # 解析训练结果
            training_results = self.parse_training_results()
            
            training_time = time.time() - start_time
            logger.info(f"训练完成，耗时: {training_time/3600:.2f}小时")
            
            return training_results
            
        except subprocess.TimeoutExpired:
            logger.error("训练超时")
            return {}
        except Exception as e:
            logger.error(f"训练执行错误: {e}")
            return {}
    
    def parse_training_results(self) -> Dict:
        """解析训练结果"""
        logger.info("解析训练结果...")
        
        # 查找最新的评估结果文件
        sample_root = f"samples_{self.target_font_name}/"
        eval_dir = os.path.join(sample_root, "eval_outputs")
        
        if not os.path.exists(eval_dir):
            logger.error(f"评估目录不存在: {eval_dir}")
            return {}
        
        # 查找最新的评估结果
        eval_files = []
        for root, dirs, files in os.walk(eval_dir):
            for file in files:
                if file.endswith('.txt') and 'metrics' in file.lower():
                    eval_files.append(os.path.join(root, file))
        
        if not eval_files:
            logger.error("未找到评估结果文件")
            return {}
        
        # 按修改时间排序，取最新的
        eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_eval_file = eval_files[0]
        
        logger.info(f"解析评估文件: {latest_eval_file}")
        
        try:
            with open(latest_eval_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析评估指标
            results = {}
            
            # 查找PSNR
            psnr_match = re.search(r'PSNR\s*\│\s*([\d.]+)', content)
            if psnr_match:
                results['PSNR'] = float(psnr_match.group(1))
            
            # 查找SSIM
            ssim_match = re.search(r'SSIM\s*\│\s*([\d.]+)', content)
            if ssim_match:
                results['SSIM'] = float(ssim_match.group(1))
            
            # 查找LPIPS
            lpips_match = re.search(r'LPIPS\s*\│\s*([\d.]+)', content)
            if lpips_match:
                results['LPIPS'] = float(lpips_match.group(1))
            
            # 查找FID
            fid_match = re.search(r'FID\s*\│\s*([\d.]+)', content)
            if fid_match:
                results['FID'] = float(fid_match.group(1))
            
            # 验证是否成功解析
            if len(results) == 4:
                logger.info(f"成功解析评估结果: PSNR={results['PSNR']:.4f}, "
                           f"SSIM={results['SSIM']:.4f}, "
                           f"LPIPS={results['LPIPS']:.4f}, "
                           f"FID={results['FID']:.4f}")
                return results
            else:
                logger.error(f"解析不完整，只找到 {len(results)}/4 个指标")
                logger.error(f"找到的指标: {list(results.keys())}")
                return {}
                
        except Exception as e:
            logger.error(f"解析评估文件失败: {e}")
            return {}
    
    def should_stop(self, current_results: Dict, iteration: int) -> bool:
        """判断是否应该停止搜索"""
        # 达到目标分数
        if (current_results.get('PSNR', 0) >= self.target_scores['PSNR'] and
            current_results.get('SSIM', 0) >= self.target_scores['SSIM'] and
            current_results.get('LPIPS', 0) <= self.target_scores['LPIPS'] and
            current_results.get('FID', 100) <= self.target_scores['FID']):
            logger.info("达到目标分数，停止搜索！")
            return True
        
        # 达到最大迭代次数
        if iteration >= self.max_iterations:
            logger.info(f"达到最大迭代次数 {self.max_iterations}，停止搜索")
            return True
        
        # 连续多次无改善
        if len(self.search_history) >= 5:
            recent_scores = [h['score'] for h in self.search_history[-5:]]
            if max(recent_scores) - min(recent_scores) < 0.01:
                logger.info("连续5次无显著改善，停止搜索")
                return True
        
        return False
    
    def run_search(self) -> None:
        """运行自动化参数搜索"""
        logger.info(f"开始自动化参数搜索，最大迭代次数: {self.max_iterations}")
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"迭代 {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*50}")
            
            # 执行训练
            results = self.execute_training()
            if not results:
                logger.error("训练失败，跳过此次迭代")
                continue
            
            # 记录结果
            score = self.calculate_composite_score(
                results['PSNR'], results['SSIM'], 
                results['LPIPS'], results['FID']
            )
            
            search_record = {
                'iteration': iteration + 1,
                'config': self.current_config.copy(),
                'results': results,
                'score': score,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.search_history.append(search_record)
            
            # 保存搜索历史
            self.save_search_history()
            
            # 检查是否应该停止
            if self.should_stop(results, iteration + 1):
                break
            
            # 生成下一组参数
            next_config = self.generate_next_config(results)
            self.update_config_files(next_config)
            
            # 等待一段时间再开始下一轮
            logger.info("等待30秒后开始下一轮训练...")
            time.sleep(30)
        
        # 搜索完成，显示结果
        self.show_final_results()
    
    def update_training_log(self, iteration: int, config: Dict, results: Dict, score: float) -> None:
        """自动更新训练日志"""
        log_file = "TRAINING_LOG.md"
        
        if not os.path.exists(log_file):
            logger.warning("训练日志文件不存在，跳过更新")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建新的日志条目
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            new_entry = f"""
### 训练批次 #{iteration + 8} - 自动化搜索迭代 {iteration + 1}
**模型配置：**
- VQ-VAE: `encoder_channels={config['vqvae']['encoder_channels']}, latent_dim={config['vqvae']['latent_dim']}, codebook_size={config['vqvae']['codebook_size']}`
- LDM: `unet_channels={config['ldm']['unet_channels']}, time_emb_dim={config['ldm']['time_emb_dim']}, time_steps={config['ldm']['time_steps']}`

**训练参数：**
- VQ-VAE: `batch_size={config['training']['vqvae_batch_size']}, epochs={config['training']['vqvae_epochs']}, lr={config['training']['vqvae_lr']}`
- LDM: `batch_size={config['training']['ldm_batch_size']}, epochs={config['training']['ldm_epochs']}, lr={config['training']['ldm_lr']}, sample_steps={config['training']['sample_steps']}`

**优化策略：**
- **自动化参数搜索**：基于前一轮结果智能调整参数
- **综合评分优化**：综合分数 {score:.4f}（越低越好）
- **迭代优化**：第{iteration + 1}次自动化搜索

**评估指标：**
| 指标 | 分数 | 目标分数 | 状态 | 变化 |
|------|------|----------|------|------|
| PSNR | {results.get('PSNR', 0):.4f} | >20 | {'✅' if results.get('PSNR', 0) >= 20 else '❌'} | - |
| SSIM | {results.get('SSIM', 0):.4f} | >0.95 | {'✅' if results.get('SSIM', 0) >= 0.95 else '❌'} | - |
| LPIPS | {results.get('LPIPS', 0):.4f} | <0.05 | {'✅' if results.get('LPIPS', 0) <= 0.05 else '❌'} | - |
| FID | {results.get('FID', 0):.4f} | <5 | {'✅' if results.get('FID', 0) <= 5 else '❌'} | - |

**自动化分析：**
- **综合评分**：{score:.4f}
- **搜索状态**：第{iteration + 1}次迭代
- **时间戳**：{timestamp}

**下一步策略：**
- 系统将自动分析结果并生成下一组参数
- 继续优化直到达到目标分数或无法改善

---
"""
            
            # 在文件开头插入新条目
            if "## 训练批次记录" in content:
                # 找到训练批次记录部分，在其后插入
                insert_pos = content.find("## 训练批次记录") + len("## 训练批次记录")
                content = content[:insert_pos] + new_entry + content[insert_pos:]
            else:
                # 如果没有找到，在文件开头插入
                content = new_entry + content
            
            # 写回文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"训练日志已更新: {log_file}")
            
        except Exception as e:
            logger.error(f"更新训练日志失败: {e}")
    
    def save_search_history(self) -> None:
        """保存搜索历史"""
        history_file = f"search_history_{self.target_font_name}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.search_history, f, indent=2, ensure_ascii=False)
        logger.info(f"搜索历史已保存到 {history_file}")
        
        # 同时更新训练日志
        if self.search_history:
            latest_record = self.search_history[-1]
            self.update_training_log(
                latest_record['iteration'] - 1,
                latest_record['config'],
                latest_record['results'],
                latest_record['score']
            )
    
    def show_final_results(self) -> None:
        """显示最终结果"""
        logger.info(f"\n{'='*60}")
        logger.info("自动化参数搜索完成！")
        logger.info(f"{'='*60}")
        
        if self.best_config:
            logger.info(f"最佳配置 (综合分数: {self.best_score:.4f}):")
            logger.info(f"VQ-VAE: 通道{self.best_config['vqvae']['encoder_channels']}, "
                       f"潜在维度{self.best_config['vqvae']['latent_dim']}")
            logger.info(f"LDM: 通道{self.best_config['ldm']['unet_channels']}, "
                       f"时间嵌入{self.best_config['ldm']['time_emb_dim']}")
        
        logger.info(f"\n搜索历史已保存到 search_history_{self.target_font_name}.json")
        logger.info("您可以查看详细结果并决定是否使用最佳配置")


def main():
    parser = argparse.ArgumentParser(description="自动化参数搜索")
    parser.add_argument("--target_font_path", type=str, default="fonts/M8.ttf",
                       help="目标字体路径")
    parser.add_argument("--max_iterations", type=int, default=20,
                       help="最大搜索迭代次数")
    
    args = parser.parse_args()
    
    # 创建搜索实例
    searcher = AutoParameterSearch(args.target_font_path, args.max_iterations)
    
    # 开始搜索
    searcher.run_search()


if __name__ == "__main__":
    main()
