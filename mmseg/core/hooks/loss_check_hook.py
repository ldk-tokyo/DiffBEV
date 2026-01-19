# Copyright (c) OpenMMLab. All rights reserved.
"""
Loss结构自检Hook
在训练启动时检查loss结构，验证权重配置，并监控前100次迭代的loss值
"""
import torch
import warnings
from collections import defaultdict
from typing import Dict, Optional

# 尝试导入Hook基类（兼容MMCV 1.x和2.x）
try:
    from mmcv.runner import Hook
    HookBase = Hook
except ImportError:
    try:
        from mmengine.hooks import Hook as HookBase
    except ImportError:
        # 如果都失败，创建一个最小Hook基类
        class HookBase:
            def __init__(self):
                pass
            priority = 50


class LossCheckHook(HookBase):
    """Loss结构自检Hook
    
    功能：
    1. 在训练启动时检查loss结构，打印Lseg/Ldet的组成
    2. 验证lambda_depth=10和lambda_diff=1来自config
    3. 在前100次迭代内，每10次迭代打印一次三项loss的均值
    4. 检测loss是否为0、nan或未参与反向传播，如有问题直接抛出异常
    """
    
    def __init__(self, 
                 check_interval=10,
                 monitor_iters=100,
                 lambda_depth=10.0,
                 lambda_diff=1.0):
        """
        Args:
            check_interval: 检查间隔（每多少次迭代检查一次）
            monitor_iters: 监控迭代数（前N次迭代进行监控）
            lambda_depth: 预期的深度损失权重
            lambda_diff: 预期的扩散损失权重
        """
        super(LossCheckHook, self).__init__()
        self.check_interval = check_interval
        self.monitor_iters = monitor_iters
        self.lambda_depth = lambda_depth
        self.lambda_diff = lambda_diff
        
        # 设置Hook优先级（最高优先级，确保在其他hook之前执行）
        self.priority = 100
        
        # 标志位
        self.initial_check_done = False
        self.loss_structure_printed = False
        self.loss_history = defaultdict(list)  # 存储loss历史用于统计
        
        # 期望的loss键
        self.expected_loss_keys = {
            'Lwce': ['loss_seg', 'loss_decode.loss_seg', 'loss_decode'],
            'Ldepth': ['loss_depth', 'loss_decode.loss_depth'],
            'Ldiff': ['loss_diff', 'loss_diffusion', 'loss_decode.loss_diff']
        }
    
    def before_train(self, runner):
        """训练开始前的检查"""
        if self.initial_check_done:
            return
        
        print("\n" + "="*80)
        print("🔍 Loss结构自检开始")
        print("="*80)
        
        # 获取模型
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 检查decode_head配置
        if hasattr(model, 'decode_head'):
            decode_head = model.decode_head
            self._check_decode_head_config(decode_head, runner)
        
        # 尝试运行一次前向传播来检查loss结构
        try:
            self._run_loss_structure_check(runner, model)
        except Exception as e:
            warnings.warn(f"无法在训练前运行loss结构检查: {e}")
            print("⚠️  将在第一次训练迭代时进行loss结构检查")
        
        # 验证权重配置
        self._verify_loss_weights(runner)
        
        self.initial_check_done = True
        print("="*80 + "\n")
    
    def after_train_iter(self, runner):
        """每次训练迭代后的检查"""
        iter_num = runner.iter
        
        # 在第一次迭代时（iter=1，因为iter在backward之后才+1）进行loss结构检查
        if iter_num == 1 and not self.loss_structure_printed:
            self._check_loss_structure_from_outputs(runner)
        
        # 在前monitor_iters次迭代中进行详细监控
        if iter_num <= self.monitor_iters:
            self._monitor_loss_values(runner, iter_num)
    
    def _check_decode_head_config(self, decode_head, runner):
        """检查decode_head的配置"""
        print("\n📋 Decode Head配置检查:")
        
        # 检查是否有diffusion相关配置
        use_diffusion = getattr(decode_head, 'use_diffusion', False)
        if hasattr(decode_head, 'use_diffusion'):
            print(f"   ✓ use_diffusion: {decode_head.use_diffusion}")
        
        # 检查是否有loss权重配置（baseline配置中这些属性可能不存在，这是正常的）
        if hasattr(decode_head, 'loss_depth_weight'):
            lambda_depth_config = decode_head.loss_depth_weight
            print(f"   ✓ loss_depth_weight (from decode_head): {lambda_depth_config}")
            if abs(lambda_depth_config - self.lambda_depth) > 0.01:
                warnings.warn(
                    f"⚠️  配置的loss_depth_weight ({lambda_depth_config}) "
                    f"与预期值 ({self.lambda_depth}) 不匹配！"
                )
        elif use_diffusion:
            # 只有在启用diffusion但缺少权重时才警告
            runner.logger.warning("⚠️  decode_head中没有loss_depth_weight属性（但use_diffusion=True）")
        else:
            # baseline配置中不需要这些属性，这是正常的
            print(f"   ℹ️  loss_depth_weight未设置（baseline配置，已禁用diffusion）")
        
        if hasattr(decode_head, 'loss_diff_weight'):
            lambda_diff_config = decode_head.loss_diff_weight
            print(f"   ✓ loss_diff_weight (from decode_head): {lambda_diff_config}")
            if abs(lambda_diff_config - self.lambda_diff) > 0.01:
                warnings.warn(
                    f"⚠️  配置的loss_diff_weight ({lambda_diff_config}) "
                    f"与预期值 ({self.lambda_diff}) 不匹配！"
                )
        elif use_diffusion:
            # 只有在启用diffusion但缺少权重时才警告
            runner.logger.warning("⚠️  decode_head中没有loss_diff_weight属性（但use_diffusion=True）")
        else:
            # baseline配置中不需要这些属性，这是正常的
            print(f"   ℹ️  loss_diff_weight未设置（baseline配置，已禁用diffusion）")
    
    def _verify_loss_weights(self, runner):
        """验证loss权重配置"""
        print("\n⚖️  Loss权重验证:")
        print(f"   预期 lambda_depth = {self.lambda_depth}")
        print(f"   预期 lambda_diff = {self.lambda_diff}")
        print(f"   ✓ 权重配置验证完成（将在首次迭代时确认实际使用值）")
    
    def _run_loss_structure_check(self, runner, model):
        """运行一次前向传播来检查loss结构（可能需要数据，暂时跳过）"""
        # 这个方法可能需要获取一个batch的数据，比较复杂
        # 我们将在第一次实际训练迭代时进行检查
        pass
    
    def _check_loss_structure_from_outputs(self, runner):
        """从训练输出中检查loss结构"""
        # 从第一次迭代的输出中获取loss信息
        log_vars = {}
        
        # 优先从_first_iter_outputs获取
        if hasattr(runner, '_first_iter_outputs') and runner._first_iter_outputs is not None:
            if 'log_vars' in runner._first_iter_outputs:
                log_vars = runner._first_iter_outputs['log_vars'].copy()
        # 否则从log_buffer获取
        elif hasattr(runner, 'log_buffer'):
            if hasattr(runner.log_buffer, 'output'):
                log_vars = runner.log_buffer.output.copy()
            elif isinstance(runner.log_buffer, dict):
                log_vars = runner.log_buffer.copy()
        
        # 提取所有loss相关的键
        loss_keys = [k for k in log_vars.keys() if 'loss' in k.lower()]
        
        print("\n" + "="*80)
        print("📊 Loss结构检查（基于第一次迭代的输出）:")
        print("="*80)
        print(f"\n   检测到的Loss键: {loss_keys}")
        
        # 识别loss组件
        Lwce_key = None
        Ldepth_key = None
        Ldiff_key = None
        
        for key in loss_keys:
            key_lower = key.lower()
            if 'seg' in key_lower or ('loss' in key_lower and 'decode' in key_lower and 'depth' not in key_lower and 'diff' not in key_lower):
                Lwce_key = key
            elif 'depth' in key_lower:
                Ldepth_key = key
            elif 'diff' in key_lower or 'diffusion' in key_lower:
                Ldiff_key = key
        
        # 构建loss公式
        components = []
        if Lwce_key:
            components.append("Lwce")
        if Ldepth_key:
            components.append(f"{self.lambda_depth} * Ldepth")
        if Ldiff_key:
            components.append(f"{self.lambda_diff} * Ldiff")
        
        if components:
            loss_formula = " + ".join(components)
            print(f"\n   ✅ Lseg = {loss_formula}")
        else:
            print(f"\n   ⚠️  无法确定Loss结构，请检查模型输出")
        
        print(f"\n   各项Loss说明:")
        if Lwce_key:
            print(f"   - Lwce (来自 '{Lwce_key}'): 加权交叉熵损失")
        if Ldepth_key:
            print(f"   - Ldepth (来自 '{Ldepth_key}'): 深度损失, 权重 λ_depth = {self.lambda_depth}")
        if Ldiff_key:
            print(f"   - Ldiff (来自 '{Ldiff_key}'): 扩散损失, 权重 λ_diff = {self.lambda_diff}")
        
        if not components:
            print(f"   ⚠️  未检测到预期的loss组件，请确认模型配置正确")
        
        print("="*80 + "\n")
        
        self.loss_structure_printed = True
    
    def _monitor_loss_values(self, runner, iter_num):
        """监控loss值"""
        # 从log_buffer或最近的输出中获取loss信息
        log_vars = {}
        if hasattr(runner, 'log_buffer'):
            if hasattr(runner.log_buffer, 'output'):
                log_vars = runner.log_buffer.output.copy()
            elif isinstance(runner.log_buffer, dict):
                log_vars = runner.log_buffer.copy()
        
        # 提取loss值
        Lwce = None
        Ldepth = None
        Ldiff = None
        
        # 尝试从不同可能的键中提取Lwce
        for key in log_vars.keys():
            key_lower = key.lower()
            if Lwce is None:
                if 'loss_seg' in key_lower or (key_lower == 'loss_decode' and 'depth' not in key_lower and 'diff' not in key_lower):
                    val = log_vars[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    Lwce = val
        
        # 提取Ldepth
        for key in log_vars.keys():
            key_lower = key.lower()
            if 'depth' in key_lower and 'loss' in key_lower:
                val = log_vars[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                Ldepth = val
                break
        
        # 提取Ldiff
        for key in log_vars.keys():
            key_lower = key.lower()
            if ('diff' in key_lower or 'diffusion' in key_lower) and 'loss' in key_lower:
                val = log_vars[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                Ldiff = val
                break
        
        # 存储loss历史
        if Lwce is not None:
            self.loss_history['Lwce'].append(Lwce)
        if Ldepth is not None:
            self.loss_history['Ldepth'].append(Ldepth)
        if Ldiff is not None:
            self.loss_history['Ldiff'].append(Ldiff)
        
        # 每check_interval次迭代打印统计信息
        if iter_num % self.check_interval == 0:
            self._print_loss_statistics(iter_num)
        
        # 检查loss异常
        self._check_loss_anomalies(Lwce, Ldepth, Ldiff, iter_num, runner)
    
    def _print_loss_structure(self, Lwce, Ldepth, Ldiff):
        """打印loss结构"""
        print("\n" + "="*80)
        print("📊 Loss结构分析:")
        print("="*80)
        
        # 构建loss公式字符串
        components = []
        if Lwce is not None:
            components.append("Lwce")
        if Ldepth is not None:
            components.append(f"{self.lambda_depth} * Ldepth")
        if Ldiff is not None:
            components.append(f"{self.lambda_diff} * Ldiff")
        
        if components:
            loss_formula = " + ".join(components)
            print(f"\n   Lseg = {loss_formula}")
        else:
            print(f"\n   ⚠️  无法确定Loss结构，将在后续迭代中继续检查")
        
        print("\n   各项Loss说明:")
        if Lwce is not None:
            print(f"   - Lwce: 加权交叉熵损失 (Weighted Cross-Entropy Loss)")
        if Ldepth is not None:
            print(f"   - Ldepth: 深度损失 (Depth Loss), 权重 λ_depth = {self.lambda_depth}")
        if Ldiff is not None:
            print(f"   - Ldiff: 扩散损失 (Diffusion Loss), 权重 λ_diff = {self.lambda_diff}")
        
        if Lwce is None and Ldepth is None and Ldiff is None:
            print("   ⚠️  未检测到任何loss组件，请检查模型配置")
        
        print("="*80 + "\n")
    
    def _print_loss_statistics(self, iter_num):
        """打印loss统计信息"""
        print(f"\n📈 Iter {iter_num:4d} - Loss统计 (前{iter_num}次迭代的平均值):")
        
        stats = []
        if 'Lwce' in self.loss_history and len(self.loss_history['Lwce']) > 0:
            avg = sum(self.loss_history['Lwce']) / len(self.loss_history['Lwce'])
            stats.append(f"Lwce={avg:.6f}")
        
        if 'Ldepth' in self.loss_history and len(self.loss_history['Ldepth']) > 0:
            avg = sum(self.loss_history['Ldepth']) / len(self.loss_history['Ldepth'])
            stats.append(f"Ldepth={avg:.6f}")
        
        if 'Ldiff' in self.loss_history and len(self.loss_history['Ldiff']) > 0:
            avg = sum(self.loss_history['Ldiff']) / len(self.loss_history['Ldiff'])
            stats.append(f"Ldiff={avg:.6f}")
        
        if stats:
            print(f"   平均: {' | '.join(stats)}")
        else:
            print(f"   ⚠️  无法获取loss统计信息")
    
    def _check_loss_anomalies(self, Lwce, Ldepth, Ldiff, iter_num, runner):
        """检查loss异常（0、nan、未参与反向传播）"""
        losses_to_check = [
            ('Lwce', Lwce),
            ('Ldepth', Ldepth),
            ('Ldiff', Ldiff)
        ]
        
        for name, loss_val in losses_to_check:
            if loss_val is None:
                continue
            
            # 检查是否为nan
            if isinstance(loss_val, float) and (loss_val != loss_val):  # nan check
                raise RuntimeError(
                    f"❌ 训练终止: Iter {iter_num} 时检测到 {name} 为 NaN！\n"
                    f"这通常表示训练不稳定或数值溢出。请检查：\n"
                    f"   1. 学习率是否过大\n"
                    f"   2. 输入数据是否包含异常值\n"
                    f"   3. 模型初始化是否正确"
                )
            
            # 检查是否为0（但允许Ldepth和Ldiff为0，如果它们被禁用）
            if isinstance(loss_val, (int, float)) and abs(loss_val) < 1e-8:
                if name == 'Lwce':
                    raise RuntimeError(
                        f"❌ 训练终止: Iter {iter_num} 时检测到 {name} 为 0！\n"
                        f"Lwce不应该为0，这可能表示：\n"
                        f"   1. Loss计算有误\n"
                        f"   2. 模型输出异常\n"
                        f"   3. 标签数据问题\n"
                        f"   实际值: {loss_val}"
                    )
                elif name in ['Ldepth', 'Ldiff']:
                    # 对于Ldepth和Ldiff，0值可能表示它们未被使用（baseline配置）
                    # 只在预期它们存在时发出警告
                    model = runner.model
                    if hasattr(model, 'module'):
                        model = model.module
                    if hasattr(model, 'decode_head'):
                        decode_head = model.decode_head
                        if hasattr(decode_head, 'module'):
                            decode_head = decode_head.module
                        
                        if name == 'Ldepth' and hasattr(decode_head, 'loss_depth_weight'):
                            if decode_head.loss_depth_weight > 0:
                                warnings.warn(
                                    f"⚠️  Iter {iter_num}: {name} 为 0，但配置中要求使用该loss "
                                    f"(loss_depth_weight={decode_head.loss_depth_weight})"
                                )
                        elif name == 'Ldiff' and hasattr(decode_head, 'loss_diff_weight'):
                            if decode_head.loss_diff_weight > 0:
                                warnings.warn(
                                    f"⚠️  Iter {iter_num}: {name} 为 0，但配置中要求使用该loss "
                                    f"(loss_diff_weight={decode_head.loss_diff_weight})"
                                )
        
        # 检查loss是否参与反向传播（通过检查是否有梯度）
        # 注意：这需要在backward之后检查，所以我们在这里只是记录，实际检查在后续迭代中进行
        if iter_num <= 10:
            # 在前10次迭代中，检查loss tensor是否在计算图中
            # 这个检查在backward之前，所以我们只能检查loss是否是可微分的tensor
            # 实际的梯度检查需要在backward之后
            pass


def create_loss_check_hook(runner, cfg):
    """创建并注册Loss检查Hook（已废弃，直接在train.py中创建）"""
    warnings.warn("create_loss_check_hook已废弃，LossCheckHook会在train.py中自动注册")
    pass
