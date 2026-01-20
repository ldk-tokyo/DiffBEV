# Copyright (c) OpenMMLab. All rights reserved.
"""
MMCV 1.x Runner 兼容类：提供与MMCV 1.x Runner兼容的接口
"""
import warnings
import os
import time
import torch
from collections import OrderedDict

# FP16支持（优先使用新的torch.amp API）
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
    AMP_NEW_API = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
        AMP_NEW_API = False
    except ImportError:
        AMP_AVAILABLE = False
        AMP_NEW_API = False
        autocast = None
        GradScaler = None


class MMCVRunnerCompat(object):
    """兼容MMCV 1.x Runner接口的包装类
    
    这个类实现了MMCV 1.x Runner的基本接口，以支持现有的训练流程。
    注意：这是一个简化的实现，某些高级功能可能不支持。
    """
    
    def __init__(self,
                 model=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 batch_processor=None,
                 runner_type='IterBasedRunner',
                 max_iters=None,
                 max_epochs=None):
        """初始化Runner
        
        Args:
            model: 模型
            optimizer: 优化器
            work_dir: 工作目录
            logger: 日志记录器
            meta: 元数据
            batch_processor: 批处理器（已废弃）
            runner_type: Runner类型 ('IterBasedRunner' 或 'EpochBasedRunner')
            max_iters: 最大迭代次数（IterBasedRunner）
            max_epochs: 最大epoch数（EpochBasedRunner）
        """
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta or {}
        self.batch_processor = batch_processor
        self.runner_type = runner_type
        self.max_iters = max_iters
        self.max_epochs = max_epochs
        
        # Runner状态
        self.iter = 0
        self.epoch = 0
        self.inner_iter = 0
        self.mode = 'train'
        self.timestamp = None
        
        # Hooks存储
        self.hooks = []
        self.hook_priority_map = {
            'LOWEST': 0,
            'LOW': 10,
            'NORMAL': 50,
            'HIGH': 90,
            'HIGHEST': 100
        }
        
        # 初始化iter/epoch
        if runner_type == 'IterBasedRunner':
            self.by_epoch = False
        else:
            self.by_epoch = True
        
        # metrics_logger将在首次使用时初始化
        self.metrics_logger = None
        
        # 保存第一次迭代的输出用于loss结构检查
        self._first_iter_outputs = None
        
        # FP16/BF16支持（将在配置时初始化）
        self.fp16_enabled = False
        self.bf16_enabled = False
        self.fp16_scaler = None
        self.amp_dtype = None  # 'float16' 或 'bfloat16'
        
        # 梯度裁剪配置（将在register_training_hooks时设置）
        self.grad_clip = None
    
    def register_training_hooks(self,
                                lr_config=None,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                timer_config=dict(type='IterTimerHook')):
        """注册训练hooks（兼容MMCV 1.x接口）
        
        Args:
            lr_config: 学习率配置
            optimizer_config: 优化器配置
            checkpoint_config: checkpoint配置
            log_config: 日志配置
            momentum_config: momentum配置
            timer_config: timer配置
        """
        warnings.warn(
            "register_training_hooks 使用兼容实现，某些功能可能不完整。"
            "建议检查训练流程是否正常。"
        )
        
        # 这里需要注册各种hooks，但为了简化，我们暂时只是保存配置
        # 实际训练时会在run方法中使用这些配置
        self.lr_config = lr_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.log_config = log_config
        self.momentum_config = momentum_config
        self.timer_config = timer_config
        
        # 从optimizer_config中提取grad_clip配置
        if optimizer_config is not None and isinstance(optimizer_config, dict):
            self.grad_clip = optimizer_config.get('grad_clip', None)
            if self.grad_clip is not None and self.logger is not None:
                self.logger.info(f"✅ 梯度裁剪已启用: {self.grad_clip}")
        
        # TODO: 实现实际的hook注册逻辑
    
    def register_hook(self, hook, priority='NORMAL'):
        """注册hook
        
        Args:
            hook: Hook实例
            priority: Hook优先级（可以是字符串或整数）
        """
        # 确定要使用的优先级值
        if hasattr(hook, 'priority'):
            # Hook已经有priority属性，确保它是整数类型
            hook_priority = hook.priority
            if isinstance(hook_priority, str):
                hook.priority = self.hook_priority_map.get(hook_priority, 50)
            elif not isinstance(hook_priority, int):
                # 如果不是字符串也不是整数，尝试转换或使用默认值
                try:
                    hook.priority = int(hook_priority)
                except (ValueError, TypeError):
                    hook.priority = 50
        else:
            # Hook没有priority属性，使用传入的priority参数
            if isinstance(priority, str):
                hook.priority = self.hook_priority_map.get(priority, 50)
            else:
                hook.priority = int(priority) if priority is not None else 50
        
        self.hooks.append(hook)
        # 按优先级排序（确保所有priority都是整数）
        def get_priority_value(h):
            """获取hook的优先级数值"""
            if not hasattr(h, 'priority'):
                return 50
            p = h.priority
            if isinstance(p, int):
                return p
            elif isinstance(p, str):
                return self.hook_priority_map.get(p, 50)
            else:
                try:
                    return int(p)
                except (ValueError, TypeError):
                    return 50
        
        self.hooks.sort(key=get_priority_value, reverse=True)
    
    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """运行训练（兼容MMCV 1.x接口）
        
        Args:
            data_loaders: 数据加载器列表
            workflow: 工作流程，如 [('train', 1)]
            max_iters: 最大迭代次数
            **kwargs: 其他参数
        """
        warnings.warn(
            "run 方法使用兼容实现，某些功能可能不完整。"
            "建议检查训练流程是否正常。"
        )
        
        # 设置最大迭代次数
        if max_iters is None:
            max_iters = self.max_iters or float('inf')
        
        # 初始化hooks
        self._call_hook('before_train')
        
        # 遍历workflow中的每个阶段
        for mode, epochs in workflow:
            assert mode in ['train', 'val', 'test'], \
                f'runner mode should be train, val or test, but got {mode}'
            
            if mode == 'train':
                self.train(data_loaders[0], max_iters=max_iters)
            elif mode == 'val':
                self.val(data_loaders[1] if len(data_loaders) > 1 else data_loaders[0])
            elif mode == 'test':
                self.test(data_loaders[1] if len(data_loaders) > 1 else data_loaders[0])
        
        self._call_hook('after_train')
    
    def train(self, data_loader, max_iters=None):
        """训练模式
        
        Args:
            data_loader: 数据加载器
            max_iters: 最大迭代次数
        """
        self.model.train()
        self.mode = 'train'
        
        if max_iters is None:
            max_iters = self.max_iters or float('inf')
        
        data_loader_iter = iter(data_loader)
        
        # 添加进度条和时间跟踪
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        if use_tqdm:
            initial_iter = self.iter
            remaining_iters = max_iters - initial_iter
            pbar = tqdm(
                initial=initial_iter,
                total=max_iters,
                desc=f"Training",
                unit="iter",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        start_time = time.time()
        
        # 训练循环
        while self.iter < max_iters:
            try:
                data_batch = next(data_loader_iter)
            except StopIteration:
                # 数据加载器结束，重新开始
                data_loader_iter = iter(data_loader)
                data_batch = next(data_loader_iter)
                self.epoch += 1
            
            self._call_hook('before_train_iter', self.iter)
            
            # 获取模型设备
            if hasattr(self.model, 'module'):
                model = self.model.module
                device = next(model.parameters()).device
            else:
                model = self.model
                device = next(model.parameters()).device
            
            # 解包DataContainer（如果需要）并移动到正确的设备
            # MMDataParallel会自动解包，但如果直接调用模型，需要手动解包
            unwrapped_batch = {}
            for key, value in data_batch.items():
                # 检查是否是DataContainer
                if hasattr(value, 'data') and hasattr(value, 'stack') and hasattr(value, 'padding_value'):
                    # 是DataContainer，解包data
                    data = value.data
                    # 检查cpu_only标志
                    if not getattr(value, 'cpu_only', False) and isinstance(data, torch.Tensor):
                        # 如果不在CPU上，移动到模型设备
                        data = data.to(device)
                    unwrapped_batch[key] = data
                elif isinstance(value, torch.Tensor):
                    # 如果是Tensor，移动到模型设备
                    unwrapped_batch[key] = value.to(device)
                else:
                    # 不是DataContainer也不是Tensor，直接使用
                    unwrapped_batch[key] = value
            
            # 处理img_metas（可能是DataContainer列表，通常是cpu_only）
            if 'img_metas' in unwrapped_batch:
                img_metas = unwrapped_batch['img_metas']
                if isinstance(img_metas, list):
                    # 如果是列表，尝试解包每个元素
                    unwrapped_img_metas = []
                    for meta in img_metas:
                        if hasattr(meta, 'data'):
                            # 是DataContainer，通常img_metas是cpu_only
                            unwrapped_img_metas.append(meta.data)
                        else:
                            unwrapped_img_metas.append(meta)
                    unwrapped_batch['img_metas'] = unwrapped_img_metas
            
            # 执行训练步骤（FP16/BF16支持：使用autocast包装forward pass）
            if (self.fp16_enabled or self.bf16_enabled) and AMP_AVAILABLE:
                if AMP_NEW_API:
                    # 根据配置选择dtype
                    if self.bf16_enabled:
                        # BF16: 使用bfloat16，数值范围与FP32相同，更稳定
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model.train_step(unwrapped_batch, self.optimizer)
                    else:
                        # FP16: 默认使用float16
                        with autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model.train_step(unwrapped_batch, self.optimizer)
                else:
                    # 旧API：FP16使用默认autocast，BF16需要指定dtype
                    if self.bf16_enabled:
                        with autocast(dtype=torch.bfloat16):
                            outputs = model.train_step(unwrapped_batch, self.optimizer)
                    else:
                        with autocast():
                            outputs = model.train_step(unwrapped_batch, self.optimizer)
            else:
                outputs = model.train_step(unwrapped_batch, self.optimizer)
            
            # 执行hooks（如优化器step、学习率更新等）
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            
            # 在第一次迭代时（iter=0，实际是第1次迭代），保存outputs用于loss结构检查
            if self.iter == 0:
                self._first_iter_outputs = outputs.copy() if isinstance(outputs, dict) else {}
            
            if 'log_vars' in outputs:
                # log_buffer是OrderedDict，update()只接受一个参数
                self.log_buffer.update(outputs['log_vars'])
            
            # 反向传播和优化器更新
            if 'loss' in outputs:
                loss_tensor = outputs['loss']
                
                # 检查loss是否为nan或0（在反向传播前）
                if isinstance(loss_tensor, torch.Tensor):
                    if torch.isnan(loss_tensor).any():
                        raise RuntimeError(
                            f"❌ 训练终止: Iter {self.iter+1} 时检测到总loss为 NaN！"
                        )
                    if loss_tensor.item() == 0.0:
                        raise RuntimeError(
                            f"❌ 训练终止: Iter {self.iter+1} 时检测到总loss为 0！"
                        )
                    
                    # 检查loss是否参与计算图（requires_grad）
                    if not loss_tensor.requires_grad:
                        raise RuntimeError(
                            f"❌ 训练终止: Iter {self.iter+1} 时检测到loss未参与反向传播！"
                            f"loss.requires_grad = {loss_tensor.requires_grad}。"
                            f"请检查loss计算是否正确。"
                        )
                
                self.optimizer.zero_grad()
                
                # FP16/BF16支持：使用GradScaler进行反向传播（仅FP16需要scaler）
                if self.fp16_enabled and self.fp16_scaler is not None:
                    self.fp16_scaler.scale(loss_tensor).backward()
                    
                    # FP16: 检查scaler状态，如果出现inf/NaN，跳过此次更新
                    scaler_state = self.fp16_scaler.get_scale()
                    if scaler_state == float('inf') or scaler_state != scaler_state:  # NaN检查
                        self.logger.warning(
                            f"⚠️  Iter {self.iter+1}: FP16 scaler检测到inf/NaN，跳过此次更新并降低loss_scale"
                        )
                        # scaler会自动处理，这里只记录警告
                    
                    # FP16: unscale梯度以检查NaN（在step之前）
                    self.fp16_scaler.unscale_(self.optimizer)
                    
                    # 检查梯度是否存在和是否为NaN（在unscale之后）
                    has_grad = False
                    has_nan = False
                    nan_param_name = None
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            has_grad = True
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan = True
                                nan_param_name = name
                                break
                    
                    if not has_grad:
                        warnings.warn(
                            f"⚠️  Iter {self.iter+1}: 未检测到任何参数的梯度。"
                            f"这可能表示loss未正确连接到模型参数。"
                        )
                    
                    if has_nan:
                        # FP16: 如果检测到NaN，跳过此次更新
                        self.fp16_scaler.update()  # 这会降低loss_scale
                        self.logger.warning(
                            f"⚠️  Iter {self.iter+1}: 检测到参数 '{nan_param_name}' 的梯度为 NaN/inf，"
                            f"跳过此次更新。当前loss_scale: {self.fp16_scaler.get_scale():.2f}"
                        )
                        # 跳过optimizer.step()
                    else:
                        # 梯度裁剪（可选，但建议启用）
                        if hasattr(self, 'grad_clip') and self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), **self.grad_clip)
                        
                        # FP16: 正常更新
                        self.fp16_scaler.step(self.optimizer)
                        self.fp16_scaler.update()
                elif self.bf16_enabled:
                    # BF16模式：直接反向传播（BF16不需要GradScaler，数值范围与FP32相同）
                    loss_tensor.backward()
                    
                    # 检查梯度是否存在
                    has_grad = False
                    has_nan = False
                    nan_param_name = ""
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if param.grad.numel() > 0:  # 确保梯度张量非空
                                has_grad = True
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    has_nan = True
                                    nan_param_name = name
                                    break
                    
                    if not has_grad:
                        warnings.warn(
                            f"⚠️  Iter {self.iter+1}: 未检测到任何参数的梯度。"
                            f"这可能表示loss未正确连接到模型参数。"
                        )
                    
                    if has_nan:
                        warnings.warn(
                            f"⚠️  Iter {self.iter+1}: 检测到参数 '{nan_param_name}' 的梯度为 NaN/inf，跳过此次更新。"
                        )
                    else:
                        # 梯度裁剪（可选，但建议启用）
                        if hasattr(self, 'grad_clip') and self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), **self.grad_clip)
                        
                        # BF16: 直接更新优化器（不需要scaler）
                        self.optimizer.step()
                else:
                    # FP32模式：直接反向传播
                    loss_tensor.backward()
                    
                    # 检查梯度是否存在
                    has_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                param_name = next((name for name, p in model.named_parameters() if p is param), "unknown")
                                raise RuntimeError(
                                    f"❌ 训练终止: Iter {self.iter+1} 时检测到参数 '{param_name}' 的梯度为 NaN！"
                                )
                            has_grad = True
                            break
                    
                    if not has_grad:
                        warnings.warn(
                            f"⚠️  Iter {self.iter+1}: 未检测到任何参数的梯度。"
                            f"这可能表示loss未正确连接到模型参数。"
                        )
                    
                    # 梯度裁剪（FP32模式）
                    if hasattr(self, 'grad_clip') and self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), **self.grad_clip)
                    
                    self.optimizer.step()
            
            # 定期记录训练指标到日志（每50次迭代）
            # 注意：必须在iter递增之前检查，否则会错过记录（如20050变成20051后不满足条件）
            log_interval = getattr(self, 'log_interval', 50)
            current_iter = self.iter  # 保存当前iter值用于记录
            
            # 检查是否应该记录（使用当前iter，而不是递增后的iter）
            if (current_iter + 1) % log_interval == 0 and hasattr(self, 'logger'):
                log_msg_parts = [f"iter={current_iter + 1}"]
                if 'log_vars' in outputs:
                    for key, value in outputs['log_vars'].items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        if isinstance(value, (int, float)):
                            log_msg_parts.append(f"{key}={value:.6f}")
                        else:
                            log_msg_parts.append(f"{key}={value}")
                if hasattr(self.logger, 'info'):
                    self.logger.info(" | ".join(log_msg_parts))
                
                # 使用metrics_logger记录训练指标到TensorBoard和CSV
                # 注意：这里使用current_iter+1作为step，因为这是当前迭代完成后的iter值
                self._log_training_metrics_to_tb_and_csv(outputs, iter_to_log=current_iter + 1)
            
            # after_train_iter hook只接受runner参数，不传递额外的iter参数
            self._call_hook('after_train_iter')
            
            # 保存checkpoint（根据checkpoint_config）
            self._save_checkpoint_if_needed()
            
            self.iter += 1
            self.inner_iter += 1
            
            # 更新进度条
            if use_tqdm:
                # 获取当前损失值（如果有）
                loss_info = ""
                if 'log_vars' in outputs and 'loss' in outputs['log_vars']:
                    loss_val = outputs['log_vars'].get('loss', 0)
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    loss_info = f" loss={loss_val:.4f}"
                
                # 提取其他loss分量（如果存在）
                if 'log_vars' in outputs:
                    if 'loss_seg' in outputs['log_vars']:
                        seg_val = outputs['log_vars']['loss_seg']
                        if isinstance(seg_val, torch.Tensor):
                            seg_val = seg_val.item()
                        loss_info += f" seg={seg_val:.4f}"
                    if 'loss_depth' in outputs['log_vars']:
                        depth_val = outputs['log_vars']['loss_depth']
                        if isinstance(depth_val, torch.Tensor):
                            depth_val = depth_val.item()
                        loss_info += f" depth={depth_val:.4f}"
                    # 提取学习率
                    if 'learning_rate' in outputs['log_vars']:
                        lr_val = outputs['log_vars']['learning_rate']
                        if isinstance(lr_val, torch.Tensor):
                            lr_val = lr_val.item()
                        loss_info += f" lr={lr_val:.6f}"
                    elif hasattr(self, 'optimizer') and hasattr(self.optimizer, 'param_groups'):
                        lr_val = self.optimizer.param_groups[0].get('lr', 0)
                        if lr_val > 0:
                            loss_info += f" lr={lr_val:.6f}"
                
                # 计算平均速度
                current_time = time.time()
                elapsed_time = current_time - start_time
                if self.iter > initial_iter:
                    avg_time_per_iter = elapsed_time / (self.iter - initial_iter)
                    remaining_iters = max_iters - self.iter
                    eta_seconds = remaining_iters * avg_time_per_iter
                    eta_str = f"ETA={eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"ETA={eta_seconds/60:.1f}m"
                else:
                    eta_str = "ETA=计算中..."
                
                # 更新进度条描述
                pbar.set_description(f"Training{loss_info} {eta_str}")
                pbar.update(1)
            
            # 检查是否达到最大迭代次数
            if self.iter >= max_iters:
                break
        
        # 关闭进度条
        if use_tqdm:
            total_time = time.time() - start_time
            pbar.set_description(f"Training完成 - 总耗时: {total_time/3600:.2f}小时")
            pbar.close()
    
    def val(self, data_loader):
        """验证模式
        
        Args:
            data_loader: 数据加载器
        """
        self.model.eval()
        self.mode = 'val'
        
        self._call_hook('before_val')
        
        # 验证循环
        for i, data_batch in enumerate(data_loader):
            self._call_hook('before_val_iter')
            
            with torch.no_grad():
                outputs = self.model.val_step(data_batch, None)
            
            self._call_hook('after_val_iter')
        
        self._call_hook('after_val')
    
    def test(self, data_loader):
        """测试模式
        
        Args:
            data_loader: 数据加载器
        """
        self.model.eval()
        self.mode = 'test'
        
        self._call_hook('before_test')
        
        # 测试循环
        for i, data_batch in enumerate(data_loader):
            self._call_hook('before_test_iter')
            
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, None)
            
            self._call_hook('after_test_iter')
        
        self._call_hook('after_test')
    
    def _call_hook(self, fn_name, *args, **kwargs):
        """调用hooks
        
        Args:
            fn_name: hook函数名，如 'before_train', 'after_train_iter' 等
            *args: 传递给hook的位置参数
            **kwargs: 传递给hook的关键字参数
        """
        import inspect
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                hook_fn = getattr(hook, fn_name)
                # 检查hook函数的签名
                try:
                    sig = inspect.signature(hook_fn)
                    # 获取参数列表
                    params = list(sig.parameters.keys())
                    # 如果hook函数只需要runner参数，只传递self
                    if len(params) == 1:
                        hook_fn(self)
                    else:
                        # 如果hook函数需要更多参数，传递self和args
                        hook_fn(self, *args, **kwargs)
                except (ValueError, TypeError):
                    # 如果无法获取签名，尝试直接调用
                    try:
                        hook_fn(self, *args, **kwargs)
                    except TypeError:
                        # 如果调用失败，尝试只传递self
                        hook_fn(self)
    
    @property
    def log_buffer(self):
        """日志缓冲区"""
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = OrderedDict()
        return self._log_buffer
    
    def _log_training_metrics_to_tb_and_csv(self, outputs, iter_to_log=None):
        """记录训练指标到TensorBoard和CSV
        
        Args:
            outputs: 训练迭代的输出
            iter_to_log: 要记录的iter值（如果为None，使用self.iter）
        
        Args:
            outputs: train_step的输出，包含log_vars
        """
        try:
            from mmseg.utils.metrics_logger import MetricsLogger
            
            # 初始化metrics_logger（如果还没有初始化）
            if self.metrics_logger is None:
                self.metrics_logger = MetricsLogger(
                    work_dir=self.work_dir,
                    csv_filename='metrics.csv',
                    mode='train'
                )
            
            if 'log_vars' not in outputs:
                return
            
            log_vars = outputs['log_vars']
            
            # 提取训练损失指标
            Lwce = None
            Ldepth = None
            Ldiff = None
            loss_total = None
            learning_rate = None
            
            # 尝试从log_vars中提取各种损失
            # Lwce可能在loss_seg或其他键中
            if 'loss_seg' in log_vars:
                Lwce = log_vars['loss_seg']
                if isinstance(Lwce, torch.Tensor):
                    Lwce = Lwce.item()
            elif 'loss_decode.loss_seg' in log_vars:
                Lwce = log_vars['loss_decode.loss_seg']
                if isinstance(Lwce, torch.Tensor):
                    Lwce = Lwce.item()
            
            # Ldepth
            if 'loss_depth' in log_vars:
                Ldepth = log_vars['loss_depth']
                if isinstance(Ldepth, torch.Tensor):
                    Ldepth = Ldepth.item()
            
            # Ldiff
            if 'loss_diff' in log_vars:
                Ldiff = log_vars['loss_diff']
                if isinstance(Ldiff, torch.Tensor):
                    Ldiff = Ldiff.item()
            elif 'loss_diffusion' in log_vars:
                Ldiff = log_vars['loss_diffusion']
                if isinstance(Ldiff, torch.Tensor):
                    Ldiff = Ldiff.item()
            
            # 总损失
            if 'loss' in log_vars:
                loss_total = log_vars['loss']
                if isinstance(loss_total, torch.Tensor):
                    loss_total = loss_total.item()
            
            # 学习率
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # 尝试从优化器中获取学习率
                if hasattr(self.optimizer, 'param_groups') and len(self.optimizer.param_groups) > 0:
                    learning_rate = self.optimizer.param_groups[0].get('lr', None)
            if learning_rate is None and 'lr' in log_vars:
                learning_rate = log_vars['lr']
                if isinstance(learning_rate, torch.Tensor):
                    learning_rate = learning_rate.item()
            if learning_rate is None and 'learning_rate' in log_vars:
                learning_rate = log_vars['learning_rate']
                if isinstance(learning_rate, torch.Tensor):
                    learning_rate = learning_rate.item()
            
            # 记录到metrics_logger
            # 使用iter_to_log（如果提供），否则使用self.iter
            step_to_log = iter_to_log if iter_to_log is not None else self.iter
            self.metrics_logger.log_training_losses(
                Lwce=Lwce,
                Ldepth=Ldepth,
                Ldiff=Ldiff,
                loss_total=loss_total,
                learning_rate=learning_rate,
                step=step_to_log,
                prefix='train',
                mode='train'
            )
            
            # 刷新缓冲区
            self.metrics_logger.flush()
            
        except Exception as e:
            # 如果记录失败，记录警告但继续执行
            if hasattr(self, 'logger'):
                self.logger.warning(f'Failed to log training metrics to TensorBoard/CSV: {e}')
            else:
                print(f'Warning: Failed to log training metrics to TensorBoard/CSV: {e}')
    
    def resume(self, checkpoint):
        """恢复训练
        
        Args:
            checkpoint: checkpoint路径
        """
        import os
        import torch
        
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint}")
        
        if self.logger is not None:
            self.logger.info(f"📂 从checkpoint恢复训练: {checkpoint}")
        
        # 加载checkpoint
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        
        # 恢复迭代次数和epoch
        if 'iter' in checkpoint_data:
            self.iter = checkpoint_data['iter']
        if 'epoch' in checkpoint_data:
            self.epoch = checkpoint_data['epoch']
        
        # 恢复模型状态
        if 'state_dict' in checkpoint_data:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint_data['state_dict'])
            else:
                self.model.load_state_dict(checkpoint_data['state_dict'])
        
        # 恢复优化器状态
        if 'optimizer' in checkpoint_data and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        
        # 恢复FP16 scaler状态
        if 'fp16_scaler' in checkpoint_data and self.fp16_enabled and self.fp16_scaler is not None:
            self.fp16_scaler.load_state_dict(checkpoint_data['fp16_scaler'])
        
        if self.logger is not None:
            self.logger.info(f"✅ 已恢复训练状态: iter={self.iter}, epoch={self.epoch}")
    
    def load_checkpoint(self, filename):
        """加载checkpoint（仅加载模型权重，不恢复训练状态）
        
        Args:
            filename: checkpoint文件路径
        """
        import os
        import torch
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint文件不存在: {filename}")
        
        if self.logger is not None:
            self.logger.info(f"📂 加载checkpoint: {filename}")
        
        checkpoint_data = torch.load(filename, map_location='cpu')
        
        # 只加载模型权重
        if 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            # 如果没有state_dict键，假设整个checkpoint就是state_dict
            state_dict = checkpoint_data
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        
        if self.logger is not None:
            self.logger.info("✅ 模型权重已加载")
    
    def save_checkpoint(self,
                       out_dir,
                       filename_tmpl='iter_{}.pth',
                       meta=None,
                       create_symlink=True):
        """保存checkpoint
        
        Args:
            out_dir: 输出目录
            filename_tmpl: 文件名模板
            meta: 元数据
            create_symlink: 是否创建符号链接
        """
        import os
        import torch
        
        os.makedirs(out_dir, exist_ok=True)
        
        # 准备checkpoint数据
        checkpoint = {
            'meta': meta or self.meta.copy(),
            'iter': self.iter,
            'epoch': self.epoch,
        }
        
        # 保存模型状态
        if hasattr(self.model, 'module'):
            checkpoint['state_dict'] = self.model.module.state_dict()
        else:
            checkpoint['state_dict'] = self.model.state_dict()
        
        # 保存优化器状态
        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        
        # 保存FP16 scaler状态
        if self.fp16_enabled and self.fp16_scaler is not None:
            checkpoint['fp16_scaler'] = self.fp16_scaler.state_dict()
        
        # 生成文件名
        filename = filename_tmpl.format(self.iter)
        filepath = os.path.join(out_dir, filename)
        
        # 保存checkpoint
        torch.save(checkpoint, filepath)
        
        if self.logger is not None:
            self.logger.info(f"✅ Checkpoint已保存: {filepath}")
        
        # 创建latest.pth符号链接
        if create_symlink:
            latest_path = os.path.join(out_dir, 'latest.pth')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(filename, latest_path)
        
        return filepath
    
    def _save_checkpoint_if_needed(self):
        """根据checkpoint_config检查是否需要保存checkpoint"""
        if self.checkpoint_config is None:
            return
        
        # 检查是否到了保存间隔
        interval = self.checkpoint_config.get('interval', 5000)
        by_epoch = self.checkpoint_config.get('by_epoch', False)
        
        should_save = False
        if by_epoch:
            # 基于epoch保存
            if self.epoch > 0 and self.epoch % interval == 0:
                should_save = True
        else:
            # 基于iteration保存
            if self.iter > 0 and self.iter % interval == 0:
                should_save = True
        
        if should_save:
            # 获取max_keep_ckpts配置
            max_keep_ckpts = self.checkpoint_config.get('max_keep_ckpts', -1)
            
            # 保存checkpoint
            self.save_checkpoint(
                out_dir=self.work_dir,
                filename_tmpl='iter_{}.pth',
                create_symlink=True,
                meta=self.meta
            )
            
            # 清理旧checkpoint（如果设置了max_keep_ckpts）
            if max_keep_ckpts > 0:
                self._cleanup_old_checkpoints(max_keep_ckpts)
    
    def _cleanup_old_checkpoints(self, max_keep):
        """清理旧的checkpoint文件，只保留最近的max_keep个"""
        import os
        import glob
        
        # 查找所有checkpoint文件
        pattern = os.path.join(self.work_dir, 'iter_*.pth')
        checkpoint_files = glob.glob(pattern)
        
        # 排除latest.pth符号链接
        checkpoint_files = [f for f in checkpoint_files if not f.endswith('latest.pth')]
        
        # 按修改时间排序
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # 删除多余的checkpoint
        if len(checkpoint_files) > max_keep:
            for old_file in checkpoint_files[max_keep:]:
                try:
                    os.remove(old_file)
                    if self.logger is not None:
                        self.logger.info(f"🗑️  删除旧checkpoint: {os.path.basename(old_file)}")
                except Exception as e:
                    if self.logger is not None:
                        self.logger.warning(f"⚠️  无法删除旧checkpoint {old_file}: {e}")
    
    @property
    def rank(self):
        """当前进程的rank（分布式训练）"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except:
            pass
        return 0
    
    @property
    def world_size(self):
        """总进程数（分布式训练）"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_world_size()
        except:
            pass
        return 1
