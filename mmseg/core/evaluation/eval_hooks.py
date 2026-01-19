# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
# 尝试从 mmcv.runner 导入（mmcv 1.x），如果失败则尝试其他位置
try:
    from mmcv.runner import DistEvalHook as _DistEvalHook
    from mmcv.runner import EvalHook as _EvalHook
except ImportError:
    # MMCV 2.x中没有mmcv.runner，需要创建一个兼容层
    # 从mmengine导入基础Hook类
    from mmengine.hooks import Hook as _HookBase
    
    # 创建一个简化的EvalHook兼容实现
    # 注意：这是最小实现，可能不包含所有MMCV 1.x的功能
    class _EvalHookCompat(_HookBase):
        """兼容性EvalHook（基于mmengine.Hook的最小实现）"""
        
        def __init__(self, dataloader, by_epoch=False, **kwargs):
            super().__init__()
            self.dataloader = dataloader
            self.by_epoch = by_epoch
            # 默认属性
            if 'start' not in kwargs:
                kwargs['start'] = 1  # 从第1次迭代开始评估，避免在iter=0时评估
            if 'interval' not in kwargs:
                kwargs['interval'] = 1
            # 复制所有kwargs到self
            for key, value in kwargs.items():
                setattr(self, key, value)
            # 其他默认属性
            if not hasattr(self, 'save_best'):
                self.save_best = None
            if not hasattr(self, 'evaluate'):
                self.evaluate = lambda runner, results: {}
        
        def _should_evaluate(self, runner):
            """判断是否应该执行评估"""
            if self.by_epoch:
                return runner.epoch >= self.start and (runner.epoch + 1) % self.interval == 0
            else:
                return runner.iter >= self.start and runner.iter % self.interval == 0
        
        def _do_evaluate(self, runner):
            """执行评估"""
            if not self._should_evaluate(runner):
                return
            # 这个方法会被子类重写
            pass
        
        def after_train_iter(self, runner):
            """训练迭代后执行"""
            if not self.by_epoch:
                self._do_evaluate(runner)
        
        def after_train_epoch(self, runner):
            """训练epoch后执行"""
            if self.by_epoch:
                self._do_evaluate(runner)
    
    class _DistEvalHookCompat(_HookBase):
        """兼容性DistEvalHook（基于mmengine.Hook的最小实现）"""
        
        def __init__(self, dataloader, by_epoch=False, **kwargs):
            super().__init__()
            self.dataloader = dataloader
            self.by_epoch = by_epoch
            # 复制所有kwargs到self
            for key, value in kwargs.items():
                setattr(self, key, value)
            # 默认属性
            if not hasattr(self, 'save_best'):
                self.save_best = None
            if not hasattr(self, 'broadcast_bn_buffer'):
                self.broadcast_bn_buffer = False
            if not hasattr(self, 'tmpdir'):
                self.tmpdir = None
            if not hasattr(self, 'gpu_collect'):
                self.gpu_collect = False
            if not hasattr(self, 'evaluate'):
                self.evaluate = lambda runner, results: {}
        
        def _should_evaluate(self, runner):
            """判断是否应该执行评估"""
            if self.by_epoch:
                return runner.epoch >= self.start and (runner.epoch + 1) % self.interval == 0
            else:
                return runner.iter >= self.start and runner.iter % self.interval == 0
        
        def _do_evaluate(self, runner):
            """执行评估"""
            if not self._should_evaluate(runner):
                return
            # 这个方法会被子类重写
            pass
        
        def after_train_iter(self, runner):
            """训练迭代后执行"""
            if not self.by_epoch:
                self._do_evaluate(runner)
        
        def after_train_epoch(self, runner):
            """训练epoch后执行"""
            if self.by_epoch:
                self._do_evaluate(runner)
    
    _EvalHook = _EvalHookCompat
    _DistEvalHook = _DistEvalHookCompat
    
    warnings.warn(
        "MMCV 2.x中EvalHook的位置已改变，使用兼容实现。"
        "某些功能可能不完整，建议检查评估功能是否正常。"
    )

from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        # self.pre_eval = pre_eval
        self.efficent_test = efficient_test
        # 初始化最佳分数跟踪
        if not hasattr(self, 'best_score'):
            self.best_score = None
        # 从kwargs中获取metric（如果存在）
        if 'metric' in kwargs:
            self.metric = kwargs['metric']
        elif not hasattr(self, 'metric'):
            self.metric = 'mIoU'  # 默认metric
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')
    
    def _save_ckpt(self, runner, key_score):
        """保存最佳checkpoint
        
        Args:
            runner: Runner对象
            key_score: 评估指标分数（字典）
        """
        if not self.save_best or key_score is None:
            return
        
        import torch
        
        # 获取要比较的键（如果save_best是字符串，使用它；否则使用第一个greater_key）
        if isinstance(self.save_best, str):
            key = self.save_best
        else:
            # 尝试找到第一个在greater_keys中的键
            key = None
            for k in self.greater_keys:
                if k in key_score:
                    key = k
                    break
            if key is None:
                # 如果没有找到，使用key_score中的第一个键
                key = list(key_score.keys())[0] if key_score else None
        
        if key is None or key not in key_score:
            return
        
        current_score = float(key_score[key])
        should_save = False
        
        # 判断是否应该保存（分数更大表示更好，对于greater_keys中的指标）
        if not hasattr(self, 'best_score') or self.best_score is None:
            should_save = True
            self.best_score = current_score
        elif key in self.greater_keys:
            # 对于greater_keys中的指标，分数越大越好
            if current_score > self.best_score:
                should_save = True
                self.best_score = current_score
        else:
            # 对于其他指标，假设也是越大越好（可以根据需要修改）
            if current_score > self.best_score:
                should_save = True
                self.best_score = current_score
        
        if should_save:
            # 保存checkpoint
            best_ckpt_path = osp.join(runner.work_dir, f'best_{key}.pth')
            try:
                # 获取模型（可能是DataParallel包装的）
                model = runner.model
                if hasattr(model, 'module'):
                    model = model.module
                
                # 构建checkpoint字典
                checkpoint = {
                    'meta': {
                        'iter': runner.iter if hasattr(runner, 'iter') else 0,
                        'epoch': runner.epoch if hasattr(runner, 'epoch') else 0,
                        'best_score': self.best_score,
                        'best_key': key,
                        'key_score': key_score
                    },
                    'state_dict': model.state_dict()
                }
                
                # 如果有优化器，也保存
                if hasattr(runner, 'optimizer'):
                    checkpoint['optimizer'] = runner.optimizer.state_dict()
                
                # 保存checkpoint
                torch.save(checkpoint, best_ckpt_path)
                runner.logger.info(
                    f'Saved best checkpoint: {best_ckpt_path} '
                    f'({key}={self.best_score:.4f})'
                )
            except Exception as e:
                runner.logger.warning(f'Failed to save best checkpoint: {e}')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        import shutil
        from mmseg.apis import single_gpu_test
        
        # 如果使用efficient_test，将临时目录放在work_dir下
        efficient_test_tmpdir = None
        if self.efficent_test:
            efficient_test_tmpdir = osp.join(runner.work_dir, '.efficient_test')
            # 清理之前的临时文件（如果存在）
            if osp.exists(efficient_test_tmpdir):
                try:
                    shutil.rmtree(efficient_test_tmpdir)
                    runner.logger.info(f'Cleaned up previous efficient_test directory: {efficient_test_tmpdir}')
                except Exception as e:
                    runner.logger.warning(f'Failed to clean up efficient_test directory: {e}')
        
        try:
            # here we remove pre_eval
            results = single_gpu_test(
                runner.model, 
                self.dataloader, 
                show=False,
                efficient_test=self.efficent_test,
                tmpdir=efficient_test_tmpdir if self.efficent_test else None)
            # here we remove runner.log_buffer.clear() so that we get key_score
            # runner.log_buffer.clear()
            # 兼容处理：log_buffer可能是OrderedDict，也可能有output属性
            if hasattr(runner.log_buffer, 'output'):
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            else:
                runner.log_buffer['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
        finally:
            # 评估完成后清理临时文件
            if self.efficent_test and efficient_test_tmpdir and osp.exists(efficient_test_tmpdir):
                try:
                    shutil.rmtree(efficient_test_tmpdir)
                    runner.logger.info(f'Cleaned up efficient_test directory: {efficient_test_tmpdir}')
                except Exception as e:
                    runner.logger.warning(f'Failed to clean up efficient_test directory: {e}')
    
    def evaluate(self, runner, results):
        """评估结果并返回评估指标
        
        Args:
            runner: Runner对象
            results: single_gpu_test返回的结果列表
            
        Returns:
            dict: 评估指标字典，包含mIoU等指标
        """
        dataset = self.dataloader.dataset
        
        # 如果results是文件路径列表（efficient_test=True），需要先检查
        # 否则，如果dataset有pre_eval方法，调用它转换格式
        if hasattr(dataset, 'pre_eval'):
            # 构建indices列表（results的索引对应数据集的索引）
            indices = list(range(len(results)))
            
            # 调用pre_eval转换格式（从分割结果转换为(tp, fp, fn, valid)）
            pre_eval_results = dataset.pre_eval(results, indices)
            
            # 调用dataset.evaluate进行评估
            eval_results = dataset.evaluate(
                pre_eval_results,
                metric=self.metric if hasattr(self, 'metric') else 'mIoU',
                logger=runner.logger
            )
        else:
            # 如果没有pre_eval方法，直接调用evaluate（可能期望不同格式）
            eval_results = dataset.evaluate(
                results,
                metric=self.metric if hasattr(self, 'metric') else 'mIoU',
                logger=runner.logger
            )
        
        # 使用metrics_logger记录指标到TensorBoard和CSV
        self._log_metrics_to_tb_and_csv(runner, eval_results, dataset)
        
        return eval_results
    
    def _log_metrics_to_tb_and_csv(self, runner, eval_results, dataset):
        """将评估指标记录到TensorBoard和CSV
        
        Args:
            runner: Runner对象
            eval_results: 评估结果字典
            dataset: 数据集对象
        """
        try:
            from mmseg.utils.metrics_logger import MetricsLogger
            
            # 初始化metrics_logger（如果还没有初始化）
            if not hasattr(runner, 'metrics_logger'):
                runner.metrics_logger = MetricsLogger(
                    work_dir=runner.work_dir,
                    csv_filename='metrics.csv',
                    mode='eval'
                )
            
            # 获取当前步数
            step = runner.iter if hasattr(runner, 'iter') else (runner.epoch if hasattr(runner, 'epoch') else 0)
            
            # 记录分割指标
            if 'mIoU' in eval_results:
                mIoU = eval_results['mIoU']
                
                # 提取各类别IoU
                class_IoU = {}
                class_names = getattr(dataset, 'CLASSES', [])
                for key, value in eval_results.items():
                    # 查找IoU.{class_name}格式的键
                    if key.startswith('IoU.') and isinstance(value, (int, float)):
                        class_name = key.replace('IoU.', '')
                        class_IoU[class_name] = value
                
                # 记录到metrics_logger
                runner.metrics_logger.log_segmentation_metrics(
                    mIoU=mIoU,
                    class_IoU=class_IoU if class_IoU else None,
                    class_names=class_names,
                    step=step,
                    prefix='eval/segmentation'
                )
                
                # 如果eval_results中有其他mIoU变体，也记录
                for key in ['mIoUv1', 'mIoUv2']:
                    if key in eval_results:
                        runner.metrics_logger.log({key: eval_results[key]}, step=step, prefix='eval/segmentation')
            
            # 记录检测指标（如果存在）
            detection_metrics = {}
            for key in ['NDS', 'mAP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE']:
                if key in eval_results:
                    detection_metrics[key] = eval_results[key]
            
            if detection_metrics:
                runner.metrics_logger.log_detection_metrics(
                    step=step,
                    prefix='eval/detection',
                    **detection_metrics
                )
            
            # 刷新缓冲区
            runner.metrics_logger.flush()
            
        except Exception as e:
            # 如果记录失败，记录警告但继续执行
            if hasattr(runner, 'logger'):
                runner.logger.warning(f'Failed to log metrics to TensorBoard/CSV: {e}')
            else:
                print(f'Warning: Failed to log metrics to TensorBoard/CSV: {e}')


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        # 初始化最佳分数跟踪
        if not hasattr(self, 'best_score'):
            self.best_score = None
        # 从kwargs中获取metric（如果存在）
        if 'metric' in kwargs:
            self.metric = kwargs['metric']
        elif not hasattr(self, 'metric'):
            self.metric = 'mIoU'  # 默认metric
    
    def _save_ckpt(self, runner, key_score):
        """保存最佳checkpoint（与EvalHook相同的实现）"""
        EvalHook._save_ckpt(self, runner, key_score)
    
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        import shutil
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        
        # 如果使用efficient_test，将临时目录放在work_dir下
        efficient_test_tmpdir = None
        if self.efficient_test:
            efficient_test_tmpdir = osp.join(runner.work_dir, '.efficient_test')
            # 清理之前的临时文件（如果存在）
            if runner.rank == 0 and osp.exists(efficient_test_tmpdir):
                try:
                    shutil.rmtree(efficient_test_tmpdir)
                    runner.logger.info(f'Cleaned up previous efficient_test directory: {efficient_test_tmpdir}')
                except Exception as e:
                    runner.logger.warning(f'Failed to clean up efficient_test directory: {e}')
        
        try:
            from mmseg.apis import multi_gpu_test
            # here we remove pre.eval
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
                efficient_test=self.efficient_test,
                efficient_test_tmpdir=efficient_test_tmpdir if self.efficient_test else None)
            # here we remove runner.log_buffer.clear() so that we get key_score
            # runner.log_buffer.clear()
            # to see the results
            # print('-'*50)
            # print(len(results))
            # print('='*50)
            # print(results)
            # print('-'*50)

            if runner.rank == 0:
                print('\n')
                # 兼容处理：log_buffer可能是OrderedDict，也可能有output属性
                if hasattr(runner.log_buffer, 'output'):
                    runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                else:
                    runner.log_buffer['eval_iter_num'] = len(self.dataloader)
                key_score = self.evaluate(runner, results)

                if self.save_best:
                    # DistEvalHook继承并使用EvalHook的_save_ckpt方法
                    EvalHook._save_ckpt(self, runner, key_score)
        finally:
            # 评估完成后清理临时文件
            if self.efficient_test and efficient_test_tmpdir and runner.rank == 0 and osp.exists(efficient_test_tmpdir):
                try:
                    shutil.rmtree(efficient_test_tmpdir)
                    runner.logger.info(f'Cleaned up efficient_test directory: {efficient_test_tmpdir}')
                except Exception as e:
                    runner.logger.warning(f'Failed to clean up efficient_test directory: {e}')
    
    def evaluate(self, runner, results):
        """评估结果并返回评估指标（与EvalHook相同的实现）"""
        eval_results = EvalHook.evaluate(self, runner, results)
        # 记录指标到TensorBoard和CSV（已在EvalHook.evaluate中处理）
        return eval_results