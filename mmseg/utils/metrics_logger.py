# Copyright (c) OpenMMLab. All rights reserved.
"""
通用指标记录工具
支持将训练和评估指标同时写入TensorBoard和CSV文件
"""
import os
import os.path as osp
import csv
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Union
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    warnings.warn("TensorBoard不可用，将仅记录CSV格式的指标")


class MetricsLogger:
    """通用指标记录器
    
    支持同时将指标写入TensorBoard和CSV文件。
    
    Args:
        work_dir (str): 工作目录，CSV文件将保存在此目录
        csv_filename (str): CSV文件名，默认为 'metrics.csv'
        tensorboard_dir (str, optional): TensorBoard日志目录，默认为 work_dir/tf_logs
        mode (str): 记录模式，'train' 或 'eval'
    """
    
    def __init__(self,
                 work_dir: str,
                 csv_filename: str = 'metrics.csv',
                 tensorboard_dir: Optional[str] = None,
                 mode: str = 'train'):
        """
        初始化指标记录器
        
        Args:
            work_dir: 工作目录
            csv_filename: CSV文件名
            tensorboard_dir: TensorBoard日志目录（可选）
            mode: 记录模式 ('train' 或 'eval')
        """
        self.work_dir = work_dir
        self.mode = mode
        self.csv_path = osp.join(work_dir, csv_filename)
        
        # 初始化TensorBoard
        if TENSORBOARD_AVAILABLE:
            if tensorboard_dir is None:
                tensorboard_dir = osp.join(work_dir, 'tf_logs')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.tb_writer = None
        
        # CSV文件列头
        self.csv_header_written = False
        self.csv_columns = None
        
        # 确保工作目录存在
        os.makedirs(work_dir, exist_ok=True)
    
    def log(self,
            metrics: Dict[str, Union[float, int]],
            step: Optional[int] = None,
            prefix: str = '',
            mode: Optional[str] = None):
        """
        记录指标到TensorBoard和CSV
        
        Args:
            metrics: 指标字典，key为指标名，value为指标值
            step: 步数（迭代数或epoch数），如果为None则自动递增
            prefix: 指标名前缀，用于区分train/eval等模式
        """
        if step is None:
            # 如果没有指定step，尝试从metrics中获取
            step = metrics.get('iter', metrics.get('epoch', 0))
        
        # 准备记录的数据（规范化基础字段）
        record_dict = OrderedDict()
        record_dict['mode'] = mode if mode is not None else self.mode
        record_dict['phase'] = prefix or ''
        record_dict['step'] = step
        if 'iter' in metrics:
            record_dict['iter'] = metrics['iter']
        if 'epoch' in metrics:
            record_dict['epoch'] = metrics.get('epoch', 0)
        
        # 写入TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['iter', 'epoch']:
                    tag = f"{prefix}/{key}" if prefix else key
                    self.tb_writer.add_scalar(tag, value, step)
        
        # 准备CSV记录（排除非数值类型的指标）
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                record_dict[key] = value
            elif isinstance(value, (list, tuple)):
                # 对于列表类型（如各类别IoU），展开记录
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        record_dict[f"{key}_{i}"] = v
            elif isinstance(value, dict):
                # 对于字典类型，展开记录
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        record_dict[f"{key}_{sub_key}"] = sub_value
        
        # 写入CSV
        self._write_csv(record_dict)
    
    def _write_csv(self, record_dict: Dict):
        """写入CSV文件
        
        Args:
            record_dict: 要记录的字典
        """
        # 写入CSV
        file_exists = osp.exists(self.csv_path)

        # 如果文件已存在且尚未记录表头，先读取已有表头
        if file_exists and not self.csv_header_written:
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                    if existing_header:
                        self.csv_columns = existing_header
                        self.csv_header_written = True
            except Exception:
                # 读取失败则忽略，后续会重新写入表头
                pass

        # 确定CSV列
        if self.csv_columns is None:
            self.csv_columns = list(record_dict.keys())

        # 确保所有列都存在（记录新列是否出现）
        columns_changed = False
        for key in record_dict.keys():
            if key not in self.csv_columns:
                self.csv_columns.append(key)
                columns_changed = True

        # 规范列顺序：固定基础列在前，其余按字母排序
        preferred = ['mode', 'phase', 'step', 'iter', 'epoch']
        if any(col in self.csv_columns for col in preferred):
            fixed = [col for col in preferred if col in self.csv_columns]
            rest = [col for col in self.csv_columns if col not in fixed]
            rest_sorted = sorted(rest)
            new_order = fixed + rest_sorted
            if new_order != self.csv_columns:
                self.csv_columns = new_order
                columns_changed = True

        # 如果新增了列且文件已存在，需要重写表头以保证可读性
        if file_exists and self.csv_header_written and columns_changed:
            self._rewrite_csv_with_new_header()
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns, 
                                  extrasaction='ignore')
            
            # 写入表头（如果是新文件）
            if not file_exists or not self.csv_header_written:
                writer.writeheader()
                self.csv_header_written = True
            
            # 确保所有列都有值
            row = {col: record_dict.get(col, '') for col in self.csv_columns}
            writer.writerow(row)
            f.flush()  # 立即刷新，确保数据写入磁盘
            os.fsync(f.fileno())  # 强制同步到磁盘，确保文件修改时间更新

    def _rewrite_csv_with_new_header(self):
        """当新增列时，重写CSV文件以更新表头。"""
        if not osp.exists(self.csv_path):
            return

        temp_path = self.csv_path + '.tmp'
        try:
            # 读取现有数据
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 写入新表头和旧数据
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns, extrasaction='ignore')
                writer.writeheader()
                for row in rows:
                    # 补齐缺失列
                    full_row = {col: row.get(col, '') for col in self.csv_columns}
                    writer.writerow(full_row)

            shutil.move(temp_path, self.csv_path)
        except Exception:
            # 如果重写失败，删除临时文件并继续使用原文件
            if osp.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def log_segmentation_metrics(self,
                                 mIoU: float,
                                 class_IoU: Optional[Dict[str, float]] = None,
                                 class_names: Optional[list] = None,
                                 step: Optional[int] = None,
                                 prefix: str = '',
                                 mode: Optional[str] = None):
        """
        记录分割指标
        
        Args:
            mIoU: 平均IoU
            class_IoU: 各类别IoU字典，key为类别名，value为IoU值
            class_names: 类别名列表（如果class_IoU为None，会从class_names创建占位符）
            step: 步数
            prefix: 指标前缀
        """
        metrics = {'mIoU': mIoU}
        
        # 添加各类别IoU
        if class_IoU is not None:
            for class_name, iou_value in class_IoU.items():
                metrics[f'IoU_{class_name}'] = iou_value
        elif class_names is not None:
            # 如果提供了类别名但没有IoU值，创建占位符（会在后续更新中填充）
            pass
        
        self.log(metrics, step=step, prefix=prefix or 'eval/segmentation', mode=mode)
    
    def log_detection_metrics(self,
                              NDS: Optional[float] = None,
                              mAP: Optional[float] = None,
                              ATE: Optional[float] = None,
                              ASE: Optional[float] = None,
                              AOE: Optional[float] = None,
                              AVE: Optional[float] = None,
                              AAE: Optional[float] = None,
                              step: Optional[int] = None,
                              prefix: str = '',
                              mode: Optional[str] = None):
        """
        记录检测指标
        
        Args:
            NDS: NuScenes Detection Score
            mAP: mean Average Precision
            ATE: Average Translation Error
            ASE: Average Scale Error
            AOE: Average Orientation Error
            AVE: Average Velocity Error
            AAE: Average Attribute Error
            step: 步数
            prefix: 指标前缀
        """
        metrics = {}
        if NDS is not None:
            metrics['NDS'] = NDS
        if mAP is not None:
            metrics['mAP'] = mAP
        if ATE is not None:
            metrics['ATE'] = ATE
        if ASE is not None:
            metrics['ASE'] = ASE
        if AOE is not None:
            metrics['AOE'] = AOE
        if AVE is not None:
            metrics['AVE'] = AVE
        if AAE is not None:
            metrics['AAE'] = AAE
        
        if metrics:
            self.log(metrics, step=step, prefix=prefix or 'eval/detection', mode=mode)
    
    def log_training_losses(self,
                           Lwce: Optional[float] = None,
                           Ldepth: Optional[float] = None,
                           Ldiff: Optional[float] = None,
                           loss_total: Optional[float] = None,
                           learning_rate: Optional[float] = None,
                           step: Optional[int] = None,
                           prefix: str = '',
                           mode: Optional[str] = None):
        """
        记录训练损失
        
        Args:
            Lwce: 加权交叉熵损失
            Ldepth: 深度损失
            Ldiff: 扩散损失
            loss_total: 总损失
            learning_rate: 学习率
            step: 步数
            prefix: 指标前缀
        """
        metrics = {}
        if Lwce is not None:
            metrics['Lwce'] = Lwce
        if Ldepth is not None:
            metrics['Ldepth'] = Ldepth
        if Ldiff is not None:
            metrics['Ldiff'] = Ldiff
        if loss_total is not None:
            metrics['loss'] = loss_total
        if learning_rate is not None:
            metrics['lr'] = learning_rate
        
        if metrics:
            self.log(metrics, step=step, prefix=prefix or 'train', mode=mode)
    
    def flush(self):
        """刷新缓冲区"""
        if self.tb_writer is not None:
            self.tb_writer.flush()
    
    def close(self):
        """关闭记录器"""
        if self.tb_writer is not None:
            self.tb_writer.close()
