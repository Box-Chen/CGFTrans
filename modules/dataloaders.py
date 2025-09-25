import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import MultiImageDataset, MimiccxrSingleImageDataset
from torch.nn.utils.rnn import pad_sequence

# 在modules/dataloaders.py中
class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        # 原逻辑可能绑定了固定双图的数据集，需修改为多图数据集
        if self.dataset_name == 'iu_xray' or self.dataset_name == 'mimic_cxr':
            # 替换为新的多图数据集类
            self.dataset = MultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        # 其他保持不变
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        """
        保留静态方法，确保返回5个元素，匹配训练器的解包需求
        参数:
            data: 列表，每个元素为元组 (image_id, image_tensor, report_ids, report_masks, seq_length)
        返回:
            5个元素的元组，与训练器中的解包对应
        """
        # 提取数据字段（包含seq_lengths_batch）
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)

        # 1. 处理图片数据（动态padding）
        max_num_images = max(img.size(0) for img in image_batch) if image_batch else 0
        padded_images = []

        for img_tensor in image_batch:
            num_images = img_tensor.size(0)
            if num_images < max_num_images:
                pad_num = max_num_images - num_images
                pad_tensor = torch.zeros(
                    (pad_num, img_tensor.size(1), img_tensor.size(2), img_tensor.size(3)),
                    dtype=img_tensor.dtype,
                    device=img_tensor.device
                )
                padded_img = torch.cat([img_tensor, pad_tensor], dim=0)
                padded_images.append(padded_img)
            else:
                padded_images.append(img_tensor)

        image_batch = torch.stack(padded_images, dim=0)

        # 2. 处理报告序列
        max_seq_length = max(seq_lengths_batch) if seq_lengths_batch else 0
        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        # 3. 返回5个元素（关键修改：包含seq_lengths_batch）
        return (
            image_id_batch,  # 1. 病例ID列表
            image_batch,  # 2. 图片批次张量
            torch.LongTensor(target_batch),  # 3. 报告ID张量
            torch.FloatTensor(target_masks_batch),  # 4. 报告掩码张量
            seq_lengths_batch  # 5. 序列长度列表（新增，匹配训练器解包）
        )
