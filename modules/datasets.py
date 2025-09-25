import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r',encoding = 'utf-8').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class MultiImageDataset(BaseDataset):  # 重命名类以适配多图场景
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']

        # 关键修改：确保image_folder是字符串（原代码可能误为列表）
        if isinstance(example['image_path'], list):
            # 如果是列表，取第一个元素（根据实际数据格式调整）
            image_folder = example['image_path'][0]
        else:
            image_folder = example['image_path']

        # 正确拼接路径（均为字符串）
        folder_path = os.path.join(self.image_dir, image_folder)
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # 读取并预处理所有图片
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        # 堆叠成张量 (num_images, C, H, W)
        image = torch.stack(images, dim=0)
        # 保持报告相关处理不变
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        return (image_id, image, report_ids, report_masks, seq_length)


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
