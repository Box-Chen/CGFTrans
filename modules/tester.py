import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import spacy
import scispacy
import torch

from modules.utils import generate_heatmap


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 设备配置（保持不变）
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader
        # 多图场景：创建报告保存目录（保持原始日志路径兼容）
        os.makedirs(os.path.join(self.save_dir, 'reports'), exist_ok=True)

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            # 多图场景：调整报告保存路径，保持与原始"test_LGK.txt"兼容
            report_path = os.path.join(self.save_dir, 'reports', "test_LGK.txt") if hasattr(self.args,
                                                                                            'save_dir') else "test_LGK.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                # 多图场景：解包适配dataloader返回的5个元素（含_）
                for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(self.test_dataloader):
                    # 多图场景：images已为(batch_size, max_num_images, C, H, W)张量，直接移至设备
                    images = images.to(self.device)
                    reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)

                    output, _ = self.model(images, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                    # 多图场景：按病例ID保存报告（而非单图ID）
                    for i in range(len(reports)):
                        case_id = images_id[i]  # 病例ID（多图对应同一病例）
                        f.write(f"第{batch_idx}组数据\n")
                        f.write(f"病例ID: {case_id}\n")  # 替换原图片ID为病例ID
                        f.write(f"生成报告: {reports[i]}\n\n")

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)
        return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        # 多图场景：保持可视化目录结构兼容
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "attentions_entities"), exist_ok=True)
        ner = spacy.load("en_core_sci_sm")
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            # 多图场景：解包适配5个元素
            for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()

                char2word = [idx for word_idx, word in enumerate(report) for idx in [word_idx] * (len(word) + 1)][:-1]

                attention_weights = self.model.encoder_decoder.attention_weights[:-1]
                assert len(attention_weights) == len(report)

                # 多图场景：循环处理批次中的每张图片（原代码只处理单图images[0]）
                batch_size, num_images = images.shape[0], images.shape[1]
                for img_idx in range(num_images):
                    # 提取单张图片并反归一化
                    image = torch.clamp((images[0, img_idx].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()

                    # 为每张图片创建独立子目录（兼容原始结构，增加图片索引）
                    img_subdir = f"img_{img_idx}"

                    # 保存单图注意力权重
                    for word_idx, (attns, word) in enumerate(zip(attention_weights, report)):
                        for layer_idx, attn in enumerate(attns):
                            save_dir = os.path.join(
                                self.save_dir, "attentions", f"{batch_idx:04d}", img_subdir, f"layer_{layer_idx}"
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            heatmap = generate_heatmap(image, attn.mean(1).squeeze())
                            cv2.imwrite(
                                os.path.join(save_dir, f"{word_idx:04d}_{word}.png"),
                                heatmap
                            )

                    # 保存实体注意力权重（多图适配）
                    for ne_idx, ne in enumerate(ner(" ".join(report)).ents):
                        for layer_idx in range(len(attention_weights[0])):
                            save_dir = os.path.join(
                                self.save_dir, "attentions_entities", f"{batch_idx:04d}", img_subdir,
                                f"layer_{layer_idx}"
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            attn = [attns[layer_idx] for attns in
                                    attention_weights[char2word[ne.start_char]:char2word[ne.end_char] + 1]]
                            attn = np.concatenate(attn, axis=2)
                            heatmap = generate_heatmap(image, attn.mean(1).mean(1).squeeze())
                            cv2.imwrite(
                                os.path.join(save_dir, f"{ne_idx:04d}_{ne}.png"),
                                heatmap
                            )