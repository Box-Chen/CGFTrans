import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r',encoding = 'utf-8').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        # 定义必须保留的符号集合 - 添加左右括号
        required_symbols = {'(', ')', ',', '.', ':', ';', '/', '=', '×', '°', '%', '<', '>', '+','-'}
        
        for example in self.ann['train']:
            report = self.add_space_around_digits(example['report'])
            tokens = self.clean_report(report).split()
            for token in tokens:
                total_tokens.append(token)
        
        # 确保必须保留的符号包含在词汇表中
        for symbol in required_symbols:
            total_tokens.append(symbol)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        # 确保必须保留的符号即使频率低于阈值也被包含
        for symbol in required_symbols:
            if symbol not in vocab:
                vocab.append(symbol)
        
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def add_space_around_digits(self, text):
        """在数字字符前后添加空格"""
        # 在数字字符前后添加空格
        text = re.sub(r'(\d)', r' \1 ', text)
        # 合并多个空格为一个
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_report_iu_xray(self, report):
        # 保留所有关键符号 - 确保括号被保留
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().split('. ')
        # 保留所有关键符号 - 添加括号
        sent_cleaner = lambda t: re.sub(r'[^\w\s\d(),.:;/=×°%<>+-]', '', t.replace('"', '').replace('\\', '').replace("'", '').strip())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        # 保留所有关键符号 - 确保括号被保留
        report_cleaner = lambda t: t.replace('\n', ' ')\
            .strip().split('. ')
        # 保留所有关键符号 - 添加括号
        sent_cleaner = lambda t: re.sub(r'[^\w\s\d(),.:;/=×°%<>+-]', '', t.replace('"', '').replace('\\', '').replace("'", '').strip())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        report = self.add_space_around_digits(report)
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx > 0:
                tokens.append(self.idx2token[idx])
            else:
                break
        
        # 重建文本，处理数字字符和标点符号
        result = []
        for token in tokens:
            # 数字字符直接连接（不加空格）
            if token.isdigit() or token in ',.:;':
                if result and result[-1][-1].isdigit():
                    result[-1] += token
                else:
                    result.append(token)
            # 等号、斜杠直接连接（不加空格）
            elif token in '=/×°%<>+':
                if result:
                    result[-1] += token
                else:
                    result.append(token)
            # 括号作为独立token处理
            elif token in '()':
                result.append(token)
            # 其他token加空格
            else:
                if result:
                    # 确保标点符号前面没有空格
                    if token in '(),.:;':
                        result.append(token)
                    else:
                        result.append(' ' + token)
                else:
                    result.append(token)
        
        # 确保标点符号后面有空格
        text = ''.join(result)
        
        # 处理括号周围的空格
        # 左括号前添加空格（如果前面不是空格或开始）
        text = re.sub(r'([^\s(])(\()', r'\1 \2', text)
        # 右括号后添加空格（如果后面不是空格或结束）
        text = re.sub(r'(\))([^\s)])', r'\1 \2', text)
        # 左括号后不添加空格（内部内容直接连接）
        text = re.sub(r'\(\s+', '(', text)
        # 右括号前不添加空格
        text = re.sub(r'\s+\)', ')', text)
        
        # 处理其他标点符号
        # 在特定标点后添加空格（如果后面不是空格或结束）
        text = re.sub(r'([,.:;])([^\s])', r'\1 \2', text)
        # 处理分号后添加空格
        text = re.sub(r'(;)([^\s;])', r'\1 \2', text)
        text = re.sub(r'\. \.', '.', text)
        return text

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out