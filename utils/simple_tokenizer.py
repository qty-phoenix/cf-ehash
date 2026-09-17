"""
简单的字符级tokenizer
用于SimpleTextEncoder
"""

import torch
import string
import re


class SimpleCharTokenizer:
    """
    简单的字符级tokenizer
    """
    def __init__(self, max_length=77):
        self.max_length = max_length
        
        # 构建词汇表：字母+数字+标点+特殊字符
        chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(chars, start=2):
            self.char_to_idx[char] = i
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"[SimpleCharTokenizer] 词汇表大小: {self.vocab_size}")
    
    def encode(self, text):
        """
        将文本编码为token IDs
        
        Args:
            text: str, 输入文本
        Returns:
            torch.LongTensor, shape (max_length,)
        """
        # 转换为小写并清理
        text = text.lower()
        
        # 转换为IDs
        ids = []
        for char in text[:self.max_length]:
            ids.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
        
        # Padding
        while len(ids) < self.max_length:
            ids.append(self.char_to_idx['<PAD>'])
        
        return torch.LongTensor(ids[:self.max_length])
    
    def encode_batch(self, texts):
        """
        批量编码
        
        Args:
            texts: List[str], 文本列表
        Returns:
            torch.LongTensor, shape (B, max_length)
        """
        return torch.stack([self.encode(text) for text in texts])
    
    def decode(self, ids):
        """
        将token IDs解码为文本
        
        Args:
            ids: torch.LongTensor or list
        Returns:
            str
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        chars = []
        for idx in ids:
            if idx == self.char_to_idx['<PAD>']:
                break
            chars.append(self.idx_to_char.get(idx, '<UNK>'))
        
        return ''.join(chars)


class SimpleWordTokenizer:
    """
    简单的词级tokenizer（基于空格分词）
    """
    def __init__(self, vocab_size=10000, max_length=77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # 特殊token
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # 词汇表（需要从训练数据构建）
        self.word_to_idx = self.special_tokens.copy()
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        print(f"[SimpleWordTokenizer] 初始化完成，词汇表大小: {len(self.word_to_idx)}")
    
    def build_vocab(self, texts, min_freq=1):
        """
        从文本列表构建词汇表
        
        Args:
            texts: List[str], 文本列表
            min_freq: int, 最小词频
        """
        from collections import Counter
        
        # 统计词频
        word_freq = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_freq.update(words)
        
        # 按频率排序，取top vocab_size
        most_common = word_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        # 构建词汇表
        for word, freq in most_common:
            if freq >= min_freq and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"[SimpleWordTokenizer] 词汇表构建完成，大小: {len(self.word_to_idx)}")
    
    def _tokenize(self, text):
        """简单的分词（基于空格和标点）"""
        # 转换为小写
        text = text.lower()
        # 简单的标点分离
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # 分词
        words = text.split()
        return words
    
    def encode(self, text):
        """
        编码文本
        
        Args:
            text: str
        Returns:
            torch.LongTensor, shape (max_length,)
        """
        words = self._tokenize(text)
        
        # 转换为IDs
        ids = [self.special_tokens['<START>']]
        for word in words[:self.max_length - 2]:
            ids.append(self.word_to_idx.get(word, self.special_tokens['<UNK>']))
        ids.append(self.special_tokens['<END>'])
        
        # Padding
        while len(ids) < self.max_length:
            ids.append(self.special_tokens['<PAD>'])
        
        return torch.LongTensor(ids[:self.max_length])
    
    def encode_batch(self, texts):
        """批量编码"""
        return torch.stack([self.encode(text) for text in texts])
    
    def decode(self, ids):
        """解码"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        words = []
        for idx in ids:
            if idx == self.special_tokens['<PAD>']:
                break
            if idx == self.special_tokens['<START>'] or idx == self.special_tokens['<END>']:
                continue
            words.append(self.idx_to_word.get(idx, '<UNK>'))
        
        return ' '.join(words)


# 便捷函数
def build_tokenizer(tokenizer_type='char', **kwargs):
    """
    构建tokenizer
    
    Args:
        tokenizer_type: 'char' 或 'word'
    """
    if tokenizer_type == 'char':
        return SimpleCharTokenizer(**kwargs)
    elif tokenizer_type == 'word':
        return SimpleWordTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")










































