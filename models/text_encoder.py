# Text Encoder for Text-Guided Manager Network
# 支持多种文本编码方式：CLIP、简单Transformer、LSTM等

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTextEncoder(nn.Module):
    """
    简单的文本编码器：使用预训练的词向量 + LSTM/Transformer
    适用于不需要大型预训练模型的场景
    """
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, output_dim=256, 
                 encoder_type='lstm', num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                embed_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            encoder_output_dim = hidden_dim * 2  # bidirectional
        elif encoder_type == 'gru':
            self.encoder = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            encoder_output_dim = hidden_dim * 2
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            encoder_output_dim = embed_dim
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        
        # 投影层：将编码器输出映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(encoder_output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, text_ids, mask=None):
        """
        Args:
            text_ids: (B, seq_len) 文本token IDs
            mask: (B, seq_len) 可选的attention mask
        Returns:
            text_features: (B, output_dim) 文本特征向量
        """
        # 词嵌入
        embedded = self.embedding(text_ids)  # (B, seq_len, embed_dim)
        
        if self.encoder_type in ['lstm', 'gru']:
            # LSTM/GRU编码
            output, (hidden, _) = self.encoder(embedded) if self.encoder_type == 'lstm' else self.encoder(embedded)
            # 使用最后一层的hidden states（拼接正向和反向）
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            # hidden: (num_layers*2, B, hidden_dim)
            # 取最后一层的正向和反向hidden state
            text_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B, hidden_dim*2)
        elif self.encoder_type == 'transformer':
            # Transformer编码
            if mask is not None:
                # 转换mask格式 (padding位置为True)
                key_padding_mask = (mask == 0)
                output = self.encoder(embedded, src_key_padding_mask=key_padding_mask)
            else:
                output = self.encoder(embedded)
            # 使用mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(output.size())
                sum_out = torch.sum(output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                text_features = sum_out / sum_mask  # (B, embed_dim)
            else:
                text_features = output.mean(dim=1)  # (B, embed_dim)
        
        # 投影到目标维度
        text_features = self.projection(text_features)  # (B, output_dim)
        
        return text_features


class CLIPTextEncoder(nn.Module):
    """
    使用CLIP预训练模型的文本编码器
    需要安装：pip install transformers
    """
    def __init__(self, model_name='openai/clip-vit-base-patch32', output_dim=512, freeze=True, use_mirror=True):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")
        
        # 使用国内镜像源加速下载
        import os
        if use_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"使用HuggingFace镜像源: https://hf-mirror.com")
        
        print(f"正在加载CLIP模型: {model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, resume_download=True)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, resume_download=True)
        self.output_dim = output_dim
        
        # 是否冻结CLIP参数
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 如果输出维度不匹配，添加投影层
        clip_dim = self.text_encoder.config.hidden_size
        if clip_dim != output_dim:
            self.projection = nn.Linear(clip_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, text_descriptions):
        """
        Args:
            text_descriptions: list of strings or pre-tokenized ids
        Returns:
            text_features: (B, output_dim)
        """
        if isinstance(text_descriptions, list):
            # 如果输入是文本列表，进行tokenize
            inputs = self.tokenizer(
                text_descriptions,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=77
            ).to(next(self.text_encoder.parameters()).device)
            
            outputs = self.text_encoder(**inputs)
            # 使用pooled output (CLS token)
            text_features = outputs.pooler_output  # (B, clip_dim)
        else:
            # 假设已经是tokenized的IDs
            outputs = self.text_encoder(text_descriptions)
            text_features = outputs.pooler_output
        
        # 投影到目标维度
        text_features = self.projection(text_features)  # (B, output_dim)
        
        return text_features


class DummyTextEncoder(nn.Module):
    """占位符：不使用文本时的空编码器"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, text_input):
        # 返回全零向量
        if isinstance(text_input, list):
            batch_size = len(text_input)
        else:
            batch_size = text_input.shape[0]
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        return torch.zeros(batch_size, self.output_dim, device=device)


def build_text_encoder(cfg):
    """
    根据配置构建文本编码器
    
    cfg示例:
    {
        'text_encoder_type': 'simple',  # 'simple', 'clip', 'none'
        'text_encoder_model': 'lstm',   # 'lstm', 'gru', 'transformer' (for simple)
        'text_embed_dim': 256,
        'text_hidden_dim': 512,
        'text_output_dim': 256,
        'text_encoder_layers': 2,
        'text_encoder_dropout': 0.1,
        'clip_model_name': 'openai/clip-vit-base-patch32',  # for clip
        'clip_freeze': True
    }
    """
    encoder_type = cfg.get('text_encoder_type', 'none')
    output_dim = cfg.get('text_output_dim', 256)
    
    if encoder_type == 'none':
        return DummyTextEncoder(output_dim)
    elif encoder_type == 'simple':
        return SimpleTextEncoder(
            vocab_size=cfg.get('text_vocab_size', 10000),
            embed_dim=cfg.get('text_embed_dim', 256),
            hidden_dim=cfg.get('text_hidden_dim', 512),
            output_dim=output_dim,
            encoder_type=cfg.get('text_encoder_model', 'lstm'),
            num_layers=cfg.get('text_encoder_layers', 2),
            dropout=cfg.get('text_encoder_dropout', 0.1)
        )
    elif encoder_type == 'clip':
        return CLIPTextEncoder(
            model_name=cfg.get('clip_model_name', 'openai/clip-vit-base-patch32'),
            output_dim=output_dim,
            freeze=cfg.get('clip_freeze', True),
            use_mirror=cfg.get('clip_use_mirror', True)  # 默认使用镜像
        )
    else:
        raise ValueError(f"Unsupported text_encoder_type: {encoder_type}")


