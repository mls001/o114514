import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class MolecularTransformer(nn.Module):
    def __init__(self, vocab_size, feature_size, hidden_dim=256):
        super().__init__()
        self.feature_proj = nn.Linear(feature_size, hidden_dim)

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=6,
            n_head=8
        )
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, features):
        # 投影特征
        feature_emb = self.feature_proj(features).unsqueeze(1)

        # 生成输入嵌入
        inputs_embeds = self.transformer.wte(input_ids)

        # 拼接特征
        combined_emb = torch.cat([feature_emb, inputs_embeds], dim=1)

        # Transformer处理
        outputs = self.transformer(
            inputs_embeds=combined_emb
        ).last_hidden_state

        # 预测下一个token
        return self.lm_head(outputs[:, 1:])
