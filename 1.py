import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from rdkit import Chem
import numpy as np


# 1. 数据准备与处理
class MolecularDataset(Dataset):
    def __init__(self, smiles_list, properties, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = []
        self.properties = []

        for smi, prop in zip(smiles_list, properties):
            tokenized = self.tokenizer(smi, padding='max_length',
                                       max_length=max_length,
                                       return_tensors='pt')
            if tokenized.input_ids.shape[1] <= max_length:
                self.smiles.append(tokenized)
                self.properties.append(torch.tensor(prop, dtype=torch.float32))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'input_ids': self.smiles[idx].input_ids.squeeze(),
            'attention_mask': self.smiles[idx].attention_mask.squeeze(),
            'properties': self.properties[idx]
        }


# 2. 条件Transformer模型架构
class GaussianTransformer(nn.Module):
    def __init__(self, property_dim=256, vocab_size=256,
                 embedding_dim=512, n_head=8, n_layer=12):
        super().__init__()

        # 属性编码器
        self.property_encoder = nn.Sequential(
            nn.Linear(property_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # Transformer模型
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embedding_dim,
            n_head=n_head,
            n_layer=n_layer,
            add_cross_attention=True
        )
        self.transformer = GPT2Model(config)

        # 输出层
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, properties):
        # 编码属性
        prop_emb = self.property_encoder(properties)

        # 通过Transformer
        outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=prop_emb.unsqueeze(1)
        )

        logits = self.output(outputs.last_hidden_state)
        return logits


# 3. 训练流程
def train_model(dataset, epochs=100, batch_size=32):
    # 初始化模型
    model = GaussianTransformer(property_dim=dataset[0]['properties'].shape[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch['input_ids']
            props = batch['properties']

            # 前向传播
            outputs = model(inputs, props)

            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             inputs.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

    return model


# 4. 分子生成与验证
def generate_molecule(model, properties, tokenizer, max_length=128):
    model.eval()
    generated = torch.zeros((1, max_length), dtype=torch.long)
    properties = torch.tensor(properties, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for i in range(max_length - 1):
            outputs = model(generated[:, :i + 1], properties)
            next_token = torch.argmax(outputs[0, i], dim=-1)
            generated[0, i + 1] = next_token

            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0].tolist())


# 使用示例
if __name__ == "__main__":
    # 假设已有数据集
    smiles = ["CCO", "CCN", "C1CCCCC1"]  # 示例SMILES
    properties = np.random.randn(3, 256)  # 高斯计算结果特征


    # 初始化自定义tokenizer（需要实现）
    class SMILESTokenizer:
        # 实现SMILES字符到ID的转换
        pass


    tokenizer = SMILESTokenizer()
    dataset = MolecularDataset(smiles, properties, tokenizer)

    # 训练模型
    trained_model = train_model(dataset)

    # 生成新分子
    new_properties = np.random.randn(256)  # 目标属性
    generated_smiles = generate_molecule(trained_model, new_properties, tokenizer)

    # 验证分子有效性
    mol = Chem.MolFromSmiles(generated_smiles)
    if mol:
        print(f"Valid molecule generated: {generated_smiles}")
    else:
        print("Invalid SMILES generated")
