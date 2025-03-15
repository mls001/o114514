import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from a2 import MolecularTransformer


class MoleculeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=100):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        tokens = self.tokenizer.encode(item['smiles'])
        features = torch.FloatTensor(item['features'])

        return {
            'input_ids': torch.LongTensor(tokens[:-1]),
            'labels': torch.LongTensor(tokens[1:]),
            'features': features
        }


# 初始化模型和优化器
model = MolecularTransformer(vocab_size=128, feature_size=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    for batch in train_loader:
        outputs = model(
            input_ids=batch['input_ids'],
            features=batch['features']
        )

        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)),
            batch['labels'].view(-1)
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()