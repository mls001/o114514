import torch

a = [1.0, 2.0, 3.0, 4.0]
print(a)
print(torch.is_tensor(a))

b = torch.tensor([1.0, 2.0, 3.0])
print(b)
print(torch.is_tensor(b))

c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(c)
print(torch.is_tensor(c))

d = torch.zeros(3, 3)
print(d)

f = torch.eye(4, 4)
print(f)
print(f.storage())
print(f.storage()[0])
g = f.storage()
print(g[0])
