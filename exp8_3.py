# 1. 基于Pytorch框架实现深层架构的搭建。
# 2. 基于RNA序列数据集实验，并撰写小论文详细汇报所用方法、实验设置和最终分类正确率。
# 3. 网络的深层架构自由设计，充分利用学过的所有深度学习方法搭建网络，并用相关优化方法不断优化模型。
# 相关超参数的设置根据实验结果进一步优化调整，最后选择结果最优的设置在小论文中汇报。
import numpy as np
import torch
import preprocess
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn


# 训练样本数据
posi = r".\ALKBH5_Baltz2012.train.positives.fa"
nega = r".\ALKBH5_Baltz2012.train.negatives.fa"
train_bags, train_labels = preprocess.get_data(posi, nega, window_size = 501)
# (2170, 1, 4, 507)     (2170,)

# 数据打乱
train_bags, train_labels = np.array(train_bags), np.array(train_labels)
shuffle_index = torch.randperm(2170)
train_x = np.zeros_like(train_bags)
train_y = np.zeros_like(train_labels)
for i in range(2170):
    train_x[i] = train_bags[shuffle_index[i]]
    train_y[i] = train_labels[shuffle_index[i]]
train_bags,train_labels = train_x,train_y



# 测试样本数据
val_p = r'.\ALKBH5_Baltz2012.val.positives.fa'
val_n = r'.\ALKBH5_Baltz2012.val.negatives.fa'
val_bags, val_labels = preprocess.get_data(val_p, val_n, window_size = 501)

# # 数据打乱
# val_bags, val_labels = np.array(val_bags), np.array(val_labels)
# shuffle_index = torch.randperm(240)
# val_x = np.zeros_like(val_bags)
# val_y = np.zeros_like(val_labels)
# for i in range(240):
#     val_x[i] = val_bags[shuffle_index[i]]
#     val_y[i] = val_labels[shuffle_index[i]]
# val_bags,val_labels = val_x,val_y

# 转换数据为Tensor
train_bags = torch.tensor(np.array(train_bags), dtype=torch.float32)
train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
val_bags = torch.tensor(np.array(val_bags), dtype=torch.float32)
val_labels = torch.tensor(np.array(val_labels), dtype=torch.long)



# 简化模型
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),
    # nn.BatchNorm2d(6),
    nn.ReLU(),  # ([1, 32, 2, 505])
    nn.MaxPool2d(kernel_size=2,padding=(1,0)),  # ([1, 32, 2, 252])
    # nn.Conv2d(6, 16, kernel_size=2,padding=(1,0)),
    # nn.ReLU(),  # ([1, 16, 5, 252])
    #
    # nn.MaxPool2d(kernel_size=2),  # ([1, 16, 2, 126])
    # nn.BatchNorm2d(16),
    nn.Flatten(),  # ([1, 16*2*126])
    nn.Linear(32 * 2 * 252, 120),
    nn.BatchNorm1d(120),
    nn.ReLU(),
    # nn.Dropout(0.1),  # 添加Dropout层，丢弃概率为0.2
    nn.Linear(120, 60),
    nn.ReLU(),
    # nn.BatchNorm1d(84),
    nn.Linear(60, 2)
)




# 打印模型结构
# print(model)
# X = torch.rand(size=(1, 1, 5, 507), dtype=torch.float32)
# for layer in model:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape: \t',X.shape)






# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# for name, param in model.named_parameters():
#        print(name, "   ", param)


# 训练模型
# num_epochs = 10
# batch_size = 32
num_epochs = 10
batch_size = 128
# print(train_bags.shape)


for epoch in range(num_epochs):
    running_loss = 0.0
    num_batches = len(train_bags) // batch_size

    for i in range(num_batches):
        # 提取当前批次的数据和标签
        batch_bags = train_bags[i * batch_size : (i+1) * batch_size]
        batch_labels = train_labels[i * batch_size : (i+1) * batch_size]

        # 前向传播
        outputs = model(batch_bags)
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印训练信息
    average_loss = running_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")






# 使用验证集测试   设置模型为评估模式
model.eval()

# 在无需计算梯度的情况下进行预测
with torch.no_grad():
    # 前向传播并获取预测结果
    outputs = model(val_bags)
    _, predicted = torch.max(outputs, 1)
print(predicted)

# 计算预测准确率
total = val_labels.size(0)
correct = (predicted == val_labels).sum().item()
accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")

