import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


### 1. 数据集加载模块
def load_dataset(dataset_name):
    if dataset_name in ['Cora', 'Citeseer']:
        # 加载中小型引文网络数据集
        dataset = Planetoid(
            root=f'data/{dataset_name}',
            name=dataset_name,
            transform=NormalizeFeatures()  # 特征归一化
        )
    elif dataset_name == 'Flickr':
        # 加载大型社交网络数据集
        dataset = Flickr(
            root='data/Flickr',
            transform=NormalizeFeatures()
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    data = dataset[0].to(device)
    print(f"\n数据集 {dataset_name} 信息:")
    print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
    print(f"特征维度: {data.num_node_features}, 类别数: {dataset.num_classes}")
    print(f"训练集大小: {data.train_mask.sum().item()}, 测试集大小: {data.test_mask.sum().item()}")
    return dataset, data


class GCN(torch.nn.Module):
    """图卷积网络(GCN)"""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        # 第一层使用多头注意力
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # 第二层使用单头注意力输出类别
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # GAT推荐使用ELU激活函数
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # GIN的聚合函数使用MLP
        self.mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),  # 批归一化加速训练
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


### 3. 训练与评估函数
def train_full_graph(model, data, optimizer, criterion, epochs=200):
    model.train()
    total_time = 0.0  # 总训练时间
    best_val_acc = 0.0  # 最佳验证集准确率
    best_test_acc = 0.0  # 对应最佳验证集的测试集准确率

    for epoch in range(epochs):
        start_time = time.time()

        # 前向传播与优化
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 全图输入
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time

        # 验证与测试（每20轮打印一次）
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

        # 更新最佳指标
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | 耗时: {epoch_time:.4f}s")

    avg_time_per_epoch = total_time / epochs
    return best_test_acc, avg_time_per_epoch


def train_sampled_graph(model, data, optimizer, criterion, num_layers=2, batch_size=1024, epochs=200):
    model.train()
    total_time = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0

    # 定义子图采样器：每层采样20个邻居
    loader = NeighborLoader(
        data,
        num_neighbors=[20] * num_layers,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0

        # 小批量训练
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        total_time += epoch_time


        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

        # 更新最佳指标
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Avg Loss: {total_loss / len(loader):.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | 耗时: {epoch_time:.4f}s")

    avg_time_per_epoch = total_time / epochs
    return best_test_acc, avg_time_per_epoch


def run_experiment():
    """运行完整实验并记录结果"""
    # 实验配置
    datasets = ['Cora','Citeseer']
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'GIN': GIN
    }  # 模型字典
    hidden_channels = 128  # 隐藏层维度
    epochs = 200  # 训练轮数
    lr = 0.01  # 学习率
    weight_decay = 5e-4  # 权重衰减（L2正则化）

    # 存储实验结果：{数据集: {模型: {全图准确率, 全图时间, 子图准确率, 子图时间}}}
    results = {ds: {m: {'full_acc': 0, 'full_time': 0, 'sampled_acc': 0, 'sampled_time': 0} for m in models} for ds in
               datasets}

    # 遍历所有数据集和模型
    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"开始数据集 {dataset_name} 的实验")
        dataset, data = load_dataset(dataset_name)
        in_channels = dataset.num_node_features  # 输入特征维度
        out_channels = dataset.num_classes  # 输出类别数

        for model_name, ModelClass in models.items():
            print(f"\n{'=' * 30}")
            print(f"当前模型: {model_name}")

            # 全图训练
            print("\n----- 全图训练 -----")
            model = ModelClass(in_channels, hidden_channels, out_channels).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            full_test_acc, full_time = train_full_graph(model, data, optimizer, criterion, epochs=epochs)

            # 子图采样训练
            print("\n----- 子图采样训练 -----")
            model = ModelClass(in_channels, hidden_channels, out_channels).to(device)  # 重新初始化模型
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sampled_test_acc, sampled_time = train_sampled_graph(model, data, optimizer, criterion, epochs=epochs)

            # 保存结果
            results[dataset_name][model_name]['full_acc'] = full_test_acc
            results[dataset_name][model_name]['full_time'] = full_time
            results[dataset_name][model_name]['sampled_acc'] = sampled_test_acc
            results[dataset_name][model_name]['sampled_time'] = sampled_time

            # 打印单模型结果
            print(f"\n{model_name} 实验结果:")
            print(f"全图训练 - 测试准确率: {full_test_acc:.4f}, 平均每轮时间: {full_time:.4f}s")
            print(f"子图训练 - 测试准确率: {sampled_test_acc:.4f}, 平均每轮时间: {sampled_time:.4f}s")
            print(f"准确率差异: {sampled_test_acc - full_test_acc:.4f}, 时间加速比: {full_time / sampled_time:.2f}x")

    # 保存并可视化结果
    visualize_results(results)
    return results


def visualize_results(results):
    """可视化实验结果（解决中文显示问题）"""
    os.makedirs('results', exist_ok=True)
    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文的字体列表
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    for dataset_name in datasets:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'数据集 {dataset_name} 实验结果对比', fontsize=16)
        x = np.arange(len(models))
        width = 0.35

        # 准确率对比图
        full_accs = [results[dataset_name][m]['full_acc'] for m in models]
        sampled_accs = [results[dataset_name][m]['sampled_acc'] for m in models]
        ax1.bar(x - width/2, full_accs, width, label='全图训练')
        ax1.bar(x + width/2, sampled_accs, width, label='子图训练')
        ax1.set_title('测试准确率对比', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=30)
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0.5, 1.0)
        ax1.legend()

        # 时间对比图
        full_times = [results[dataset_name][m]['full_time'] for m in models]
        sampled_times = [results[dataset_name][m]['sampled_time'] for m in models]
        ax2.bar(x - width/2, full_times, width, label='全图训练')
        ax2.bar(x + width/2, sampled_times, width, label='子图训练')
        ax2.set_title('平均每轮训练时间对比 (秒)', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=30)
        ax2.set_ylabel('时间 (秒)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}_comparison.png', dpi=300)
        print(f"\n数据集 {dataset_name} 的结果图已保存至 results/{dataset_name}_comparison.png")
        plt.close()

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("实验结果汇总表")
    print("=" * 80)
    for dataset_name in datasets:
        print(f"\n数据集: {dataset_name}")
        print(
            f"{'模型':10s} | {'全图准确率':12s} | {'子图准确率':12s} | {'准确率变化':12s} | {'全图时间(s)':15s} | {'子图时间(s)':15s} | {'加速比':8s}")
        print("-" * 80)
        for model_name in models:
            f_acc = results[dataset_name][model_name]['full_acc']
            s_acc = results[dataset_name][model_name]['sampled_acc']
            f_time = results[dataset_name][model_name]['full_time']
            s_time = results[dataset_name][model_name]['sampled_time']
            acc_diff = s_acc - f_acc
            speedup = f_time / s_time if s_time > 0 else 0
            print(
                f"{model_name:10s} | {f_acc:.4f}          | {s_acc:.4f}          | {acc_diff:+.4f}          | {f_time:.4f}              | {s_time:.4f}              | {speedup:.2f}x")


if __name__ == "__main__":
    run_experiment()