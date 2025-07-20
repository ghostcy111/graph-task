import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
np.random.seed(42)


class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GCN'):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'GIN':
            self.nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, out_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(out_channels, out_channels)
            )
            self.conv1 = GINConv(self.nn1)
            self.conv2 = GINConv(self.nn2)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.convs = torch.nn.ModuleList()
        if model_type == 'GIN':
            self.convs.append(self.conv1)
            self.convs.append(self.conv2)
        else:
            self.convs.append(self.conv1)
            self.convs.append(self.conv2)

        self.reset_parameters()

    def reset_parameters(self):
        if self.model_type != 'GIN':
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()
        else:
            for layer in self.nn1:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.nn2:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def encode(self, x, edge_index):
        if isinstance(edge_index, list):
            for i, (edge_index_layer, _, size) in enumerate(edge_index):
                x_target = x[:size[1]]
                if self.model_type == 'GIN':
                    x = self.convs[i]((x, x_target), edge_index_layer)
                else:
                    x = self.convs[i](x, edge_index_layer)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
        else:
            if self.model_type == 'GIN':
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
            else:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)

        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train_full_graph(model, train_data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()


    z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))


    pos_edge_index = train_data.edge_label_index
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    pos_out = model.decode(z, pos_edge_index.to(device))
    neg_out = model.decode(z, neg_edge_index.to(device))
    out = torch.cat([pos_out, neg_out])
    target = torch.cat([
        torch.ones_like(pos_out),
        torch.zeros_like(neg_out)
    ]).to(device)

    loss = criterion(out, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train_sampled_graph(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0  # 添加总样本计数器

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 编码节点表示
        z = model.encode(batch.x, batch.edge_index)

        # 计算当前批次中的样本数（正边+负边）
        if hasattr(batch, 'edge_label_index'):
            num_samples = batch.edge_label_index.size(1)
        else:
            num_samples = batch.edge_index.size(1) * 2

        # 检查batch是否包含edge_label属性
        if hasattr(batch, 'edge_label') and hasattr(batch, 'edge_label_index'):
            # 从batch中提取正负样本
            pos_edge_index = batch.edge_label_index[:, batch.edge_label == 1]
            neg_edge_index = batch.edge_label_index[:, batch.edge_label == 0]
        else:
            # 如果batch中没有提供edge_label，手动生成负样本
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )

        # 解码并计算损失
        pos_out = model.decode(z, pos_edge_index)
        neg_out = model.decode(z, neg_edge_index)
        out = torch.cat([pos_out, neg_out])
        target = torch.cat([
            torch.ones_like(pos_out),
            torch.zeros_like(neg_out)
        ]).to(device)

        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * num_samples  # 使用样本数加权
        total_samples += num_samples

    return total_loss / total_samples  # 返回样本加权平均损失


# 评估函数
@torch.no_grad()
def test(model, data, device):
    model.eval()

    # 编码节点表示
    z = model.encode(data.x.to(device), data.edge_index.to(device))

    # 提取测试集中的正负样本
    if hasattr(data, 'edge_label_index') and hasattr(data, 'edge_label'):
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
    else:
        # 如果没有提供edge_label_index，使用edge_index生成测试样本
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

    # 计算预测结果
    pos_out = model.decode(z, pos_edge_index.to(device))
    neg_out = model.decode(z, neg_edge_index.to(device))

    # 计算准确率和AUC
    y_pred = torch.cat([pos_out, neg_out]).sigmoid()
    y_true = torch.cat([
        torch.ones_like(pos_out),
        torch.zeros_like(neg_out)
    ]).to(device)

    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_true).float().mean().item()
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    return accuracy, auc


# 加载数据集（适配本地路径）
def load_dataset(dataset_name):
    # 本地数据集根目录
    local_root = f"E:/new_student_task/graph/.venv/data/{dataset_name}/{dataset_name}"

    if dataset_name in ['Cora', 'Citeseer']:
        # 检查本地是否存在数据集
        if os.path.exists(local_root):
            print(f"使用本地数据集: {local_root}")
            # 直接从本地路径加载
            dataset = Planetoid(
                root=f"E:/new_student_task/graph/.venv/data/{dataset_name}",
                name=dataset_name,
                transform=NormalizeFeatures()
            )
        else:
            # 如果本地不存在，从网络下载
            print(f"本地数据集不存在，将从网络下载至默认路径")
            dataset = Planetoid(
                root=f"data/{dataset_name}",
                name=dataset_name,
                transform=NormalizeFeatures()
            )

        data = dataset[0]

    elif dataset_name == 'Flickr':
        dataset = Flickr(
            root='data/Flickr',
            transform=NormalizeFeatures()
        )
        data = dataset[0]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 数据分割配置
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,  # 训练集手动生成负样本
        split_labels=True,
        neg_sampling_ratio=1.0
    )

    train_data, val_data, test_data = transform(data)

    # 确保训练数据有edge_label_index属性
    if not hasattr(train_data, 'edge_label_index'):
        train_data.edge_label_index = train_data.edge_index

    # 确保测试数据有必要属性
    if not hasattr(test_data, 'edge_label') or not hasattr(test_data, 'edge_label_index'):
        test_pos_edge_index = test_data.edge_index
        test_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index,
            num_nodes=test_data.num_nodes,
            num_neg_samples=test_pos_edge_index.size(1)
        )
        test_data.edge_label_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)
        test_data.edge_label = torch.cat([
            torch.ones(test_pos_edge_index.size(1)),
            torch.zeros(test_neg_edge_index.size(1))
        ])

    return dataset, train_data, val_data, test_data


# 运行实验
def run_experiment():
    datasets = ['Cora', 'Citeseer']
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
    epochs = 100
    hidden_channels = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    results = {}

    for dataset_name in datasets:
        print(f"\n===========================================================")
        print(f"开始数据集 {dataset_name} 的链路预测实验")

        dataset, train_data, val_data, test_data = load_dataset(dataset_name)
        in_channels = dataset.num_node_features
        out_channels = hidden_channels

        results[dataset_name] = {}

        for model_type in models:
            print(f"\n==============================")
            print(f"当前模型: {model_type}")

            # 初始化模型
            full_model = LinkPredictionModel(in_channels, hidden_channels, out_channels, model_type).to(device)
            sampled_model = LinkPredictionModel(in_channels, hidden_channels, out_channels, model_type).to(device)

            # 定义优化器和损失函数
            full_optimizer = torch.optim.Adam(full_model.parameters(), lr=0.001)
            sampled_optimizer = torch.optim.Adam(sampled_model.parameters(), lr=0.001)
            criterion = torch.nn.BCEWithLogitsLoss()

            # 创建子图采样器
            train_loader = LinkNeighborLoader(
                data=train_data,
                num_neighbors=[15, 10],
                batch_size=1024,
                shuffle=True
            )

            # 全图训练
            print(f"\n----- 全图训练 -----")
            best_full_acc = 0
            full_start_time = time.time()

            for epoch in range(1, epochs + 1):
                loss = train_full_graph(full_model, train_data, full_optimizer, criterion, device)
                test_acc, test_auc = test(full_model, test_data, device)

                if test_acc > best_full_acc:
                    best_full_acc = test_acc

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

            full_end_time = time.time()
            full_time_per_epoch = (full_end_time - full_start_time) / epochs

            # 子图采样训练
            print(f"\n----- 子图采样训练 -----")
            best_sampled_acc = 0
            sampled_start_time = time.time()

            for epoch in range(1, epochs + 1):
                loss = train_sampled_graph(sampled_model, train_loader, sampled_optimizer, criterion, device)
                test_acc, test_auc = test(sampled_model, test_data, device)

                if test_acc > best_sampled_acc:
                    best_sampled_acc = test_acc

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

            sampled_end_time = time.time()
            sampled_time_per_epoch = (sampled_end_time - sampled_start_time) / epochs

            # 记录结果
            results[dataset_name][model_type] = {
                'full_acc': best_full_acc,
                'sampled_acc': best_sampled_acc,
                'full_time': full_time_per_epoch,
                'sampled_time': sampled_time_per_epoch
            }

            print(f"\n{model_type} 在 {dataset_name} 上的最终结果:")
            print(f"全图训练准确率: {best_full_acc:.4f}, 每轮耗时: {full_time_per_epoch:.4f}s")
            print(f"子图采样训练准确率: {best_sampled_acc:.4f}, 每轮耗时: {sampled_time_per_epoch:.4f}s")

    # 可视化结果
    visualize_results(results)

    return results


# 可视化实验结果
def visualize_results(results):
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs('results/link_prediction', exist_ok=True)

    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    for dataset_name in datasets:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'数据集 {dataset_name} 链路预测实验结果对比', fontsize=16)
        x = np.arange(len(models))
        width = 0.35

        # 准确率对比图
        full_accs = [results[dataset_name][m]['full_acc'] for m in models]
        sampled_accs = [results[dataset_name][m]['sampled_acc'] for m in models]
        ax1.bar(x - width / 2, full_accs, width, label='全图训练')
        ax1.bar(x + width / 2, sampled_accs, width, label='子图训练')
        ax1.set_title('测试准确率对比', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=30)
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0.4, 1.0)
        ax1.legend()

        for i, v in enumerate(full_accs):
            ax1.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate(sampled_accs):
            ax1.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center')

        # 时间对比图
        full_times = [results[dataset_name][m]['full_time'] for m in models]
        sampled_times = [results[dataset_name][m]['sampled_time'] for m in models]
        ax2.bar(x - width / 2, full_times, width, label='全图训练')
        ax2.bar(x + width / 2, sampled_times, width, label='子图训练')
        ax2.set_title('平均每轮训练时间对比 (秒)', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=30)
        ax2.set_ylabel('时间 (秒)')
        ax2.legend()

        for i, v in enumerate(full_times):
            ax2.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate(sampled_times):
            ax2.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(f'results/link_prediction/{dataset_name}_comparison.png', dpi=300)
        print(f"\n数据集 {dataset_name} 的链路预测结果图已保存至 results/link_prediction/{dataset_name}_comparison.png")
        plt.close()


if __name__ == "__main__":
    results = run_experiment()

    # 打印最终结果汇总
    print("\n\n===========================================================")
    print("链路预测实验结果汇总:")
    for dataset_name, models_data in results.items():
        print(f"\n数据集: {dataset_name}")
        print("模型\t\t全图准确率\t子图准确率\t全图时间\t子图时间")
        for model_type, metrics in models_data.items():
            print(
                f"{model_type}\t\t{metrics['full_acc']:.4f}\t\t{metrics['sampled_acc']:.4f}\t\t{metrics['full_time']:.4f}s\t\t{metrics['sampled_time']:.4f}s")