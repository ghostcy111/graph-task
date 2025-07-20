import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
import pandas as pd


torch.manual_seed(42)
np.random.seed(42)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class GraphClassificationModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model_type, pooling_type):
        super().__init__()
        self.model_type = model_type
        self.pooling_type = pooling_type

        # 输入层
        if model_type == 'GCN':
            self.convs = torch.nn.ModuleList([
                GCNConv(in_channels, hidden_channels)
            ])
        elif model_type == 'GAT':
            self.convs = torch.nn.ModuleList([
                GATConv(in_channels, hidden_channels)
            ])
        elif model_type == 'GraphSAGE':
            self.convs = torch.nn.ModuleList([
                SAGEConv(in_channels, hidden_channels)
            ])
        elif model_type == 'GIN':
            self.convs = torch.nn.ModuleList([
                GINConv(torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),  # 添加BatchNorm提高稳定性
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                ))
            ])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        for _ in range(num_layers - 1):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_channels, hidden_channels))
            elif model_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif model_type == 'GIN':
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),  # 添加BatchNorm提高稳定性
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )))

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if self.pooling_type == 'avg':
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling_type == 'min':
            x = -global_max_pool(-x, batch)
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")

        # 全连接分类层
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    nan_detected = False
    batch_count = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)

        if torch.isnan(out).any():
            print("警告: 模型输出包含nan，跳过此批次")
            nan_detected = True
            continue

        loss = criterion(out, data.y)

        if torch.isnan(loss):
            print("警告: 损失值为nan，跳过此批次")
            nan_detected = True
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        batch_count += 1

    if batch_count == 0:
        return float('nan')
    return total_loss / (sum(data.num_graphs for data in train_loader) if not nan_detected else 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)


        if torch.isnan(out).any():
            print("警告: 评估时模型输出包含nan")
            continue

        pred = out.argmax(dim=1)  # 多分类问题
        correct += int((pred == data.y).sum())
        total += data.num_graphs

    return correct / total if total > 0 else 0


@torch.no_grad()
def evaluate_rmse(model, loader, device):
    model.eval()
    mse = 0
    total = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch).view(-1)

        if torch.isnan(out).any():
            print("警告: 评估时模型输出包含nan")
            continue

        mse += F.mse_loss(out, data.y, reduction='sum').item()
        total += data.num_graphs

    return np.sqrt(mse / total) if total > 0 else float('nan')


# 加载数据集
def load_dataset(dataset_name):
    if dataset_name.startswith('TU_'):
        # TU数据集处理
        name = dataset_name[3:]  # 去掉'TU_'前缀
        dataset = TUDataset(root=f'data/TUDataset/{name}', name=name)

        if dataset.num_node_features == 0:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            mean_deg = torch.cat(degs).float().mean().item()
            std_deg = torch.cat(degs).float().std().item()

            dataset.transform = NormalizedDegree(mean_deg, std_deg)

        dataset = dataset.shuffle()
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
        test_dataset = dataset[int(len(dataset) * 0.9):]

        return dataset, train_dataset, val_dataset, test_dataset

    elif dataset_name == 'ZINC':
        transform = T.Compose([
            T.NormalizeFeatures(),
        ])
        dataset = ZINC(root='data/ZINC', subset=True, transform=transform)

        dataset = dataset.shuffle()
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
        test_dataset = dataset[int(len(dataset) * 0.9):]

        return dataset, train_dataset, val_dataset, test_dataset

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def run_experiment():
    # 实验配置
    datasets = ['TU_MUTAG', 'TU_PROTEINS']#排除‘ZINC’
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
    pooling_types = ['avg', 'max', 'min']
    hidden_channels = 64
    num_layers = 3
    batch_size = 32
    initial_lr = 0.000001  #学习率
    epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    results = {}

    for dataset_name in datasets:
        print(f"\n===========================================================")
        print(f"开始数据集 {dataset_name} 的图分类实验")

        dataset, train_dataset, val_dataset, test_dataset = load_dataset(dataset_name)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        in_channels = dataset.num_node_features
        if dataset_name == 'ZINC':
            # ZINC是回归任务
            out_channels = 1
            criterion = torch.nn.MSELoss()
            metric_name = 'RMSE'
        else:
            # TU数据集是分类任务
            out_channels = dataset.num_classes
            criterion = torch.nn.CrossEntropyLoss()
            metric_name = 'Accuracy'

        results[dataset_name] = {}

        for model_type in models:
            results[dataset_name][model_type] = {}

            for pooling_type in pooling_types:
                print(f"\n------------------------------")
                print(f"模型: {model_type}, 池化方法: {pooling_type}")

                # 初始化模型
                model = GraphClassificationModel(
                    in_channels, hidden_channels, out_channels, num_layers, model_type, pooling_type
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

                # 添加学习率调度器（修复参数传递问题）
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min' if metric_name == 'RMSE' else 'max',
                    factor=0.5, patience=5, verbose=True
                )

                best_val_metric = float('inf') if metric_name == 'RMSE' else 0
                best_test_metric = 0
                best_epoch = 0
                training_time = 0
                nan_epochs = 0

                for epoch in range(1, epochs + 1):
                    start_time = time.time()
                    loss = train(model, train_loader, optimizer, criterion, device)
                    epoch_time = time.time() - start_time
                    training_time += epoch_time

                    # 如果损失为nan，增加计数
                    if np.isnan(loss):
                        nan_epochs += 1
                        print(f"Epoch {epoch}: 损失为nan，跳过评估")

                        # 如果连续3个epoch都是nan，提前终止
                        if nan_epochs >= 3:
                            print("警告: 连续3个epoch损失为nan，终止训练")
                            break
                        continue
                    else:
                        nan_epochs = 0  # 重置nan计数

                    # 评估模型
                    if metric_name == 'RMSE':
                        val_rmse = evaluate_rmse(model, val_loader, device)
                        test_rmse = evaluate_rmse(model, test_loader, device)

                        if val_rmse < best_val_metric:
                            best_val_metric = val_rmse
                            best_test_metric = test_rmse
                            best_epoch = epoch

                        print(
                            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val {metric_name}: {val_rmse:.4f}, Test {metric_name}: {test_rmse:.4f}")

                        # 更新学习率（正确传递metrics参数）
                        scheduler.step(val_rmse)
                    else:
                        val_acc = evaluate(model, val_loader, device)
                        test_acc = evaluate(model, test_loader, device)

                        if val_acc > best_val_metric:
                            best_val_metric = val_acc
                            best_test_metric = test_acc
                            best_epoch = epoch

                        print(
                            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val {metric_name}: {val_acc:.4f}, Test {metric_name}: {test_acc:.4f}")

                        # 更新学习率（正确传递metrics参数）
                        scheduler.step(val_acc)

                avg_epoch_time = training_time / max(1, epoch)  # 避免除零错误
                results[dataset_name][model_type][pooling_type] = {
                    'metric': best_test_metric,
                    'epoch_time': avg_epoch_time,
                    'best_epoch': best_epoch
                }

                print(
                    f"\n最佳结果: Epoch {best_epoch}, Test {metric_name}: {best_test_metric:.4f}, 平均每轮耗时: {avg_epoch_time:.4f}s")

    # 可视化结果
    visualize_results(results)

    return results


# 可视化实验结果
def visualize_results(results):
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs('results/graph_classification', exist_ok=True)

    for dataset_name, models_data in results.items():
        # 创建结果表格
        metrics_df = pd.DataFrame(columns=['Model', 'AvgPool', 'MaxPool', 'MinPool'])
        time_df = pd.DataFrame(columns=['Model', 'AvgPool', 'MaxPool', 'MinPool'])

        for i, (model_type, pooling_data) in enumerate(models_data.items()):
            metrics = [pooling_data[pooling]['metric'] for pooling in ['avg', 'max', 'min']]
            times = [pooling_data[pooling]['epoch_time'] for pooling in ['avg', 'max', 'min']]

            metrics_df.loc[i] = [model_type] + metrics
            time_df.loc[i] = [model_type] + times

        # 保存结果表格
        metrics_df.to_csv(f'results/graph_classification/{dataset_name}_metrics.csv', index=False)
        time_df.to_csv(f'results/graph_classification/{dataset_name}_times.csv', index=False)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'数据集 {dataset_name} 图分类实验结果对比', fontsize=16)

        # 准确率/误差对比图
        x = np.arange(len(models_data))
        width = 0.25

        for i, pooling in enumerate(['avg', 'max', 'min']):
            values = [models_data[model][pooling]['metric'] for model in models_data]
            ax1.bar(x + i * width, values, width, label=f'{pooling.upper()}池化')

        ax1.set_title('测试性能对比', fontsize=14)
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(list(models_data.keys()), rotation=30)
        ax1.set_ylabel('准确率' if dataset_name.startswith('TU_') else 'RMSE')
        ax1.legend()

        # 添加数值标签
        for i, v in enumerate([models_data[model]['avg']['metric'] for model in models_data]):
            ax1.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate([models_data[model]['max']['metric'] for model in models_data]):
            ax1.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate([models_data[model]['min']['metric'] for model in models_data]):
            ax1.text(i + 3 * width / 2, v + 0.01, f'{v:.4f}', ha='center')

        # 训练时间对比图
        for i, pooling in enumerate(['avg', 'max', 'min']):
            values = [models_data[model][pooling]['epoch_time'] for model in models_data]
            ax2.bar(x + i * width, values, width, label=f'{pooling.upper()}池化')

        ax2.set_title('平均每轮训练时间对比 (秒)', fontsize=14)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(list(models_data.keys()), rotation=30)
        ax2.set_ylabel('时间 (秒)')
        ax2.legend()

        # 添加数值标签
        for i, v in enumerate([models_data[model]['avg']['epoch_time'] for model in models_data]):
            ax2.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate([models_data[model]['max']['epoch_time'] for model in models_data]):
            ax2.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate([models_data[model]['min']['epoch_time'] for model in models_data]):
            ax2.text(i + 3 * width / 2, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(f'results/graph_classification/{dataset_name}_comparison.png', dpi=300)
        print(
            f"\n数据集 {dataset_name} 的图分类结果图已保存至 results/graph_classification/{dataset_name}_comparison.png")
        plt.close()


if __name__ == "__main__":
    results = run_experiment()

    # 打印最终结果汇总
    print("\n\n===========================================================")
    print("图分类实验结果汇总:")
    for dataset_name, models_data in results.items():
        print(f"\n数据集: {dataset_name}")
        print("模型\t\t池化方法\t性能指标\t每轮耗时\t最佳轮次")
        for model_type, pooling_data in models_data.items():
            for pooling_type, metrics in pooling_data.items():
                metric_name = '准确率' if dataset_name.startswith('TU_') else 'RMSE'
                print(
                    f"{model_type}\t\t{pooling_type.upper()}\t\t{metrics['metric']:.4f}\t\t{metrics['epoch_time']:.4f}s\t\t{metrics['best_epoch']}")