import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class KGDataset(Dataset):
    def __init__(self, triples, entities, relations, neg_ratio=1):
        self.triples = triples
        self.entities = entities
        self.relations = relations
        self.neg_ratio = neg_ratio
        self.entity_list = list(entities)
        self.entity_set = set(entities)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        neg_samples = []
        for _ in range(self.neg_ratio):
            if random.random() < 0.5:
                h_neg = random.sample([e for e in self.entity_list if e != h], 1)[0]
                t_neg = t
            else:
                t_neg = random.sample([e for e in self.entity_list if e != t], 1)[0]
                h_neg = h
            neg_samples.append((h_neg, r, t_neg))

        return (h, r, t,
                [neg[0] for neg in neg_samples],
                [neg[2] for neg in neg_samples])


def collate_fn(batch):
    h_pos = torch.tensor([item[0] for item in batch], dtype=torch.long)
    r = torch.tensor([item[1] for item in batch], dtype=torch.long)
    t_pos = torch.tensor([item[2] for item in batch], dtype=torch.long)

    h_neg = []
    t_neg = []
    for item in batch:
        h_neg.extend(item[3])
        t_neg.extend(item[4])

    h_neg = torch.tensor(h_neg, dtype=torch.long)
    t_neg = torch.tensor(t_neg, dtype=torch.long)

    return h_pos, r, t_pos, h_neg, t_neg


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, p_norm=1):
        super(TransE, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.p_norm = p_norm

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def forward(self, h, r, t):
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)

        h_emb = F.normalize(h_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)

        score = torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=1)
        return -score


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(RotatE, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def forward(self, h, r, t):
        half_dim = self.embedding_dim // 2
        h_re, h_im = torch.chunk(self.entity_embedding(h), 2, dim=1)
        t_re, t_im = torch.chunk(self.entity_embedding(t), 2, dim=1)

        r_phase = self.relation_embedding(r)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re

        diff_re = h_rot_re - t_re
        diff_im = h_rot_im - t_im
        score = torch.norm(torch.cat([diff_re, diff_im], dim=1), p=2, dim=1)

        return -score


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, hidden_channels=32, kernel_size=3):
        super(ConvE, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        flat_size = (embedding_dim // 2) * embedding_dim * hidden_channels
        self.fc = nn.Linear(flat_size, embedding_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, h, r, t=None):
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)

        batch_size = h_emb.shape[0]
        h_emb = h_emb.view(-1, 1, self.embedding_dim // 2, 2)
        r_emb = r_emb.view(-1, 1, self.embedding_dim // 2, 2)

        stacked_inputs = torch.cat([h_emb, r_emb], 2)

        x = self.bn0(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        if t is not None:
            t_emb = self.entity_embedding(t)
            scores = torch.sum(x * t_emb, dim=1)
            return scores
        else:
            return x


def margin_ranking_loss(pos_scores, neg_scores, margin=1.0):
    neg_scores = neg_scores.view(pos_scores.shape[0], -1)
    max_neg_scores, _ = torch.max(neg_scores, dim=1)
    loss = torch.mean(torch.clamp(max_neg_scores - pos_scores + margin, min=0))
    return loss


def train(model, train_loader, optimizer, device, margin=1.0, model_type='TransE'):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=True):  # 添加此行
        h_pos, r, t_pos, h_neg, t_neg = [x.to(device) for x in batch]

        batch_size = h_pos.shape[0]
        neg_ratio = h_neg.shape[0] // batch_size

        pos_scores = model(h_pos, r, t_pos)

        neg_scores = []
        for i in range(neg_ratio):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            neg_score = model(h_neg[start_idx:end_idx], r, t_neg[start_idx:end_idx])
            neg_scores.append(neg_score)

        neg_scores = torch.cat(neg_scores)

        if model_type == 'ConvE':
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            labels = torch.cat([pos_labels, neg_labels])
            scores = torch.cat([pos_scores, neg_scores])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
        else:
            loss = margin_ranking_loss(pos_scores, neg_scores, margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_triples, all_triples, entities, device, top_k=10, model_type='TransE'):
    model.eval()
    all_triples_set = set(all_triples)
    entity_tensor = torch.tensor(entities, dtype=torch.long, device=device)
    num_entities = len(entities)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr = 0.0
    total = 0

    with torch.no_grad():
        for h, r, t in tqdm(test_triples, desc="Evaluating"):
            total += 1
            batch_size = 512
            h_rank = None

            r_tensor = torch.tensor([r], device=device).repeat(batch_size)
            t_tensor = torch.tensor([t], device=device).repeat(batch_size)

            for i in range(0, num_entities, batch_size):
                batch_entities = entity_tensor[i:i + batch_size]
                batch_size_current = len(batch_entities)
                r_batch = r_tensor[:batch_size_current]
                t_batch = t_tensor[:batch_size_current]

                mask = torch.tensor([
                    (e.item(), r, t) not in all_triples_set or e.item() == h
                    for e in batch_entities
                ], device=device)

                if model_type == 'ConvE':
                    scores = model(batch_entities, r_batch, t_batch)
                else:
                    scores = model(batch_entities, r_batch, t_batch)

                if h in batch_entities:
                    idx = (batch_entities == h).nonzero()[0].item()
                    target_score = scores[idx].item()
                    higher = (scores[mask] > target_score).sum().item()
                    h_rank = higher + 1

            h_tensor = torch.tensor([h], device=device).repeat(batch_size)
            r_tensor = torch.tensor([r], device=device).repeat(batch_size)
            t_rank = None

            for i in range(0, num_entities, batch_size):
                batch_entities = entity_tensor[i:i + batch_size]
                batch_size_current = len(batch_entities)
                h_batch = h_tensor[:batch_size_current]
                r_batch = r_tensor[:batch_size_current]

                mask = torch.tensor([
                    (h, r, e.item()) not in all_triples_set or e.item() == t
                    for e in batch_entities
                ], device=device)

                if model_type == 'ConvE':
                    scores = model(h_batch, r_batch, batch_entities)
                else:
                    scores = model(h_batch, r_batch, batch_entities)

                if t in batch_entities:
                    idx = (batch_entities == t).nonzero()[0].item()
                    target_score = scores[idx].item()
                    higher = (scores[mask] > target_score).sum().item()
                    t_rank = higher + 1

            if h_rank is not None and t_rank is not None:
                avg_rank = (h_rank + t_rank) / 2
                mrr += 1.0 / avg_rank
                if avg_rank <= 1:
                    hits_at_1 += 1
                if avg_rank <= 3:
                    hits_at_3 += 1
                if avg_rank <= 10:
                    hits_at_10 += 1

    return {
        'hits_at_1': hits_at_1 / total,
        'hits_at_3': hits_at_3 / total,
        'hits_at_10': hits_at_10 / total,
        'mrr': mrr / total
    }


def load_data(data_path):
    with open(os.path.join(data_path, 'entities.dict'), 'r') as f:
        entity_dict = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in f}

    with open(os.path.join(data_path, 'relations.dict'), 'r') as f:
        relation_dict = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in f}

    def load_triples(file_path):
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((entity_dict[h], relation_dict[r], entity_dict[t]))
        return triples

    train_triples = load_triples(os.path.join(data_path, 'train.txt'))
    valid_triples = load_triples(os.path.join(data_path, 'valid.txt'))
    test_triples = load_triples(os.path.join(data_path, 'test.txt'))

    return entity_dict, relation_dict, train_triples, valid_triples, test_triples


def plot_training_loss(train_losses, model_name, output_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.title(f'{model_name}模型训练损失曲线', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_name}_loss.png'), dpi=300)
    plt.close()


def plot_validation_metrics(metrics_history, model_name, output_path):
    """绘制验证集指标变化曲线"""
    epochs = [5 * i for i in range(1, len(metrics_history) + 1)]  # 每5个epoch验证一次

    plt.figure(figsize=(12, 8))

    # 绘制MRR曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['mrr'] for m in metrics_history], marker='s', color='r')
    plt.title('MRR变化曲线', fontsize=12)
    plt.xlabel('训练轮次', fontsize=10)
    plt.ylabel('MRR值', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制Hits@1曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['hits_at_1'] for m in metrics_history], marker='^', color='g')
    plt.title('Hits@1变化曲线', fontsize=12)
    plt.xlabel('训练轮次', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制Hits@3曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['hits_at_3'] for m in metrics_history], marker='d', color='purple')
    plt.title('Hits@3变化曲线', fontsize=12)
    plt.xlabel('训练轮次', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制Hits@10曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['hits_at_10'] for m in metrics_history], marker='o', color='orange')
    plt.title('Hits@10变化曲线', fontsize=12)
    plt.xlabel('训练轮次', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_name}_validation_metrics.png'), dpi=300)
    plt.close()


def plot_test_metrics(test_metrics, model_name, output_path):
    """绘制测试集指标柱状图"""
    metrics = ['hits_at_1', 'hits_at_3', 'hits_at_10', 'mrr']
    values = [test_metrics[m] for m in metrics]
    labels = ['Hits@1', 'Hits@3', 'Hits@10', 'MRR']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.title(f'{model_name}模型测试集性能指标', fontsize=14)
    plt.ylabel('指标值', fontsize=12)
    plt.ylim(0, max(values) + 0.1)  # 设置y轴范围
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # MRR和Hits指标使用不同的y轴格式
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0) if max(values) <= 1 else plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_name}_test_metrics.png'), dpi=300)
    plt.close()


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录（包括可视化结果目录）
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    viz_path = os.path.join(args.output_path, 'visualizations')
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    # 加载数据
    print("加载数据中...")
    entity_dict, relation_dict, train_triples, valid_triples, test_triples = load_data(args.data_path)

    entities = list(entity_dict.values())
    relations = list(relation_dict.values())

    print(f"实体数量: {len(entities)}, 关系数量: {len(relations)}")
    print(f"训练集三元组: {len(train_triples)}, 验证集三元组: {len(valid_triples)}, 测试集三元组: {len(test_triples)}")

    # 创建数据集和数据加载器
    train_dataset = KGDataset(train_triples, entities, relations, neg_ratio=args.neg_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 初始化模型
    print(f"初始化 {args.model} 模型...")
    if args.model == 'TransE':
        model = TransE(
            num_entities=len(entities),
            num_relations=len(relations),
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm
        ).to(device)
    elif args.model == 'RotatE':
        model = RotatE(
            num_entities=len(entities),
            num_relations=len(relations),
            embedding_dim=args.embedding_dim,
            margin=args.margin
        ).to(device)
    elif args.model == 'ConvE':
        model = ConvE(
            num_entities=len(entities),
            num_relations=len(relations),
            embedding_dim=args.embedding_dim,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size
        ).to(device)
    else:
        raise ValueError(f"不支持的模型: {args.model}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练和验证
    best_mrr = 0.0
    best_model = model.state_dict().copy()  # 初始化最佳模型为初始状态
    train_losses = []  # 记录训练损失
    metrics_history = []  # 记录验证集指标

    print("开始训练...")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, device, args.margin, args.model)
        train_losses.append(train_loss)
        print(f"轮次 {epoch + 1}/{args.epochs}, 训练损失: {train_loss:.4f}")

        # 每5个epoch验证一次
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, valid_triples, train_triples + valid_triples, entities, device,
                               model_type=args.model)
            metrics_history.append(metrics)
            print(
                f"验证集结果: Hits@1={metrics['hits_at_1']:.4f}, Hits@3={metrics['hits_at_3']:.4f}, "
                f"Hits@10={metrics['hits_at_10']:.4f}, MRR={metrics['mrr']:.4f}")

            # 保存最佳模型
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                best_model = model.state_dict().copy()
                print(f"在轮次 {epoch + 1} 保存最佳模型，MRR: {best_mrr:.4f}")

    # 绘制训练损失曲线
    plot_training_loss(train_losses, args.model, viz_path)

    # 如果有验证集指标，绘制验证集指标曲线
    if metrics_history:
        plot_validation_metrics(metrics_history, args.model, viz_path)

    # 加载最佳模型并在测试集上评估
    print("在测试集上评估...")
    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_triples, train_triples + valid_triples + test_triples, entities, device,
                            model_type=args.model)
    print(
        f"测试集结果: Hits@1={test_metrics['hits_at_1']:.4f}, Hits@3={test_metrics['hits_at_3']:.4f}, "
        f"Hits@10={test_metrics['hits_at_10']:.4f}, MRR={test_metrics['mrr']:.4f}")

    # 绘制测试集指标图
    plot_test_metrics(test_metrics, args.model, viz_path)
    print(f"可视化结果已保存至: {viz_path}")

    # 保存模型
    if args.save_model:
        model_path = os.path.join(args.output_path, f"{args.model}_{args.embedding_dim}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'entity_dict': entity_dict,
            'relation_dict': relation_dict,
            'args': args,
            'train_losses': train_losses,
            'validation_metrics': metrics_history,
            'test_metrics': test_metrics
        }, model_path)
        print(f"模型已保存至: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='知识图谱嵌入模型')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/FB15k-237', help='数据集路径')
    parser.add_argument('--output_path', type=str, default='models', help='模型保存路径')

    # 模型参数
    parser.add_argument('--model', type=str, default='TransE', choices=['TransE', 'RotatE', 'ConvE'], help='模型类型')
    parser.add_argument('--embedding_dim', type=int, default=200, help='嵌入维度')
    parser.add_argument('--margin', type=float, default=1.0, help='排序损失的margin值')
    parser.add_argument('--p_norm', type=int, default=1, help='TransE的p-norm值')
    parser.add_argument('--hidden_channels', type=int, default=32, help='ConvE的隐藏通道数')
    parser.add_argument('--kernel_size', type=int, default=3, help='ConvE的卷积核大小')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--neg_ratio', type=int, default=1, help='负样本比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_model', action='store_true', help='保存模型')

    args = parser.parse_args()
    main(args)