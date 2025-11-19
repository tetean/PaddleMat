import os
import yaml
import argparse
import numpy as np
import paddle
from paddle.io import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import time
from utils.loss import evidential_loss, mse_loss
from utils.model_info import print_model_info
from alignn_paddle_model import ALIGNN
from alignn_data_utils import load_dataset, ALIGNNDataset, collate_fn


def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)

    return total_params, trainable_params


def train_one_epoch(model, train_loader, optimizer, config, epoch):
    """Train for one epoch, handling batched graph data."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (graph_data, targets, cif_ids) in enumerate(pbar):
        optimizer.clear_grad()

        # 1. 解包 collate_fn 返回的 BatchGraph 和特征
        g = graph_data['g']
        lg = graph_data['lg']
        atom_features = graph_data['atom_features']
        edge_r = graph_data['edge_r']
        angle_features = graph_data['angle_features']

        # 构造模型的输入列表
        data_list = [g, lg, atom_features, edge_r, angle_features]

        # Forward pass
        if config.get('evidential') == "True":
            mu, v, alpha, beta = model(data_list)
            loss = evidential_loss(mu, v, alpha, beta, targets, coeff=config.get('coeff', 0.01))
        else:
            pred = model(data_list)
            loss = mse_loss(pred, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, val_loader, config):
    """Evaluate model, handling batched graph data."""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    ground_truths = []

    with paddle.no_grad():
        for graph_data, targets, cif_ids in tqdm(val_loader, desc="Evaluating", leave=False):

            # 1. 解包 collate_fn 返回的 BatchGraph 和特征
            g = graph_data['g']
            lg = graph_data['lg']
            atom_features = graph_data['atom_features']
            edge_r = graph_data['edge_r']
            angle_features = graph_data['angle_features']

            # 构造模型的输入列表
            data_list = [g, lg, atom_features, edge_r, angle_features]

            # Forward pass
            if config.get('evidential') == "True":

                start_time = time.time()
                mu, v, alpha, beta = model(data_list)
                end_time = time.time()
                print(f"[Dev] Evaluation forward pass time: {end_time - start_time:.4f} seconds")
                loss = evidential_loss(mu, v, alpha, beta, targets, coeff=config.get('coeff', 0.01))
                pred = mu
            else:
                pred = model(data_list)
                loss = mse_loss(pred, targets)

            total_loss += loss.item()
            num_batches += 1

            predictions.extend(pred.numpy().flatten().tolist())
            ground_truths.extend(targets.numpy().flatten().tolist())

    avg_loss = total_loss / num_batches

    # 计算MAE
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    mae = np.mean(np.abs(predictions - ground_truths))

    return avg_loss, mae


def main(args):
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device("gpu:0")
        print("Using GPU for training")
    else:
        print("Using CPU for training")

    # 加载配置
    with open(args.config, 'r') as f:
        all_configs = yaml.safe_load(f)

    model_name = args.model
    if model_name not in all_configs['Models']:
        raise ValueError(f"Model {model_name} not found in config")

    config = all_configs['Models'][model_name]
    model_setting = config.get('model_setting', {})

    print(f"Training {model_name} with config:")
    print(json.dumps(config, indent=2))

    # 加载数据
    print(f"\nLoading data from {args.data_dir}/{args.task}...")
    full_dataset = load_dataset(args.data_dir, args.task, config)

    # 划分训练集、验证集、测试集
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1 - train_ratio - val_ratio

    indices = list(range(len(full_dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=args.seed
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_ratio / (val_ratio + test_ratio), random_state=args.seed
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    # 创建数据集
    train_data = [full_dataset[i] for i in train_indices]
    val_data = [full_dataset[i] for i in val_indices]
    test_data = [full_dataset[i] for i in test_indices]

    train_dataset = ALIGNNDataset(train_data)
    val_dataset = ALIGNNDataset(val_data)
    test_dataset = ALIGNNDataset(test_data)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 创建模型
    print("\nInitializing model...")
    model = ALIGNN(
        atom_input_features=model_setting.get('atom_input_features', 92),
        edge_input_features=model_setting.get('edge_input_features', 80),
        triplet_input_features=model_setting.get('triplet_input_features', 40),
        hidden_features=model_setting.get('hidden_features', 256),
        embedding_features=model_setting.get('embedding_features', 64),
        alignn_layers=model_setting.get('alignn_layers', 4),
        gcn_layers=model_setting.get('gcn_layers', 4),
        evidential=config.get('evidential', "False"),
        mc_dropout=config.get('mc_dropout', 0.1),
    )

    print_model_info(model)

    # 创建优化器
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.00001)

    if config.get('optimizer', 'AdamW') == 'AdamW':
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=model.parameters(),
            weight_decay=weight_decay
        )
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            parameters=model.parameters(),
            weight_decay=weight_decay
        )

    # 训练循环
    epochs = config.get('epochs', 300)
    best_val_mae = float('inf')
    best_epoch = 0

    # 用于记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, config, epoch)
        history['train_loss'].append(train_loss)

        # 每个epoch都进行验证
        val_loss, val_mae = evaluate(model, val_loader, config)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # 打印结果
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE:    {val_mae:.4f}")
        print(f"[dev]  Total time: {time.time() - start_time:.2f} sec")

        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, f'{model_name}_best.pdparams')
            os.makedirs(args.save_dir, exist_ok=True)
            paddle.save(model.state_dict(), save_path)
            print(f"  ✓ New best model! Saved to {save_path}")
        else:
            print(f"  (Best Val MAE: {best_val_mae:.4f} at epoch {best_epoch})")

        print("-" * 80)

        # 可选：early stopping
        if args.early_stop > 0:
            if epoch - best_epoch >= args.early_stop:
                print(f"\nEarly stopping triggered after {args.early_stop} epochs without improvement.")
                print(f"Best epoch was {best_epoch} with Val MAE: {best_val_mae:.4f}")
                break

    end_time = time.time()

    elapsed = end_time - start_time

    print(f"\nTraining completed in {elapsed / 60:.2f} minutes.")

    # 保存训练历史
    history_path = os.path.join(args.save_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # 测试
    print("\n" + "=" * 80)
    print("Testing on test set...")
    model.set_state_dict(paddle.load(os.path.join(args.save_dir, f'{model_name}_best.pdparams')))
    test_loss, test_mae = evaluate(model, test_loader, config)
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  (Best model from epoch {best_epoch})")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ALIGNN model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='/data1/tanliqin/uq-ood-mat/data',
                        help='Root directory containing the data')
    parser.add_argument('--task', type=str, required=True,
                        help='Task name (subdirectory in data_dir)')
    parser.add_argument('--model', type=str, default='ALIGNN',
                        help='Model name from config')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='Early stopping patience (0 to disable)')

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.task, time.strftime("%Y%m%d-%H%M%S"))

    # 设置随机种子
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    main(args)