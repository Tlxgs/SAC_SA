from config import config
config.apply_env()

import os
import torch
import copy
import numpy as np
import argparse
import random
from models import ClusterHead
from eval_utils import cluster_metric
from torch.utils.data import DataLoader, TensorDataset
from loss_utils import entropy, consistency_loss, num_consistency_loss, DataContrastiveLoss, StructureAlignmentLoss
from data_utils import NeighborsDataset, mine_nearest_neighbors, TestDataset



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'




def infer(model, dataloader, fusion_weight=0.5):
    """推理阶段融合文本和图像模态的预测"""
    model.eval()
    preds, logits_combined = [], []
    epsilon = 1e-12

    with torch.no_grad():
        for _, (text, image) in enumerate(dataloader):
            text = text[0].cuda()
            image = image[0].cuda()
            
            logit_text, logit_image, _ = model(text, image)

            prob_combined = torch.pow(logit_text + epsilon, 1 - fusion_weight) * \
                           torch.pow(logit_image + epsilon, fusion_weight)
            prob_combined = prob_combined / (prob_combined.sum(dim=1, keepdim=True) + epsilon)

            preds.append(torch.argmax(prob_combined, dim=1).cpu().numpy())
            logits_combined.append(prob_combined.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(logits_combined, axis=0)


def average_weights(state_dicts):
    """对多个模型权重取平均"""
    avg_state_dict = {}
    keys = state_dicts[0].keys()
    
    for key in keys:
        param = state_dicts[0][key]
        if param.dtype in [torch.float32, torch.float64, torch.float16]:
            avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
        else:
            avg_state_dict[key] = state_dicts[-1][key]
    
    return avg_state_dict


if __name__ == "__main__":
    print("running...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config.dataset.dataset)
    parser.add_argument('--dataset_name', type=str, default=config.dataset.dataset_name)
    parser.add_argument('--consist_coeff', type=float, default=config.hyper.consist_coeff)
    parser.add_argument('--entropy_coeff', type=float, default=config.hyper.entropy_coeff)
    parser.add_argument('--dc_coeff', type=float, default=config.hyper.dc_coeff)
    parser.add_argument('--neighbor_numbers', type=int, default=config.hyper.neighbor_numbers)
    parser.add_argument('--topk', type=int, default=config.hyper.topk_neighbors)
    parser.add_argument('--alpha', type=float, default=config.hyper.alpha)
    parser.add_argument('--beta', type=float, default=config.hyper.beta)
    parser.add_argument('--ratio', type=float, default=config.hyper.ratio)
    parser.add_argument('--fusion_weight', type=float, default=config.hyper.fusion_weight)
    parser.add_argument('--sa_coeff', type=float, default=config.hyper.sa_coeff)
    parser.add_argument('--reliability_temp', type=float, default=config.hyper.reliability_temp)
    parser.add_argument('--seed', type=int, default=config.hyper.seed)
    parser.add_argument('--epochs', type=int, default=config.hyper.epochs)
    parser.add_argument('--batch_size', type=int, default=config.dataset.train_bs)
    parser.add_argument('--lr', type=float, default=config.hyper.lr)
    parser.add_argument('--betas',type=float,default=config.hyper.betas)
    args = parser.parse_args()

    if not os.path.exists(config.description_test_file):
        print(f"错误: 测试集文本嵌入文件不存在")
        exit(1)

    dataset = args.dataset
    dataset_name = args.dataset_name
    consist_coeff = args.consist_coeff
    entropy_coeff = args.entropy_coeff
    dc_coeff = args.dc_coeff
    epochs = args.epochs
    batch_size = args.batch_size
    topK = args.topk
    beta = args.beta
    ratio = args.ratio
    fusion_weight = args.fusion_weight
    sa_coeff = args.sa_coeff
    reliability_temp = args.reliability_temp
    set_seed(args.seed)
    # 确定聚类数
    cluster_num_map = {"CIFAR-10": 10, "STL-10": 10, "ImageNet-10": 10, "CIFAR-20": 20, "ImageNet-Dogs": 15}
    cluster_num = cluster_num_map.get(dataset)
    if cluster_num is None:
        raise NotImplementedError

    # 加载文本嵌入
    with open(config.description_train_file, "r") as f:
        valid_lines = [line.strip() for line in f if line.strip()]
    nouns_embedding = np.load(config.embedding_train_save_file)
    print(f"训练集文本描述: {len(valid_lines)} 个")

    with open(config.description_test_file, "r") as f:
        test_valid_lines = [line.strip() for line in f if line.strip()]
    nouns_embedding_test = np.load(config.embedding_test_save_file)
    print(f"测试集文本描述: {len(test_valid_lines)} 个")

    if config.ablate.normalize_text_embeddings:
        nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)
        nouns_embedding_test = nouns_embedding_test / np.linalg.norm(nouns_embedding_test, axis=1, keepdims=True)

    # 加载图像嵌入
    data_dir = config.dataset.data_dir
    images_embedding_train = np.load(f"{data_dir}/{dataset}_image_embedding_train.npy")
    images_embedding_test = np.load(f"{data_dir}/{dataset}_image_embedding_test.npy")
    if config.ablate.normalize_image_embeddings:
        images_embedding_train = images_embedding_train / np.linalg.norm(images_embedding_train, axis=1, keepdims=True)
        images_embedding_test = images_embedding_test / np.linalg.norm(images_embedding_test, axis=1, keepdims=True)

    labels_test = np.loadtxt(f"{data_dir}/{dataset}_labels_test.txt")

    # 构建数据集
    model = ClusterHead(in_dim=512, text_in_dim=768, num_clusters=cluster_num).cuda()

    dataset_text_train = TensorDataset(torch.from_numpy(nouns_embedding).float())
    dataset_text_test = TensorDataset(torch.from_numpy(nouns_embedding_test).float())
    dataset_image_train = TensorDataset(torch.from_numpy(images_embedding_train).float())
    dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())

    indices_text = mine_nearest_neighbors(nouns_embedding, topk=topK)
    indices_image = mine_nearest_neighbors(images_embedding_train, topk=topK)

    train_dataset = NeighborsDataset(dataset_text_train, dataset_image_train, indices_text, indices_image, k=args.neighbor_numbers)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = TestDataset(dataset_text_test, dataset_image_test)
    dataloader_test = DataLoader(test_dataset, batch_size=config.dataset.test_bs, shuffle=False, drop_last=False)

    # 初始化损失函数
    DC_loss = DataContrastiveLoss(alpha=args.alpha)
    sa_loss_fn = StructureAlignmentLoss(temperature=reliability_temp,use_reliability_weight=config.ablate.use_reliability_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)

    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"总Epochs: {epochs}, 结构对齐损失系数: {sa_coeff}")

    print("\nStart infer...")
    preds, _ = infer(model, dataloader_test, fusion_weight=fusion_weight)
    cluster_metric(labels_test, preds)

    # 记录最后三个epoch的模型权重
    last_three_states = []

    print("\nStart training...")
    ifweight = True
    epsilon = 1e-12

    for epoch in range(epochs):
        model.train()
        loss_consist_epoch = 0.0
        loss_entropy_epoch = 0.0
        loss_dc_epoch = 0.0
        loss_sa_epoch = 0.0
        num_batches = 0
        
        reliability_info_list = []

        for iter, (indices, text, image, neigh_text, neigh_image, image2text, text2image) in enumerate(dataloader_train):
            text = text[0].cuda()
            image = image[0].cuda()
            neigh_text = neigh_text[0].cuda()
            neigh_image = neigh_image[0].cuda()
            image2text = image2text[0].cuda()
            text2image = text2image[0].cuda()

            # 主模型预测
            logit_text, logit_image, text_proj = model(text, image)
            neigh_logit_text, neigh_logit_image, neigh_text_proj = model(neigh_text, neigh_image)
            logit_image2text, logit_text2image, _ = model(image2text, text2image)

            # 使用固定融合权重
            logit_fused = torch.pow(logit_text + epsilon, 1 - fusion_weight) * \
                         torch.pow(logit_image + epsilon, fusion_weight)
            logit_fused = logit_fused / (logit_fused.sum(dim=1, keepdim=True) + epsilon)

            neigh_logit_fused = torch.pow(neigh_logit_text + epsilon, 1 - fusion_weight) * \
                               torch.pow(neigh_logit_image + epsilon, fusion_weight)
            neigh_logit_fused = neigh_logit_fused / (neigh_logit_fused.sum(dim=1, keepdim=True) + epsilon)

            logit_fused_cross = torch.pow(logit_text2image + epsilon, 1 - fusion_weight) * \
                               torch.pow(logit_image2text + epsilon, fusion_weight)
            logit_fused_cross = logit_fused_cross / (logit_fused_cross.sum(dim=1, keepdim=True) + epsilon)

            loss = 0.0

            # 一致性损失
            loss_consist = torch.tensor(0.0, device=logit_text.device)
            if config.ablate.use_consistency_loss:
                loss_consist = beta * consistency_loss(logit_text, logit_image) + \
                              (1 - beta) * num_consistency_loss(logit_text, logit_image)
                loss += consist_coeff * loss_consist
                loss_consist_epoch += loss_consist.item()

            # 熵损失（类别平衡）
            loss_entropy = torch.tensor(0.0, device=logit_text.device)
            if config.ablate.use_entropy_loss:
                loss_entropy = entropy(logit_text) + entropy(logit_image)
                loss += entropy_coeff * loss_entropy
                loss_entropy_epoch += loss_entropy.item()

            # 对比损失

            loss_dc = torch.tensor(0.0, device=logit_text.device)
            if config.ablate.use_data_contrastive_loss:
                loss_dc = DC_loss(logit_fused, logit_fused_cross, logit_text, logit_image2text,
                                  logit_image, logit_text2image, ifweight, ratio)
                loss_dc += DC_loss(logit_fused, neigh_logit_fused, logit_text, neigh_logit_image,
                                   logit_image, neigh_logit_text, ifweight, ratio)
                loss += dc_coeff * loss_dc
                loss_dc_epoch += loss_dc.item()

            # 可靠性对比损失
            loss_sa = torch.tensor(0.0, device=logit_text.device)
            if config.ablate.use_sa_loss:
                loss_sa, reliability_info = sa_loss_fn(
                    logit_text, logit_image, text_proj, image
                )
                loss += sa_coeff * loss_sa
                loss_sa_epoch += loss_sa.item()
                reliability_info_list.append(reliability_info)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_batches = iter + 1

        # Epoch结束评估
        preds, _ = infer(model, dataloader_test, fusion_weight=fusion_weight)
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        acc = cluster_metric(labels_test, preds)
        
        # 打印损失信息
        print(f"  Loss - DC: {loss_dc_epoch/num_batches:.4f} "
              f"Consist: {loss_consist_epoch/num_batches:.4f} "
              f"Ent: {loss_entropy_epoch/num_batches:.4f} "
              f"SA: {loss_sa_epoch/num_batches:.4f}")
        
        # 打印可靠性统计
        if reliability_info_list:
            avg_rel_text = np.mean([info['reliability_text'] for info in reliability_info_list])
            avg_rel_image = np.mean([info['reliability_image'] for info in reliability_info_list])
            avg_entropy_text = np.mean([info['entropy_text'] for info in reliability_info_list])
            avg_entropy_image = np.mean([info['entropy_image'] for info in reliability_info_list])
            avg_cross_sim = np.mean([info['cross_sim'] for info in reliability_info_list])
            avg_agreement = np.mean([info['pred_agreement'] for info in reliability_info_list])
            print(f"  可靠性 - 文本: {avg_rel_text:.3f}, 图像: {avg_rel_image:.3f}")
            print(f"  熵 - 文本: {avg_entropy_text:.3f}, 图像: {avg_entropy_image:.3f}")
            print(f"  跨模态相似度: {avg_cross_sim:.3f}, 预测一致率: {avg_agreement:.3f}")

        # 记录最后三个epoch的模型权重
        if epoch >= epochs - 3:
            last_three_states.append(copy.deepcopy(model.state_dict()))

    # 对最后三个epoch的模型权重取平均
    print("\n" + "=" * 50)
    print("模型权重平均（最后三个epoch）")
    print("=" * 50)
    
    avg_state_dict = average_weights(last_three_states)
    model.load_state_dict(avg_state_dict)
    
    # 用平均模型评估
    print("\n平均模型评估结果:")
    preds, _ = infer(model, dataloader_test, fusion_weight=fusion_weight)
    cluster_metric(labels_test, preds)

    # 保存模型
    torch.save(avg_state_dict, f'best_model_{dataset_name}.pth')
    # ========== 打印超参数和消融设置 ==========
    print("\n" + "=" * 50)
    print("实验配置参数:")
    print("=" * 50)

    # 打印超参数
    print("\n[Hyperparameters]")
    for param_name in dir(args.hyper):
        if not param_name.startswith('_'):
            print(f"  {param_name}: {getattr(args.hyper, param_name)}")

    # 打印消融设置
    print("\n[Ablations]")
    for param_name in dir(config.ablate):
        if not param_name.startswith('_'):
            print(f"  {param_name}: {getattr(config.ablate, param_name)}")

    print("=" * 50)