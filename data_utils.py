import os
import faiss
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder

from config import config

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(dataset="CIFAR-10"):
    if dataset in ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "UCF101"]:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.pp.image_size_train, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(config.pp.image_size_train),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
    elif dataset in ["ImageNet-Dogs", "ImageNet-10", "ImageNet", "tinyimagenet"]:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.pp.resize_imagenet_like, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(config.pp.image_size_imagenet_like),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
    else:
        raise NotImplementedError
    return transforms


def get_dataloader(dataset=None, batch_size=None):
    dataset = dataset or config.dataset.dataset
    batch_size = batch_size or config.dataset.embed_infer_bs
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(root="./data", train=True, download=True, transform=transforms)
        data_test = CIFAR10(root="./data", train=False, download=True, transform=transforms)
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(root="./data", train=True, download=False, transform=transforms)
        data_test = CIFAR100(root="./data", train=False, download=False, transform=transforms)
    elif dataset == "STL-10":
        data_train = STL10(root="./data", split="train", download=False, transform=transforms)
        data_test = STL10(root="./data", split="test", download=False, transform=transforms)
    elif dataset == "ImageNet-10":
        data_train = ImageFolder("./data/ImageNet-10/train", transform=transforms)
        data_test = ImageFolder("./data/ImageNet-10/val", transform=transforms)
    elif dataset == "ImageNet-Dogs":
        data_train = ImageFolder("./data/ImageNet-Dogs/train", transform=transforms)
        data_test = ImageFolder("./data/ImageNet-Dogs/val", transform=transforms)
    else:
        raise NotImplementedError

    dl_train = DataLoader(data_train, batch_size=batch_size, shuffle=False, drop_last=False)
    dl_test = DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl_train, dl_test


def mine_nearest_neighbors(features, topk=50, index_file=None):
    """使用GPU加速计算最近邻"""
    print("Computing nearest neighbors with GPU acceleration...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]

    if torch.cuda.is_available():
        try:
            print(f"使用 PyTorch GPU 计算最近邻，数据维度: {n} x {dim}")
            features_tensor = torch.from_numpy(features).cuda()
            features_tensor = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
            similarity_matrix = torch.mm(features_tensor, features_tensor.t())
            _, indices_tensor = torch.topk(similarity_matrix, k=topk + 1, dim=1, largest=True)
            indices = indices_tensor.cpu().numpy()
            print("PyTorch GPU 最近邻计算完成。")
            return indices[:, 1:]  # 排除自身
        except Exception as e:
            print(f"PyTorch GPU 计算失败，回退到 CPU FAISS: {e}")

    # 回退到 CPU FAISS
    print("使用 CPU FAISS...")
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(dim)
    index.add(features)
    distances, indices = index.search(features, topk + 1)
    print("CPU FAISS 最近邻计算完成。")
    return indices[:, 1:]


class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image, k):
        super().__init__()
        self.num_neighbors = k
        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        # 修复：正确检查索引数量与数据集大小是否匹配
        assert self.indices_text.shape[0] == len(self.dataset_text), \
            f"文本索引数量 {self.indices_text.shape[0]} 与文本数据集大小 {len(self.dataset_text)} 不匹配"
        assert self.indices_image.shape[0] == len(self.dataset_image), \
            f"图像索引数量 {self.indices_image.shape[0]} 与图像数据集大小 {len(self.dataset_image)} 不匹配"

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_index_image2text = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_image2text = self.dataset_text.__getitem__(neighbor_index_image2text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_index_text2image = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text2image = self.dataset_image.__getitem__(neighbor_index_text2image)
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)
        return index, anchor_text, anchor_image, neighbor_text, neighbor_image, neighbor_image2text, neighbor_text2image


class TestDataset(Dataset):
    def __init__(self, dataset_text, dataset_image):
        super().__init__()
        self.dataset_text = dataset_text
        self.dataset_image = dataset_image

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        return self.dataset_text.__getitem__(index), self.dataset_image.__getitem__(index)
