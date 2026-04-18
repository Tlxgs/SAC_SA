# -*- coding: utf-8 -*-
"""
"""
import os


class _Paths:
    HF_ENDPOINT: str = "https://hf-mirror.com"
    TRANSFORMERS_OFFLINE: str = "1"
    CLIPModelPath: str = "../models/clip/ViT-B-32.pt"
    BLIP_2_Path: str = "../models/transformers/models--Salesforce--blip2-opt-2.7b/snapshots/59a1ef6c1e5117b3f65523d1c6066825bcf315e3"
    SentenceTransformerPath: str = "../models/sentence_transformers/all-mpnet-base-v2"


class _Dataset:
    dataset: str = "CIFAR-20"
    dataset_name: str = "cifar20"
    embed_infer_bs: int = 1024
    train_bs: int = 512
    test_bs: int = 512
    data_dir: str = "./data"
    use_cifar20_coarse_map: bool = True
    cifar20_coarse_map = [
        [72, 4, 95, 30, 55], [73, 32, 67, 91, 1], [92, 70, 82, 54, 62], [16, 61, 9, 10, 28], [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86], [20, 25, 94, 84, 5], [14, 24, 6, 7, 18], [43, 97, 42, 3, 88], [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60], [15, 21, 19, 31, 38], [75, 63, 66, 64, 34], [77, 26, 45, 99, 79], [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44], [65, 50, 74, 36, 80], [56, 52, 47, 59, 96], [8, 58, 90, 13, 48], [81, 69, 41, 89, 85],
    ]


class _Hyper:
    seed: int = 0
    epochs: int = 20

    topk_neighbors: int = 50
    neighbor_numbers: int = 500

    # 损失权重
    consist_coeff: float = 0.6
    entropy_coeff: float = -3
    alpha: float = 1.0
    ratio: float = 0.5
    beta: float = 0.5
    fusion_weight: float = 0.5

    dc_coeff: float = 1.0
    sa_coeff: float = .0   # 结构对齐损失权重
    reliability_temp: float = 0.0      # 可靠性计算温度

    # Optimizer
    lr: float = 1e-4
    betas = (0.9, 0.99)


class _Ablations:
    use_consistency_loss: bool = True
    use_entropy_loss: bool = True
    use_data_contrastive_loss: bool = True
    use_sa_loss: bool = True
    use_reliability_weight: bool = True
    normalize_image_embeddings: bool = True
    normalize_text_embeddings: bool = True


class _Preprocess:
    image_size_train: int = 224
    image_size_imagenet_like: int = 224
    resize_imagenet_like: int = 256


class _TextGen:
    @staticmethod
    def train_desc_path(dataset_name: str):
        return f"./data/{dataset_name}_train_descriptions_blip2.txt"

    @staticmethod
    def test_desc_path(dataset_name: str):
        return f"./data/{dataset_name}_test_descriptions_blip2.txt"

    @staticmethod
    def train_embed_path(dataset_name: str):
        return f"./data/{dataset_name}_sberttext_embedding_train_blip2.npy"

    @staticmethod
    def test_embed_path(dataset_name: str):
        return f"./data/{dataset_name}_sberttext_embedding_test_blip2.npy"


class Config:
    def __init__(self):
        self.paths = _Paths()
        self.dataset = _Dataset()
        self.hyper = _Hyper()
        self.ablate = _Ablations()
        self.pp = _Preprocess()
        self.text = _TextGen()

        self.CLIPModel = "ViT-B/32"
        self.CLIPModelPath = self.paths.CLIPModelPath
        self.BLIP_2_Path = self.paths.BLIP_2_Path
        self.SentenceTransformerPath = self.paths.SentenceTransformerPath
        self.dataset_name = self.dataset.dataset_name
        self.dataset_choice = self.dataset.dataset

        self.description_train_file = self.text.train_desc_path(self.dataset_name)
        self.embedding_train_save_file = self.text.train_embed_path(self.dataset_name)
        self.description_test_file = self.text.test_desc_path(self.dataset_name)
        self.embedding_test_save_file = self.text.test_embed_path(self.dataset_name)

    def apply_env(self):
        os.environ["HF_ENDPOINT"] = self.paths.HF_ENDPOINT
        os.environ["TRANSFORMERS_OFFLINE"] = self.paths.TRANSFORMERS_OFFLINE
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'


config = Config()
