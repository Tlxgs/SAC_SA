from config import config
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class ClusterHead(nn.Module):
    """双模态聚类头：分别处理文本和图像"""

    def __init__(self, in_dim=512, text_in_dim=768, num_clusters=10, device="cuda"):
        super().__init__()
        self.num_clusters = num_clusters
        self.text_proj = nn.Linear(text_in_dim, in_dim)

        # 文本聚类头: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> Softmax
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )

        # 图像聚类头: 相同结构
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )

        # 正确初始化所有Linear层的权重
        for head in [self.cluster_head_text, self.cluster_head_image]:
            trunc_normal_(head[0].weight, std=0.02)
            trunc_normal_(head[3].weight, std=0.02)
            trunc_normal_(head[6].weight, std=0.02)

    def forward(self, text, image):
        # 投影文本特征到统一维度
        text_proj = self.text_proj(text) if text.shape[1] != 512 else text
        logit_text = self.cluster_head_text(text_proj)
        logit_image = self.cluster_head_image(image)
        return logit_text, logit_image, text_proj


class CLIPModel(nn.Module):
    def __init__(self, model_name=config.CLIPModel):
        super().__init__()
        model_path = config.CLIPModelPath
        self.model = torch.jit.load(model_path, map_location="cpu").eval()

        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(config.pp.image_size_train),
            transforms.CenterCrop(config.pp.image_size_train),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, texts):
        return self.model.encode_text(texts)

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        return image_features, text_features
