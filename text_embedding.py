from config import config
config.apply_env()

import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms, datasets


def _get_dataset(dataset_choice: str, train: bool, transform):
    if dataset_choice == "CIFAR-10":
        return datasets.CIFAR10(root='./data', train=train, download=False, transform=transform)
    elif dataset_choice == "CIFAR-20":
        # CIFAR-20 uses CIFAR-100 images
        return datasets.CIFAR100(root='./data', train=train, download=False, transform=transform)
    elif dataset_choice == "STL-10":
        split = "train" if train else "test"
        return datasets.STL10(root='./data', split=split, download=True, transform=transform)
    else:
        raise NotImplementedError("不支持此数据集")


def generate_descriptions(dataset_type="train"):
    """
    生成训练集或测试集的文本描述（使用本地 BLIP-2）。
    """
    transform = transforms.Compose([transforms.ToTensor()])
    ds_choice = config.dataset.dataset
    if dataset_type == "train":
        dataset_obj = _get_dataset(ds_choice, True, transform)
        descriptions_file = config.description_train_file
        dataset_name = "训练集"
    else:
        dataset_obj = _get_dataset(ds_choice, False, transform)
        descriptions_file = config.description_test_file
        dataset_name = "测试集"

    model_path = config.BLIP_2_Path
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    descriptions = []
    if os.path.exists(descriptions_file):
        with open(descriptions_file, "r") as f:
            descriptions = [line.strip() for line in f if line.strip()]
        print(f"加载了 {len(descriptions)} 个已有{dataset_name}描述")
        start_idx = len(descriptions)
    else:
        start_idx = 0

    print(f"开始生成{dataset_name}描述，从第 {start_idx} 个开始，总共 {len(dataset_obj)} 个样本")

    for i in tqdm(range(start_idx, len(dataset_obj)), desc=f"生成{dataset_name}描述"):
        image, _ = dataset_obj[i]
        image = transforms.ToPILImage()(image)
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        answer = description.split("Answer:")[-1].strip() if "Answer:" in description else description
        if answer.strip() == "":
            print(f"Warning: {dataset_name}图像 {i} 生成了空描述")
        descriptions.append(answer)
        if (i + 1) % 100 == 0:
            with open(descriptions_file, "w") as f:
                for desc in descriptions:
                    f.write(desc + "\n")
            print(f"已保存前 {i + 1} 个{dataset_name}描述")

    print(f"总共生成了 {len(descriptions)} 个{dataset_name}描述")
    with open(descriptions_file, "w") as f:
        for desc in descriptions:
            f.write(desc + "\n")
    print(f"{dataset_name}描述已保存至: " + descriptions_file)
    return descriptions_file


def generate_embeddings(descriptions_file, dataset_type="train"):
    class TextDataProcessing(Dataset):
        def __init__(self, descriptions_file):
            self.text = []
            with open(descriptions_file, encoding="utf-8") as f:
                self.text = [line.strip() for line in f if line.strip()]
            print(f"文本长度: {len(self.text)}")

        def __len__(self):
            return len(self.text)

        def __getitem__(self, idx):
            return self.text[idx]

    def build_semantic(dataset, dataset_type):
        local_model_path = config.SentenceTransformerPath
        dataset_name = "训练集" if dataset_type == "train" else "测试集"
        print(f"使用本地 SentenceTransformer 模型生成{dataset_name}嵌入...")
        sbert = SentenceTransformer(local_model_path)
        embeddings = []
        for texts in tqdm(DataLoader(dataset, batch_size=512), desc=f"{dataset_name}文本嵌入..."):
            emb = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings.append(emb.cpu().numpy())
        embeddings = np.concatenate(embeddings)
        if dataset_type == "train":
            embedding_file = config.embedding_train_save_file
        else:
            embedding_file = config.embedding_test_save_file
        np.save(embedding_file, embeddings)
        print(f"{dataset_name}嵌入形状: {embeddings.shape}")
        print(f"{dataset_name}嵌入已保存至: " + embedding_file)
        return embedding_file

    dataset = TextDataProcessing(descriptions_file)
    return build_semantic(dataset, dataset_type)


def main():
    parser = argparse.ArgumentParser(description="生成图像文本描述和嵌入")
    parser.add_argument('--dataset_type', type=str, default='both', choices=['train', 'test', 'both'])
    args = parser.parse_args()

    if args.dataset_type in ['train', 'both']:
        print("=" * 50)
        print("开始处理训练集...")
        print("=" * 50)
        train_desc_file = generate_descriptions("train")
        train_embed_file = generate_embeddings(train_desc_file, "train")
        print("训练集文本描述和嵌入生成完成！")
    if args.dataset_type in ['test', 'both']:
        print("=" * 50)
        print("开始处理测试集...")
        print("=" * 50)
        test_desc_file = generate_descriptions("test")
        test_embed_file = generate_embeddings(test_desc_file, "test")
        print("测试集文本描述和嵌入生成完成！")
    print("所有任务完成！")


if __name__ == "__main__":
    main()
