from config import config
import os
import torch
import numpy as np

os.environ["HF_ENDPOINT"] = config.paths.HF_ENDPOINT

import data_utils
from models import CLIPModel

dataset = config.dataset.dataset  # ["CIFAR-10","CIFAR-20","STL-10","ImageNet-10","ImageNet-Dogs","DTD","UCF101","ImageNet"]

dataloader_train, dataloader_test = data_utils.get_dataloader(
    dataset=dataset, batch_size=config.dataset.embed_infer_bs
)
model = CLIPModel(model_name=config.CLIPModel).cuda()
model.eval()

features, labels = [], []
print("Inferring image features and labels...")
for iteration, (x, y) in enumerate(dataloader_train):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
    features.append(feature.cpu().numpy())
    labels.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_train)}]")
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)
print("Feature shape:", features.shape, "Label shape:", labels.shape)

features_test, labels_test = [], []
print("Inferring test image features and labels...")
for iteration, (x, y) in enumerate(dataloader_test):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
    features_test.append(feature.cpu().numpy())
    labels_test.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_test)}]")
features_test = np.concatenate(features_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)

# Optional CIFAR-20 coarse mapping moved to config
if dataset == "CIFAR-20" and config.dataset.use_cifar20_coarse_map:
    labels_copy = labels.copy()
    labels_test_copy = labels_test.copy()
    for i, group in enumerate(config.dataset.cifar20_coarse_map):
        for j in group:
            labels[labels_copy == j] = i
            labels_test[labels_test_copy == j] = i

out_dir = config.dataset.data_dir
os.makedirs(out_dir, exist_ok=True)
np.save(f"{out_dir}/{dataset}_image_embedding_train.npy", features)
np.save(f"{out_dir}/{dataset}_image_embedding_test.npy", features_test)
np.savetxt(f"{out_dir}/{dataset}_labels_train.txt", labels)
np.savetxt(f"{out_dir}/{dataset}_labels_test.txt", labels_test)
