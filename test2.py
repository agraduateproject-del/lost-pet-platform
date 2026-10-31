import torch

checkpoint = torch.load("best_cat_efficientnet22_b0.pt", map_location="cpu")
print("最後分類層的權重形狀 (Classes 數量, 隱藏特徵維度):", checkpoint["classifier.1.weight"].shape)
print("最後分類層的 Bias 長度 (品種總數):", checkpoint["classifier.1.bias"].shape)