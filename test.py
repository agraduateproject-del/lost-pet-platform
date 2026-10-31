from transformers import AutoModelForImageClassification

DOG_HF_MODEL_NAME = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"

print("正在下載/讀取 Hugging Face 狗品種模型...")
dog_model = AutoModelForImageClassification.from_pretrained(DOG_HF_MODEL_NAME)

# 取得模型的 id2label 字典
labels_dict = dog_model.config.id2label

print(f"\n模型總共有 {len(labels_dict)} 個品種，清單如下：\n")
for idx, label in sorted(labels_dict.items()):
    print(f'"{label}": "",')