import os
import torch

pt_files = [
    "best_cat_efficientnet22_b0.pt",
    "best_hamster8_efficientnet_b0.pt",
    "best_rabbit9_v3_efficientnet_b0.pt",
    "best_bird5_efficientnet_b0.pt",
    "best_pig4_v2_efficientnet_b0.pt",
    "best_turtle8_efficientnet_b0.pt"
]

for f in pt_files:
    if not os.path.exists(f):
        print(f"❌ 找不到檔案: {f}")
        continue
        
    ckpt = torch.load(f, map_location="cpu")
    print(f"\n=== 檢查 {f} ===")
    
    if isinstance(ckpt, dict):
        # 印出字典裡的第一層 key (排除龐大的權重參數)
        keys = [k for k in ckpt.keys() if not k.startswith(("features.", "classifier."))]
        print("檔案包含的 Keys:", keys[:10])
        
        # 檢查常見的類別儲存欄位
        for key_name in ["classes", "class_names", "class_to_idx", "idx_to_class"]:
            if key_name in ckpt:
                print(f"✅ 找到品種清單 ({key_name}):", ckpt[key_name])
                break
        else:
            print("⚠️ 此檔案僅包含純模型權重 (state_dict)，未內建品種名稱。")
    else:
        print("檔案格式為完整模型物件:", type(ckpt))