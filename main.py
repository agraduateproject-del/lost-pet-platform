import os
import io
import json
import uuid
import glob
import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForImageClassification
)
from rembg import remove
from supabase import create_client, Client

# -------------------------------------------------------------------------
# 1. 頁面配置與初始化
# -------------------------------------------------------------------------
st.set_page_config(page_title="走失寵物尋回系統", layout="wide")
st.title("🐾 走失寵物尋回系統")

# 初始化 Supabase 連線 (請替換為你的 Supabase URL 與 Key)
SUPABASE_URL = "https://nwveqnsdwdwburzerjad.supabase.co/"
SUPABASE_KEY = "sb_publishable_jKIR3S-2XYQVaPgiC_SQfA_0eK2mbvz"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

STORAGE_BUCKET = "pet_images"  # 存放原始照片的 Supabase Storage 名稱

# -------------------------------------------------------------------------
# 1-1. 全方位中英文品種與物種對照表 (完整保留所有翻譯字典)
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOG_HF_MODEL_NAME = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"

SPECIES_CN_TO_EN = {
    "狗": "dog",
    "貓": "cat",
    "倉鼠": "hamster",
    "兔子": "rabbit",
    "鳥": "bird",
    "豬": "pig",
    "烏龜": "turtle",
    "其他": "other"
}
SPECIES_EN_TO_CN = {v: k for k, v in SPECIES_CN_TO_EN.items()}

BREED_TRANSLATE_DICT = {
    # 狗 (Dog)
    "otterhound": "奧特獵犬", "cocker_spaniel": "可卡犬", "brittany_spaniel": "不列顛獵犬",
    "afghan_hound": "阿富汗獵犬", "maltese_dog": "瑪爾濟斯", "schipperke": "史其派克犬",
    "irish_setter": "愛爾蘭雪達犬", "pekinese": "北京犬", "golden_retriever": "黃金獵犬",
    "vizsla": "維茲拉犬", "welsh_springer_spaniel": "威爾斯史賓格犬", "staffordshire_bullterrier": "斯塔福郡牛頭㹴",
    "border_collie": "邊境牧羊犬", "irish_terrier": "愛爾蘭㹴", "eskimo_dog": "愛斯基摩犬",
    "pug": "巴哥犬", "kelpie": "卡爾比犬", "yorkshire_terrier": "約克夏爹利",
    "tibetan_terrier": "西藏㹴", "walker_hound": "步行者獵犬", "affenpinscher": "阿芬賓莎犬",
    "cardigan": "卡地甘威爾斯柯基犬", "english_springer": "英國史賓格犬", "english_foxhound": "英國獵狐犬",
    "west_highland_white_terrier": "西高地白㹴", "lakeland_terrier": "湖畔㹴", "rhodesian_ridgeback": "羅德西亞背脊犬",
    "gordon_setter": "戈登雪達犬", "lhasa": "拉薩阿普索", "curly": "捲毛尋回犬",
    "beagle": "米格魯", "tibetan_mastiff": "藏獒", "sussex_spaniel": "蘇塞克斯西班牙獵犬",
    "saint_bernard": "聖伯納犬", "toy_terrier": "玩具㹴", "standard_poodle": "標準貴賓狗",
    "bernese_mountain_dog": "伯恩山犬", "pomeranian": "博美犬", "ibizan_hound": "伊比贊獵犬",
    "redbone": "紅骨獵浣熊犬", "toy_poodle": "玩具貴賓狗", "basset": "巴吉度獵犬",
    "scottish_deerhound": "蘇格蘭獵鹿犬", "miniature_pinscher": "迷你品犬", "basenji": "巴森吉犬",
    "border_terrier": "邊境㹴", "bedlington_terrier": "貝林頓㹴", "kerry_blue_terrier": "凱利藍㹴",
    "weimaraner": "威瑪犬", "english_setter": "英國雪達犬", "bluetick": "布魯蒂克獵浣熊犬",
    "boston_bull": "波士頓㹴", "italian_greyhound": "義大利靈緹", "dandie_dinmont": "丹迪丁蒙㹴",
    "airedale": "萬能㹴", "irish_water_spaniel": "愛爾蘭水獵犬", "norfolk_terrier": "諾福克㹴",
    "wire": "剛毛獵狐㹴", "french_bulldog": "法國鬥牛犬", "soft": "愛爾蘭軟毛梗",
    "komondor": "可蒙犬", "african_hunting_dog": "非洲野犬", "siberian_husky": "西伯利亞雪橇犬",
    "newfoundland": "紐芬蘭犬", "bouvier_des_flandres": "法蘭德斯畜牧犬", "saluki": "薩路基獵犬",
    "shetland_sheepdog": "喜樂蒂牧羊犬", "collie": "牧羊犬", "rottweiler": "洛威拿",
    "silky_terrier": "絲毛㹴", "norwegian_elkhound": "挪威獵麋犬", "chihuahua": "吉娃娃",
    "leonberg": "蘭伯格犬", "norwich_terrier": "諾威奇㹴", "cairn": "凱恩㹴", "boxer": "拳師犬",
    "borzoi": "俄羅斯狼犬", "dhole": "亞洲野犬", "samoyed": "薩摩耶犬", "german_shepherd": "德國牧羊犬",
    "labrador_retriever": "拉布拉多獵犬", "blenheim_spaniel": "查理士王小獵犬", "groenendael": "格羅安達犬",
    "doberman": "杜賓犬", "great_dane": "大丹犬", "flat": "平毛尋回犬", "appenzeller": "阿彭策爾山犬",
    "shih": "西施犬", "japanese_spaniel": "日本獮猴犬", "greater_swiss_mountain_dog": "大瑞士山地犬",
    "black": "黑色俄羅斯㹴", "dingo": "澳洲野犬", "great_pyrenees": "大白熊犬", "whippet": "惠比特犬",
    "keeshond": "荷蘭毛獅犬", "malinois": "比利時瑪利諾犬", "american_staffordshire_terrier": "美國斯塔福郡㹴",
    "mexican_hairless": "墨西哥無毛犬", "giant_schnauzer": "巨型雪納瑞", "brabancon_griffon": "布魯塞爾格里芬犬",
    "kuvasz": "庫瓦斯犬", "miniature_poodle": "迷你貴賓狗", "irish_wolfhound": "愛爾蘭獵狼犬",
    "briard": "伯瑞犬", "clumber": "克倫伯西班牙獵犬", "standard_schnauzer": "標準雪納瑞",
    "bull_mastiff": "牛獒", "malamute": "阿拉斯加雪橇犬", "sealyham_terrier": "西裡哈姆㹴",
    "entlebucher": "恩特布赫山犬", "chow": "鬆獅犬", "papillon": "蝴蝶犬", "pembroke": "潘布魯克威爾斯柯基犬",
    "german_short": "德國短毛指示犬", "old_english_sheepdog": "古代英國牧羊犬", "chesapeake_bay_retriever": "切薩皮克灣尋回犬",
    "scotch_terrier": "蘇格蘭㹴", "australian_terrier": "澳洲㹴", "miniature_schnauzer": "迷你雪納瑞", "bloodhound": "血堤獵犬",

    # 貓 (Cat)
    "abyssinian": "阿比西尼亞貓", "american_shorthair": "美國短毛貓", "bengal": "孟加拉貓",
    "birman": "伯曼貓", "bombay": "孟買貓", "british_shorthair": "英國短毛貓",
    "cornish_rex": "康沃爾帝王貓", "devon_rex": "德文卷毛貓", "egyptian_mau": "埃及貓",
    "himalayan": "喜馬拉雅貓", "maine_coon": "緬因貓", "manx": "曼島貓",
    "norwegian_forest": "挪威森林貓", "oriental_shorthair": "東方短毛貓", "persian": "波斯貓",
    "ragdoll": "布偶貓", "russian_blue": "俄羅斯藍貓", "savannah": "薩凡納貓",
    "scottish_fold": "蘇格蘭摺耳貓", "siamese": "暹羅貓", "sphynx": "斯芬克斯無毛貓",
    "turkish_angora": "土耳其安哥拉貓",
    
    # 鳥 (Bird)
    "chicken": "雞", "duck": "鴨", "goose": "鵝", "other_bird": "其他鳥類", "parrot": "鸚鵡",
    
    # 倉鼠 (Hamster)
    "campbell_hamster": "坎貝爾矮子倉鼠", "chinese_hamster": "中國倉鼠", "european_hamster": "歐洲倉鼠",
    "grey_dwarf_hamster": "灰矮倉鼠", "roborovski_hamster": "老公公鼠", "romanian_hamster": "羅馬尼亞倉鼠",
    "syrian_hamster": "黃金鼠", "winter_white_hamster": "三線鼠",
    
    # 豬 (Pig)
    "american_guinea_hog": "美國天竺豬", "juliana": "朱莉安娜迷你豬", "kunekune": "庫內庫內豬",
    "vietnamese_pot_bellied": "越南大肚豬",
    
    # 兔子 (Rabbit)
    "californian": "加利福尼亞兔", "dutch": "荷蘭兔", "english_angora": "英國安哥拉兔",
    "flemish_giant": "法蘭德斯巨兔", "holland_lop": "荷蘭垂耳兔", "lionhead": "獅子頭兔",
    "mini_lop": "迷你垂耳兔", "mini_rex": "迷你雷克斯兔", "netherland_dwarf": "荷蘭侏儒兔",
    
    # 烏龜 (Turtle)
    "chinese_striape_necked_turtle": "中華花龜", "common_snapping_turtle": "鱷龜",
    "indian_star_tortoise": "印度星龜", "leopard_tortoise": "豹紋陸龜", "red_eared_slider": "巴西龜",
    "red_footed_tortoise": "紅腿陸龜", "reeves_turtle": "草龜", "sulcata_tortoise": "蘇卡達象龜",
    
    "其他": "其他 / 不知道"
}

def translate_breed(english_breed: str) -> str:
    if not english_breed:
        return "其他 / 不知道"
    cleaned = english_breed.strip().lower()
    cleaned_space = cleaned.replace("_", " ")
    
    if cleaned in BREED_TRANSLATE_DICT:
        return BREED_TRANSLATE_DICT[cleaned]
    if cleaned_space in BREED_TRANSLATE_DICT:
        return BREED_TRANSLATE_DICT[cleaned_space]
        
    return english_breed

EFFNET_SPECIES_CONFIG = {
    "cat": {
        "weight_path": os.path.join(BASE_DIR, "best_cat_efficientnet22_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "cat_classes22.json"),
        "num_classes": 22
    },
    "hamster": {
        "weight_path": os.path.join(BASE_DIR, "best_hamster8_efficientnet_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "hamster_classes8.json"),
        "num_classes": 8
    },
    "rabbit": {
        "weight_path": os.path.join(BASE_DIR, "best_rabbit9_v3_efficientnet_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "rabbit_classes9_v3.json"),
        "num_classes": 9
    },
    "bird": {
        "weight_path": os.path.join(BASE_DIR, "best_bird5_efficientnet_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "bird_classes5.json"),
        "num_classes": 5
    },
    "pig": {
        "weight_path": os.path.join(BASE_DIR, "best_pig4_v2_efficientnet_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "pig4_classes.json"),
        "num_classes": 4
    },
    "turtle": {
        "weight_path": os.path.join(BASE_DIR, "best_turtle8_efficientnet_b0.pt"),
        "json_path": os.path.join(BASE_DIR, "json", "turtle_classes8.json"),
        "num_classes": 8
    }
}

effnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_classes_from_json(species: str, expected_json_path: str, num_classes: int):
    target_file = None
    if os.path.exists(expected_json_path):
        target_file = expected_json_path
    else:
        candidates = (
            glob.glob(os.path.join(BASE_DIR, "json", f"*{species}*.json")) + 
            glob.glob(os.path.join(BASE_DIR, f"*{species}*.json"))
        )
        if candidates:
            target_file = candidates[0]

    if not target_file:
        return [f"{species}_breed_{i}" for i in range(num_classes)]

    try:
        with open(target_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for k in ["classes", "class_names", "labels", "id2label", "idx_to_class", "class_to_idx", "names"]:
                if k in data and isinstance(data[k], (list, dict)):
                    data = data[k]
                    break

        if isinstance(data, list):
            return [translate_breed(item) for item in data]

        if isinstance(data, dict):
            first_key = str(next(iter(data.keys())))
            if first_key.isdigit():
                sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                classes = [translate_breed(str(val)) for _, val in sorted_items]
            else:
                sorted_items = sorted(data.items(), key=lambda x: int(x[1]))
                classes = [translate_breed(str(key)) for key, _ in sorted_items]
            return classes
    except Exception as e:
        print(f"[Error] 解析 {target_file} 失敗: {e}")

    return [f"{species}_breed_{i}" for i in range(num_classes)]

def _load_single_effnet(weight_path: str, num_classes: int):
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource
def load_ai_models():
    yolo_base = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    yolo_best = YOLO(os.path.join(BASE_DIR, "best.pt"))
    
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dog_processor = AutoImageProcessor.from_pretrained(DOG_HF_MODEL_NAME)
    dog_model = AutoModelForImageClassification.from_pretrained(DOG_HF_MODEL_NAME)
    dog_model.eval()

    raw_dog_classes = [dog_model.config.id2label[i] for i in range(len(dog_model.config.id2label))]
    dog_classes = [translate_breed(c) for c in raw_dog_classes]

    effnet_models = {}
    effnet_classes = {"dog": dog_classes}
    
    for sp, cfg in EFFNET_SPECIES_CONFIG.items():
        effnet_classes[sp] = load_classes_from_json(sp, cfg["json_path"], cfg["num_classes"])
        w_path = cfg["weight_path"]
        if os.path.exists(w_path):
            try:
                effnet_models[sp] = _load_single_effnet(w_path, cfg["num_classes"])
            except Exception as e:
                print(f"[Warning] 載入 {w_path} 失敗: {e}")
    
    return (
        yolo_base, yolo_best, 
        dino_processor, dino_model, 
        clip_processor, clip_model,
        dog_processor, dog_model,
        effnet_models, effnet_classes
    )

(
    yolo_base, yolo_best, 
    dino_processor, dino_model, 
    clip_processor, clip_model,
    dog_processor, dog_model,
    effnet_models, effnet_classes
) = load_ai_models()

# -------------------------------------------------------------------------
# 2. 輔助函式
# -------------------------------------------------------------------------
def upload_original_image_to_supabase(original_img: Image.Image, original_filename: str) -> str:
    img_ByteArr = io.BytesIO()
    original_img.convert("RGB").save(img_ByteArr, format="JPEG", quality=90)
    file_bytes = img_ByteArr.getvalue()
    
    ext = original_filename.split(".")[-1] if "." in original_filename else "jpg"
    unique_filename = f"pet_{uuid.uuid4().hex}.{ext}"
    
    supabase.storage.from_(STORAGE_BUCKET).upload(
        path=unique_filename,
        file=file_bytes,
        file_options={"content-type": "image/jpeg"}
    )
    return supabase.storage.from_(STORAGE_BUCKET).get_public_url(unique_filename)

def remove_bg_and_letterbox(cropped_img: Image.Image, target_size: int = 224) -> Image.Image:
    fg_rgba = remove(cropped_img)
    neutral_gray_bg = Image.new("RGBA", fg_rgba.size, (128, 128, 128, 255))
    composite_img = Image.alpha_composite(neutral_gray_bg, fg_rgba).convert("RGB")
    return ImageOps.pad(composite_img, (target_size, target_size), color=(128, 128, 128))

def extract_reid_embedding(processed_img: Image.Image):
    processed_img = processed_img.convert("RGB")
    with torch.no_grad():
        dino_inputs = dino_processor(images=processed_img, return_tensors="pt")
        dino_outputs = dino_model(**dino_inputs)
        cls_token = dino_outputs.last_hidden_state[:, 0]
        patch_tokens = dino_outputs.last_hidden_state[:, 1:].mean(dim=1)
        dino_feat = F.normalize(cls_token + patch_tokens, p=2, dim=-1)
        
        clip_inputs = clip_processor(images=processed_img, return_tensors="pt")
        clip_outputs = clip_model.get_image_features(**clip_inputs)
        clip_raw = clip_outputs.image_embeds if hasattr(clip_outputs, "image_embeds") else clip_outputs[0]
        if clip_raw.dim() > 2:
            clip_raw = clip_raw.view(clip_raw.size(0), -1)
            
        clip_feat = F.normalize(clip_raw, p=2, dim=-1)
        
        if dino_feat.shape[-1] != 384:
            dino_feat = F.adaptive_avg_pool1d(dino_feat.unsqueeze(1), 384).squeeze(1)
        dino_feat = F.normalize(dino_feat, p=2, dim=-1)
            
        if clip_feat.shape[-1] != 128:
            clip_feat = F.adaptive_avg_pool1d(clip_feat.unsqueeze(1), 128).squeeze(1)
        clip_feat = F.normalize(clip_feat, p=2, dim=-1)
        
        combined = torch.cat([dino_feat * 0.85, clip_feat * 0.15], dim=-1)
        combined = F.normalize(combined, p=2, dim=-1)
        
    feat_list = combined[0].tolist()
    if len(feat_list) != 512:
        tensor_512 = F.adaptive_avg_pool1d(combined.unsqueeze(1), 512).squeeze(1)
        feat_list = tensor_512[0].tolist()
        
    return feat_list

def calibrate_confidence(raw_sim: float) -> float:
    clamped = max(0.0, min(1.0, float(raw_sim)))
    return clamped ** 1.3

def predict_pet_breed(cropped_img: Image.Image, species: str) -> str:
    norm_sp = SPECIES_CN_TO_EN.get(species, species).lower()
    rgb_crop = cropped_img.convert("RGB")

    if norm_sp == "dog":
        with torch.no_grad():
            inputs = dog_processor(images=rgb_crop, return_tensors="pt")
            logits = dog_model(**inputs).logits
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            eng_breed = dog_model.config.id2label.get(pred_idx, f"dog_breed_{pred_idx}")
            return translate_breed(eng_breed)

    if norm_sp in effnet_models:
        model = effnet_models[norm_sp]
        input_tensor = effnet_transform(rgb_crop).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())
        class_list = effnet_classes.get(norm_sp, [])
        if 0 <= pred_idx < len(class_list):
            return class_list[pred_idx]
        return f"{norm_sp}_breed_{pred_idx}"

    return "其他 / 不知道"

def identify_and_crop_pet(image: Image.Image):
    """已調整為 best.pt 優先，若未偵測到才使用 yolov8n.pt"""
    cropped_image = image
    w, h = image.size
    detected_species = "cat"
    
    # 1. 優先使用自行訓練的 best.pt
    best_results = yolo_best(image)
    for r in best_results:
        if len(r.boxes) > 0:
            cls_id = int(r.boxes[0].cls[0])
            detected_species = yolo_best.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0].tolist())
            pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
            cropped_image = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)))
            return cropped_image, detected_species

    # 2. 若 best.pt 沒抓到，才使用通用 yolov8n.pt
    base_results = yolo_base(image)
    for r in base_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = yolo_base.names[cls_id].lower()
            if class_name in ["cat", "dog", "bird"]:
                detected_species = class_name
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                cropped_image = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)))
                return cropped_image, detected_species
            
    return cropped_image, detected_species

# -------------------------------------------------------------------------
# 3. 使用者介面 (報案上傳與資料庫瀏覽功能)
# -------------------------------------------------------------------------
st.sidebar.header("📤 填寫報案與上傳資料")

status_cn = st.sidebar.selectbox("報案狀態", options=["拾獲", "走失"])
user_name = st.sidebar.text_input("報案人稱呼")
contact = st.sidebar.text_input("聯絡方式")

species_options = ["請選擇物種", "狗", "貓", "倉鼠", "兔子", "鳥", "豬", "烏龜", "其他"]
selected_species = st.sidebar.selectbox("物種選擇", options=species_options)

selected_breed = None
if selected_species not in ["請選擇物種", "其他"]:
    eng_sp = SPECIES_CN_TO_EN.get(selected_species, "")
    translated_breeds = list(effnet_classes.get(eng_sp, []))
    if "其他 / 不知道" in translated_breeds:
        translated_breeds.remove("其他 / 不知道")
    translated_breeds.insert(0, "其他 / 不知道")
    
    selected_breed = st.sidebar.selectbox("品種選擇", options=translated_breeds)

location = st.sidebar.text_input("發生地點")
remark = st.sidebar.text_area("備註說明 (非必填)")

uploaded_file = st.sidebar.file_uploader("上傳寵物照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    if st.button("🚀 開始比對與上傳報案"):
        if not user_name or not contact or not location:
            st.warning("⚠️ 請完整填寫「報案人稱呼」、「聯絡方式」與「發生地點」！")
        else:
            with st.spinner("正在進行 AI 影像裁切、特徵萃取與相反狀態交叉比對..."):
                
                cropped_img, detected_species_raw = identify_and_crop_pet(original_image)
                
                if selected_species == "請選擇物種":
                    final_species = SPECIES_EN_TO_CN.get(detected_species_raw, "貓")
                    final_breed = predict_pet_breed(cropped_img, final_species)
                    is_ai_predicted = True
                elif selected_species == "其他":
                    final_species = "其他"
                    final_breed = "其他 / 不知道"
                    is_ai_predicted = False
                else:
                    final_species = selected_species
                    if not selected_breed or selected_breed == "其他 / 不知道":
                        final_breed = predict_pet_breed(cropped_img, final_species)
                        is_ai_predicted = True
                    else:
                        final_breed = selected_breed
                        is_ai_predicted = False

                clean_reid_img = remove_bg_and_letterbox(cropped_img)
                query_embedding = extract_reid_embedding(clean_reid_img)
                
                public_img_url = upload_original_image_to_supabase(original_image, uploaded_file.name)
                opposite_status = "走失" if status_cn == "拾獲" else "拾獲"
                
                insert_data = {
                    "status": status_cn,
                    "user_name": user_name,
                    "contact": contact,
                    "species": final_species,
                    "breed": final_breed,
                    "location": location,
                    "remark": remark,
                    "image_url": public_img_url,
                    "embedding": query_embedding
                }
                insert_res = supabase.table("pets").insert(insert_data).execute()
                excluded_ids = []
                if insert_res.data and len(insert_res.data) > 0:
                    current_inserted_id = insert_res.data[0].get("id")
                    if current_inserted_id:
                        excluded_ids.append(current_inserted_id)

                with st.expander("🔍 檢視 AI 影像前處理過程", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(original_image, caption="原始上傳照片", use_container_width=True)
                    with c2:
                        st.image(cropped_img, caption="YOLO 框選裁切", use_container_width=True)
                    with c3:
                        st.image(clean_reid_img, caption="AI 去背灰底圖 (Re-ID)", use_container_width=True)

                if is_ai_predicted:
                    st.info(f"品種辨識結果：物種 【{final_species}】 / 品種 【{final_breed}】")

                target_count = 5
                
                match_response = supabase.rpc("match_pets_by_status", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.25,
                    "match_count": target_count,
                    "filter_status": opposite_status,
                    "filter_breed": final_breed,
                    "excluded_ids": excluded_ids
                }).execute()
                
                same_breed_results = match_response.data if match_response.data else []
                for item in same_breed_results:
                    item["match_stage"] = "✅ 同品種交叉比對"
                    excluded_ids.append(item.get("id"))

                global_fill_results = []
                if len(same_breed_results) < target_count:
                    needed_count = target_count - len(same_breed_results)
                    global_match_response = supabase.rpc("match_pets_by_status", {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.12,
                        "match_count": needed_count,
                        "filter_status": opposite_status,
                        "filter_breed": None,
                        "excluded_ids": excluded_ids
                    }).execute()
                    if global_match_response.data:
                        global_fill_results = global_match_response.data
                        for item in global_fill_results:
                            item["match_stage"] = "🌐 跨品種相似補齊"

                final_results = (same_breed_results + global_fill_results)[:target_count]

                target_section_title = "拾獲紀錄" if opposite_status == "拾獲" else "走失紀錄"
                st.subheader(target_section_title)
                
                if final_results and len(final_results) > 0:
                    cols = st.columns(5)
                    for idx, match in enumerate(final_results):
                        raw_sim = match.get('similarity', 0)
                        conf_score = calibrate_confidence(raw_sim)
                        m_user = match.get('user_name', '善心人士')
                        m_contact = match.get('contact', '未提供')
                        m_species = match.get('species', '未知')
                        m_breed = match.get('breed', '未知')
                        m_loc = match.get('location', '未填寫')
                        m_remark = match.get('remark', '無')
                        stage_tag = match.get('match_stage', '')
                        img_url = match.get('image_url', '')
                        m_status = match.get('status', opposite_status)

                        if conf_score >= 0.70:
                            level_badge = "🟢 極高機率同一隻！"
                        elif conf_score >= 0.50:
                            level_badge = "🟡 高度相似"
                        else:
                            level_badge = "⚪ 參考候選"

                        with cols[idx]:
                            st.markdown(f"#### 第 {idx + 1} 名")
                            if img_url and img_url.startswith("http"):
                                st.image(img_url, use_container_width=True)
                            else:
                                st.caption("*(無圖片)*")
                                
                            st.markdown(
                                f"**{stage_tag}**  \n"
                                f"狀態: `{m_status}`  \n"
                                f"相似度: `{conf_score:.2%}`  \n"
                                f"{level_badge}  \n"
                                f"聯絡人: `{m_user}`  \n"
                                f"電話: `{m_contact}`  \n"
                                f"地點: `{m_loc}`  \n"
                                f"品種: `{m_species} / {m_breed}`  \n"
                                f"備註: `{m_remark}`",
                                unsafe_allow_html=True
                            )
                else:
                    st.info("目前資料庫中尚無符合條件的對應寵物紀錄。")

# -------------------------------------------------------------------------
# 4. 資料庫紀錄瀏覽與多條件篩選功能
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📚 資料庫紀錄瀏覽與篩選")

db_col1, db_col2, db_col3 = st.columns(3)
with db_col1:
    filter_status = st.selectbox("依狀態篩選", options=["全部", "走失", "拾獲"], key="db_status")
with db_col2:
    filter_species = st.selectbox("依物種篩選", options=["全部", "狗", "貓", "倉鼠", "兔子", "鳥", "豬", "烏龜", "其他"], key="db_species")
with db_col3:
    filter_breed_options = ["全部"]
    if filter_species != "全部" and filter_species != "其他":
        eng_sp = SPECIES_CN_TO_EN.get(filter_species, "")
        breeds_list = list(effnet_classes.get(eng_sp, []))
        if "其他 / 不知道" in breeds_list:
            breeds_list.remove("其他 / 不知道")
        breeds_list.insert(0, "其他 / 不知道")
        filter_breed_options.extend(breeds_list)
    filter_breed = st.selectbox("依品種篩選", options=filter_breed_options, key="db_breed")

if st.button("🔍 查詢資料庫"):
    with st.spinner("正在從 Supabase 撈取資料..."):
        query = supabase.table("pets").select("*")
        
        if filter_status != "全部":
            query = query.eq("status", filter_status)
        if filter_species != "全部":
            query = query.eq("species", filter_species)
        if filter_breed != "全部":
            query = query.eq("breed", filter_breed)
            
        response = query.order("created_at", desc=True).execute()
        db_records = response.data if response.data else []
        
        st.success(f"共找到 {len(db_records)} 筆符合條件的紀錄：")
        
        if len(db_records) > 0:
            cols = st.columns(4)
            for idx, record in enumerate(db_records):
                col_idx = idx % 4
                r_status = record.get("status", "未知")
                r_user = record.get("user_name", "未知")
                r_contact = record.get("contact", "未提供")
                r_species = record.get("species", "未知")
                r_breed = record.get("breed", "未知")
                r_loc = record.get("location", "未填寫")
                r_remark = record.get("remark", "無")
                r_img = record.get("image_url", "")
                r_time = record.get("created_at", "")[:10]
                
                with cols[col_idx]:
                    if r_img and r_img.startswith("http"):
                        st.image(r_img, use_container_width=True)
                    else:
                        st.caption("*(無圖片)*")
                        
                    st.markdown(
                        f"狀態: `{r_status}`  \n"
                        f"聯絡人: `{r_user}`  \n"
                        f"電話: `{r_contact}`  \n"
                        f"地點: `{r_loc}`  \n"
                        f"品種: `{r_species} / {r_breed}`  \n"
                        f"備註: `{r_remark}`  \n"
                        f"時間: `{r_time}`",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
        else:
            st.info("資料庫中沒有找到符合該篩選條件的紀錄。")