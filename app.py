import datetime
import streamlit as st
from supabase import Client, create_client

# Supabase 連線設定
SUPABASE_URL = "https://nwveqnsdwdwburzerjad.supabase.co"
SUPABASE_KEY = "sb_publishable_jKIR3S-2XYQVaPgiC_SQfA_0eK2mbvz"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("寵物走失與拾獲通報系統")
st.write("請填寫以下資訊進行通報")

# 建立前端表單
with st.form("pet_report_form"):
  status = st.selectbox(
      "回報類型",
      ["lost", "found"],
      format_func=lambda x: "走失協尋" if x == "lost" else "好心拾獲",
  )

  user_name = st.text_input("聯絡人姓名")
  contact = st.text_input("聯絡電話")
  species = st.text_input("寵物種類")
  breed = st.text_input("寵物品種")
  location = st.text_input("發生地點")

  # 圖片上傳欄位
  uploaded_file = st.file_uploader(
      "上傳寵物照片", type=["jpg", "jpeg", "png"]
  )

  submitted = st.form_submit_button("確認送出通報")

  if submitted:
    if not user_name or not contact or not uploaded_file:
      st.error("請填寫姓名、電話並上傳寵物照片！")
    else:
      try:
        with st.spinner("正在上傳照片與儲存資料..."):
          # 1. 處理圖片上傳至 Supabase Storage
          file_bytes = uploaded_file.getvalue()
          file_ext = uploaded_file.name.split(".")[-1]
          file_name = (
              f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}"
          )

          # 上傳到 pet-images 儲存桶
          storage_res = supabase.storage.from_("pet-images").upload(
              path=file_name,
              file=file_bytes,
              file_options={"content-type": f"image/{file_ext}"},
          )

          # 取得圖片的公開訪問網址
          image_url = supabase.storage.from_("pet-images").get_public_url(
              file_name
          )

          # 2. 準備要寫入資料庫的內容
          pet_data = {
              "status": status,
              "user_name": user_name,
              "contact": contact,
              "species": species,
              "breed": breed,
              "location": location,
              "image_url": image_url,
              "embedding": None,  # 這邊先留空，等組員的 AI 程式碼對接後再寫入
          }

          # 3. 寫入 pets_reports 表格
          db_res = supabase.table("pets_reports").insert(pet_data).execute()

        st.success("🎉 通報成功！資料已成功送至雲端資料庫。")

      except Exception as e:
        st.error(f"發生錯誤：{e}")