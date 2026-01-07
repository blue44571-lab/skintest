# main.py
# 先打開終端機安裝所需套件
# pip install streamlit ultralytics opencv-python-headless pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

# --- 設定頁面標題 ---
st.set_page_config(page_title="皮膚偵測 AI 系統", layout="wide")
st.title("🔍 皮膚偵測與分析系統")
st.write("請選擇上傳圖片或直接拍照，並調整亮度進行即時 AI 偵測")

# --- 載入模型 (快取處理) ---
@st.cache_resource
def load_model():
    # 請確保 best.pt 放在與 main.py 同一個資料夾下
    # 如果沒有 best.pt，可以暫時改用 'yolov8n.pt' 來測試流程
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"模型載入失敗，請確認 'best.pt' 是否存在。錯誤訊息: {e}")
    st.stop()

# --- 側邊欄設定 ---
st.sidebar.header("功能設定")

# 1. 選擇輸入來源 (新增功能)
input_source = st.sidebar.radio("選擇圖片來源：", ("上傳圖片", "使用相機拍照"))

st.sidebar.markdown("---")
st.sidebar.header("參數調整")
# 亮度滑桿：範圍 0.5 到 2.0，預設 1.0 (不變)
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
# 信心度門檻
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 圖片獲取區域 ---
img_file_buffer = None

if input_source == "上傳圖片":
    img_file_buffer = st.file_uploader("請選擇一張皮膚照片 (jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])
elif input_source == "使用相機拍照":
    # 啟用相機功能
    img_file_buffer = st.camera_input("請點擊下方按鈕拍照")

# --- 核心處理邏輯 ---
if img_file_buffer is not None:
    # 讀取圖片 (無論是上傳還是拍照，格式都是 BytesIO，可以直接用 Image.open)
    image = Image.open(img_file_buffer)
    
    # 1. 調整亮度 (使用 PIL ImageEnhance)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    # 建立左右對照畫面
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("待測圖片 (已調亮度)")
        st.image(processed_image, caption="來源影像", use_container_width=True)
    
    # 2. 進行 YOLOv8 偵測
    # 將 PIL 轉為 OpenCV 格式供模型使用
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 按鈕觸發偵測
    if st.button("開始 AI 偵測", type="primary"):
        with st.spinner('AI 正在分析中...'):
            # 執行預測
            results = model.predict(source=img_bgr, conf=conf_threshold)
            
            # 取得畫好框的圖片 (OpenCV BGR -> PIL RGB)
            # results[0].plot() 回傳的是 BGR numpy array
            annotated_img_bgr = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("偵測結果")
                st.image(annotated_img_rgb, caption="AI 分析結果", use_container_width=True)
                
            # 顯示偵測統計
            # results[0].boxes 包含所有的偵測框
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            if num_detections > 0:
                st.success(f"偵測完成！共發現 {num_detections} 處目標。")
                
                # (選用) 如果你想顯示偵測到的類別名稱，可以解開下方註解
                # class_names = model.names
                # for box in boxes:
                #     cls_id = int(box.cls[0])
                #     conf = float(box.conf[0])
                #     st.info(f"偵測到: {class_names[cls_id]} (信心度: {conf:.2f})")
            else:
                st.warning("未偵測到任何目標，建議調整亮度或降低信心度門檻。")

#打開終端機執行指令
# streamlit run main.py --server.fileWatcherType none