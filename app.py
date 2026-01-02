import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp
import urllib.parse
import math

# ----------------------------------------------------------
# 👇 여기에 아까 성공했던 '진짜 API 키'를 따옴표 안에 넣으세요!
GOOGLE_API_KEY = "AIzaSyAgWZ2KiMIAuIMMpWK--SB476Csa_e8Yrg"
# ----------------------------------------------------------

# 페이지 설정 (탭 아이콘과 제목)
st.set_page_config(page_title="AI 스타일리스트 제니", page_icon="✨", layout="centered")

# --- ✨ 디자인 업그레이드 (CSS) ---
st.markdown("""
    <style>
        /* 웹 폰트 적용 (Pretendard) */
        @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
        html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
        
        /* 전체 배경색 */
        .stApp { background-color: #F8F9FA; }
        
        /* 메인 컨테이너 디자인 */
        .block-container {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 800px;
        }
        
        /* 제목 스타일 */
        h1 { color: #FF6B6B; text-align: center; font-weight: 800; }
        
        /* 버튼 디자인 (동글동글하고 예쁘게) */
        .stButton > button {
            width: 100%;
            border-radius: 30px;
            background: linear-gradient(90deg, #FF8E53 0%, #FF6B6B 100%);
            color: white;
            border: none;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 14px rgba(255, 107, 107, 0.4);
        }
        
        /* 탭 디자인 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 15px;
            background-color: #F1F3F5;
            font-weight: bold;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF6B6B !important;
            color: white !important;
        }
        
        /* 불필요한 요소 숨기기 */
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# API 설정
try:
    genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# --- 사이드바 (공유 기능) ---
with st.sidebar:
    st.header("📢 친구에게 자랑하기")
    my_app_url = "https://ai-stylist-hg7yfg6f4lzxpxu5xvt26k.streamlit.app"
    st.caption("👇 링크를 복사해서 공유하세요!")
    st.code(my_app_url, language="text")
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={my_app_url}"
    st.image(qr_url, caption="📱 카메라로 찍으면 바로 접속!")
    st.markdown("---")
    st.info("✨ Tip: 친구들도 무료로 진단받을 수 있어요!")

# --- AI 모델 자동 선택 ---
def get_working_model_name():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
                if 'pro' in m.name: return m.name
        return list(genai.list_models())[0].name
    except:
        return "models/gemini-1.5-flash"

def ask_gemini(prompt):
    model_name = get_working_model_name()
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 응답 오류: {e}"

# --- 분석 로직 (MediaPipe) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# 1. 퍼스널 컬러 분석
def analyze_personal_color(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(np.array(image))
        if not results.multi_face_landmarks: return None, "얼굴을 인식하지 못했습니다."
        landmarks = results.multi_face_landmarks[0].landmark
        img_np = np.array(image)
        h, w, _ = img_np.shape
        cx, cy = int(landmarks[116].x * w), int(landmarks[116].y * h)
        if cx >= w or cy >= h: return None, "얼굴이 화면 밖입니다."
        pixel = img_np[cy, cx]
        tone = "웜톤 (Warm Tone)" if pixel[0] > pixel[2] else "쿨톤 (Cool Tone)"
        return tone, None

# 2. 체형 분석
def analyze_body_shape(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(np.array(image))
        if not results.pose_landmarks: return None, None
        lm = results.pose_landmarks.landmark
        shoulder = abs(lm[11].x - lm[12].x)
        hip = abs(lm[23].x - lm[24].x)
        if hip == 0: hip = 0.1
        ratio = shoulder / hip
        if ratio > 1.05: type_ = "역삼각형 (어깨 발달형)"
        elif ratio < 0.95: type_ = "삼각형 (골반 발달형)"
        else: type_ = "모래시계형 (균형 잡힌 체형)"
        return ratio, type_

# 3. 🔥 [신규] 얼굴형 분석 (헤어스타일용)
def analyze_face_shape(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
        results = face_mesh.process(np.array(image))
        if not results.multi_face_landmarks: return None, "얼굴 인식 실패"
        lm = results.multi_face_landmarks[0].landmark
        img_h, img_w, _ = np.array(image).shape
        
        # 주요 좌표 가져오기
        top = lm[10].y * img_h      # 이마 상단
        bottom = lm[152].y * img_h  # 턱 끝
        left = lm[234].x * img_w    # 왼쪽 광대
        right = lm[454].x * img_w   # 오른쪽 광대
        
        face_height = bottom - top
        face_width = right - left
        
        if face_width == 0: return None, "측정 오류"
        ratio = face_height / face_width
        
        # 간단한 비율 기반 얼굴형 판단
        if ratio > 1.5: shape = "긴형 (Long)"
        elif ratio < 1.2: shape = "둥근형 (Round)"
        else:
            # 턱 각도 체크 (간단 버전)
            jaw_width = abs(lm[58].x - lm[288].x) * img_w
            if jaw_width / face_width > 0.9: shape = "각진형 (Square)"
            else: shape = "계란형 (Oval)"
            
        return shape, None

# --- 메인 화면 구성 ---
st.title("✨ AI 스타일리스트 : 제니")
st.write("당신의 사진을 분석해 퍼스널 컬러, 체형, 헤어스타일까지 완벽하게 컨설팅해드립니다.")

# 탭 메뉴 구성 (3개)
tab1, tab2, tab3 = st.tabs(["🎨 퍼스널 컬러", "👗 체형 코디", "💇‍♀️ 헤어스타일"])

# 탭 1: 퍼스널 컬러
with tab1:
    st.header("🎨 퍼스널 컬러 진단")
    img_file = st.file_uploader("얼굴이 잘 나온 사진을 올려주세요", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=250)
        if st.button("💄 메이크업&코디 추천받기", key="btn_face"):
            with st.spinner('AI 제니가 피부톤을 분석 중입니다...✨'):
                tone, err = analyze_personal_color(image)
                if tone:
                    st.success(f"당신의 톤은 **{tone}** 입니다!")
                    prompt = f"사용자는 '{tone}'이야. 10년차 스타일리스트 '제니'로서 립/블러셔 컬러, 베스트 의상 컬러 추천과 격려의 말을 다정하게 이모지 섞어서 해줘."
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error(err)

# 탭 2: 체형 코디
with tab2:
    st.header("👗 체형 맞춤 코디")
    img_file = st.file_uploader("전신 사진을 올려주세요", type=["jpg", "png"], key="body")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=250)
        if st.button("👖 체형 보완 코디 추천받기", key="btn_body"):
            with st.spinner('AI 제니가 체형 비율을 계산 중입니다...📏'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    st.success(f"당신의 체형은 **{body_type}** 입니다!")
                    prompt = f"사용자 체형은 '{body_type}'이야. '제니'로서 체형을 보완하는 상의/하의 핏과 스타일링 팁을 자신감 있게 알려줘."
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error("전신이 잘 나온 사진이 필요합니다.")

# 탭 3: 헤어스타일 (신규!)
with tab3:
    st.header("💇‍♀️ 인생 헤어스타일 찾기")
    st.write("얼굴형을 분석해 가장 잘 어울리는 헤어를 찾아드려요.")
    img_file = st.file_uploader("정면 얼굴 사진을 올려주세요", type=["jpg", "png"], key="hair")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=250)
        if st.button("✂️ 헤어스타일 추천받기", key="btn_hair"):
            with st.spinner('AI 제니가 얼굴형을 분석 중입니다...📐'):
                shape, err = analyze_face_shape(image)
                if shape:
                    st.success(f"당신의 얼굴형은 **{shape}** 입니다!")
                    prompt = f"""
                    사용자의 얼굴형은 '{shape}'이야. 청담동 헤어 디자이너 '제니'로서:
                    1. 이 얼굴형의 특징과 장점
                    2. 어울리는 앞머리 유무와 스타일
                    3. 베스트 커트 스타일 (기장 포함)
                    4. 추천 펌과 염색 컬러
                    
                    전문적이면서도 친근하게 이모지를 섞어서 제안해 줘.
                    """
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error(err)














