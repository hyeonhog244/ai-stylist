import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp
import urllib.parse

# 페이지 설정
st.set_page_config(
    page_title="AI 스타일리스트 제니", 
    page_icon="✨", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# CSS 스타일 (진단 카드 디자인 추가)
st.markdown("""
    <style>
        @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
        html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
        .stApp { background-color: #F8F9FA; }
        
        /* 메인 컨테이너 */
        .block-container {
            background-color: #FFFFFF; padding: 2rem; border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 800px;
        }
        
        /* 진단 결과 카드 스타일 (NEW!) */
        .result-card {
            background-color: #FFF5F5; /* 연한 핑크 배경 */
            border: 2px solid #FFD6D6;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
            text-align: left;
        }
        .result-title {
            color: #FF6B6B;
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 10px;
            border-bottom: 2px dashed #FFD6D6;
            padding-bottom: 10px;
        }
        .result-content {
            font-size: 16px;
            line-height: 1.6;
            color: #495057;
        }
        
        h1 { color: #FF6B6B; text-align: center; font-weight: 800; }
        
        /* 버튼 스타일 */
        .stButton > button {
            width: 100%; border-radius: 30px; border: none; padding: 15px 20px;
            font-weight: bold; font-size: 16px; transition: all 0.3s ease;
            background: linear-gradient(90deg, #FF8E53 0%, #FF6B6B 100%); color: white;
        }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 10px rgba(0,0,0,0.2); }
        
        a[href*="oliveyoung"] { color: #86C041 !important; font-weight: bold; }
        a[href*="musinsa"] { color: #000000 !important; font-weight: bold; }
        
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 🔒 비밀 금고에서 API 키 가져오기
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key, transport='rest')
except Exception as e:
    st.error("🚨 API 키 오류: Streamlit Settings > Secrets에 키가 저장되어 있는지 확인해주세요.")
    st.stop()

# --- 📊 사이드바 ---
with st.sidebar:
    st.header("📢 앱 공유하기")
    my_app_url = "https://ai-stylist-hg7yfg6f4lzxpxu5xvt26k.streamlit.app"
    
    badge_url = f"https://hits.seeyoufarm.com/api/count/incr/badge.svg?url={my_app_url}&count_bg=%23FF6B6B&title_bg=%23555555&icon=streamlit.svg&icon_color=%23E7E7E7&title=VISITORS&edge_flat=false"
    st.markdown(f'<img src="{badge_url}" style="display:none">', unsafe_allow_html=True)

    if "view" in st.query_params and st.query_params["view"] == "master":
        st.markdown("### 👁️ (관리자용) 방문자 수")
        st.image(badge_url)
        st.caption("비밀 모드로 보고 계십니다!")

    st.markdown("---")
    st.caption("👇 링크 복사")
    st.code(my_app_url, language="text")
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={my_app_url}"
    st.image(qr_url, caption="📷 카메라로 접속!")

# 모델 자동 선택
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

# 분석 로직들
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def analyze_personal_color(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(np.array(image))
        if not results.multi_face_landmarks: return None, "얼굴 인식 실패"
        lm = results.multi_face_landmarks[0].landmark
        img_np = np.array(image)
        h, w, _ = img_np.shape
        cx, cy = int(lm[116].x * w), int(lm[116].y * h)
        if cx >= w or cy >= h: return None, "화면 밖 얼굴"
        pixel = img_np[cy, cx]
        tone = "웜톤" if pixel[0] > pixel[2] else "쿨톤"
        return tone, None

def analyze_body_shape(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(np.array(image))
        if not results.pose_landmarks: return None, None
        lm = results.pose_landmarks.landmark
        shoulder = abs(lm[11].x - lm[12].x)
        hip = abs(lm[23].x - lm[24].x)
        if hip == 0: hip = 0.1
        ratio = shoulder / hip
        if ratio > 1.05: type_ = "역삼각형"
        elif ratio < 0.95: type_ = "삼각형"
        else: type_ = "모래시계형"
        return ratio, type_

def analyze_face_shape(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
        results = face_mesh.process(np.array(image))
        if not results.multi_face_landmarks: return None, "인식 실패"
        lm = results.multi_face_landmarks[0].landmark
        img_h, img_w, _ = np.array(image).shape
        top, bottom = lm[10].y * img_h, lm[152].y * img_h
        left, right = lm[234].x * img_w, lm[454].x * img_w
        face_h, face_w = bottom - top, right - left
        if face_w == 0: return None, "오류"
        ratio = face_h / face_w
        if ratio > 1.5: shape = "긴 얼굴형"
        elif ratio < 1.2: shape = "둥근 얼굴형"
        else:
            jaw = abs(lm[58].x - lm[288].x) * img_w
            shape = "각진 얼굴형" if jaw/face_w > 0.9 else "계란형"
        return shape, None

# --- 메인 화면 ---
st.title("✨ AI 스타일리스트 : 제니")
st.write("AI가 분석하고, 어울리는 아이템까지 추천해드립니다.")

tab1, tab2, tab3 = st.tabs(["🎨 뷰티/메이크업", "👗 패션/코디", "💇‍♀️ 헤어스타일"])

with tab1:
    st.header("🎨 퍼스널 컬러 & 화장품 추천")
    img_file = st.file_uploader("얼굴 사진", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("진단 시작", key="btn_face"):
            with st.spinner('분석 중...'):
                tone, err = analyze_personal_color(image)
                if tone:
                    # 결과 카드 디자인 적용 (HTML/CSS)
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">🎨 진단 결과: {tone}</div>
                            <div class="result-content">
                                AI 제니가 분석한 당신의 퍼스널 컬러입니다.<br>
                                아래 추천 팁을 확인해보세요! 👇
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    result = ask_gemini(f"사용자는 '{tone}'이야. 10년차 뷰티 에디터로서 어울리는 립/블러셔 컬러와 메이크업 꿀팁을 핵심만 요약해서 알려줘.")
                    st.info(result)
                    
                    keyword = urllib.parse.quote(f"{tone}")
                    link = f"https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query={keyword}"
                    st.link_button(f"🫒 올리브영에서 '{tone}' 꿀템 찾기", link)
                else:
                    st.error(err)

with tab2:
    st.header("👗 체형 분석 & 코디 추천")
    img_file = st.file_uploader("전신 사진", type=["jpg", "png"], key="body")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("코디 추천받기", key="btn_body"):
            with st.spinner('분석 중...'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    # 결과 카드 디자인 적용
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">👗 체형 타입: {body_type}</div>
                            <div class="result-content">
                                신체 비율을 분석한 결과입니다.<br>
                                장점은 살리고 단점은 보완하는 코디법을 알려드릴게요! 👇
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    result = ask_gemini(f"체형 '{body_type}'에 어울리는 베스트 코디와 피해야 할 옷을 패션 MD처럼 콕 집어서 알려줘.")
                    st.info(result)
                    
                    link = "https://www.musinsa.com/main/musinsa/ranking"
                    st.link_button(f"🔥 무신사 랭킹 보고 옷 고르기", link)
                else:
                    st.error("전신 사진 필요")

with tab3:
    st.header("💇‍♀️ 얼굴형 맞춤 헤어")
    img_file = st.file_uploader("정면 얼굴", type=["jpg", "png"], key="hair")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("헤어 추천받기", key="btn_hair"):
            with st.spinner('분석 중...'):
                shape, err = analyze_face_shape(image)
                if shape:
                    # 결과 카드 디자인 적용
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">💇‍♀️ 얼굴형 진단: {shape}</div>
                            <div class="result-content">
                                얼굴의 가로/세로 비율을 분석했습니다.<br>
                                인생 머리를 찾아드릴게요! 👇
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    result = ask_gemini(f"얼굴형 '{shape}'에 찰떡인 앞머리/기장/펌 스타일을 헤어 디자이너처럼 추천해줘.")
                    st.info(result)
                    
                    keyword = urllib.parse.quote(f"{shape} 헤어스타일 추천")
                    link = f"https://www.youtube.com/results?search_query={keyword}"
                    st.link_button(f"▶️ 유튜브에서 '{shape}' 스타일 영상 보기", link)
                else:
                    st.error(err)






