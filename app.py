import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp

# ----------------------------------------------------------
# 👇 여기에 아까 성공했던 '진짜 API 키'를 붙여넣으세요!
GOOGLE_API_KEY = "AIzaSyAOyVgnmN-3qnGt53ftiS8NmCfkfKvx7LI" 
# ----------------------------------------------------------

# API 설정
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# 페이지 설정
st.set_page_config(page_title="Personal AI Stylist Pro", page_icon="✨", layout="centered")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

# --- 🔥 [핵심] 무적의 AI 연결 함수 (자동으로 되는 놈 찾기) ---
def ask_gemini(prompt):
    # AI 이름 후보들을 싹 다 준비했습니다. (이 중에 하나는 무조건 됩니다)
    candidates = [
        "gemini-1.5-flash",          # 1순위: 최신형 (이름표 없음)
        "models/gemini-1.5-flash",   # 2순위: 최신형 (이름표 있음)
        "gemini-pro",                # 3순위: 구형 (안정적)
        "models/gemini-pro",         # 4순위: 구형 (이름표 있음)
        "gemini-1.0-pro"             # 5순위: 구형 다른 이름
    ]
    
    last_error = None
    
    # 하나씩 다 찔러봅니다.
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text # 성공하면 바로 결과 주고 끝!
        except Exception as e:
            last_error = e
            continue # 실패하면 다음 후보로 넘어감 (조용히)
            
    # 다 실패하면 그때 에러를 띄웁니다.
    return f"AI 연결 실패: 모든 모델이 응답하지 않습니다. (마지막 에러: {last_error})"

# --- 분석 로직 (눈) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def analyze_personal_color(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(np.array(image))
        if not results.multi_face_landmarks:
            return None, "얼굴을 인식하지 못했습니다."
        landmarks = results.multi_face_landmarks[0].landmark
        img_np = np.array(image)
        h, w, _ = img_np.shape
        cx, cy = int(landmarks[116].x * w), int(landmarks[116].y * h)
        if cx >= w or cy >= h: return None, "얼굴이 화면 밖입니다."
        pixel = img_np[cy, cx]
        tone = "웜톤 (Warm Tone)" if pixel[0] > pixel[2] else "쿨톤 (Cool Tone)"
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
        if ratio > 1.05: type_ = "역삼각형 (어깨 발달형)"
        elif ratio < 0.95: type_ = "삼각형 (골반 발달형)"
        else: type_ = "모래시계형 (균형 잡힌 체형)"
        return ratio, type_

# --- 메인 화면 ---
st.title("✨ AI Stylist : 제니")
st.write("이제 에러 없이 당신을 코디해드립니다.")

tab1, tab2 = st.tabs(["🎨 퍼스널 컬러", "👗 체형 코디"])

with tab1:
    img_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("AI 스타일링 받기", key="btn_face"):
            with st.spinner('AI 제니가 눈에 불을 켜고 분석 중입니다...🔥'):
                tone, err = analyze_personal_color(image)
                if tone:
                    st.success(f"당신의 톤: **{tone}**")
                    prompt = f"사용자는 '{tone}'이야. 10년차 스타일리스트로서 립/블러셔 추천, 옷 색깔 추천, 피해야 할 색, 격려의 말을 이모지 섞어서 다정하게 해줘."
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error(err)

with tab2:
    img_file = st.file_uploader("전신 사진 업로드", type=["jpg", "png"], key="body")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("AI 코디 추천 받기", key="btn_body"):
            with st.spinner('AI 제니가 최적의 코디를 찾는 중입니다...👗'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    st.success(f"체형 타입: **{body_type}**")
                    prompt = f"사용자 체형은 '{body_type}'이야. 상의/하의 스타일 추천, 전체적인 팁을 자신감 있게 알려줘."
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error("전신이 잘 나온 사진을 올려주세요.")















