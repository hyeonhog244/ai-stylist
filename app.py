import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp

# 페이지 설정
st.set_page_config(page_title="Personal AI Stylist Pro", page_icon="✨", layout="centered")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🔑 사이드바: 키 입력
# ----------------------------------------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")
    st.info("반드시 'Create new project'로 만든 새 키를 넣어주세요!")
    
    # 공백/따옴표 자동 제거 기능 포함
    raw_api_key = st.text_input("Google AI Key 입력", type="password", placeholder="AIza... 붙여넣기")
    api_key = raw_api_key.strip().replace('"', '').replace("'", "")

    if not api_key:
        st.warning("👈 왼쪽 빈칸에 API 키를 넣어주세요!")
        st.stop()

    # 🔥 [핵심] 일반 통신(REST) 모드로 설정 (서버 차단 회피)
    try:
        genai.configure(api_key=api_key, transport='rest')
    except Exception as e:
        st.error(f"설정 오류: {e}")

# --- AI 도우미 함수 ---
def ask_gemini(prompt):
    # 'latest'를 붙여서 가장 최신 버전 강제 호출
    model_name = 'models/gemini-1.5-flash-latest'
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 실패 시 상세 에러 메시지 출력
        return f"죄송합니다. 오류가 발생했습니다.\n\n원인: {e}\n\n💡 해결팁: AI Studio에서 'Create new project'로 키를 다시 발급받아 보세요."

# --- 분석 로직 (MediaPipe) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

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
st.write("AI가 당신을 분석하고 맞춤형 스타일링을 제안합니다.")

tab1, tab2 = st.tabs(["🎨 퍼스널 컬러", "👗 체형 코디"])

with tab1:
    img_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        if st.button("AI 스타일링 받기", key="btn_face"):
            with st.spinner('AI 제니가 분석 중입니다...'):
                tone, err = analyze_personal_color(image)
                if tone:
                    st.success(f"당신의 톤: **{tone}**")
                    prompt = f"사용자는 '{tone}'이야. 10년차 스타일리스트로서 립/블러셔/옷 컬러 추천과 격려를 이모지 섞어서 다정하게 해줘."
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
            with st.spinner('AI 제니가 코디를 찾는 중입니다...'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    st.success(f"체형 타입: **{body_type}**")
                    prompt = f"사용자 체형은 '{body_type}'이야. 상의/하의 스타일 추천과 팁을 자신감 있게 알려줘."
                    result = ask_gemini(prompt)
                    st.markdown(result)
                else:
                    st.error("전신이 잘 나온 사진을 올려주세요.")














