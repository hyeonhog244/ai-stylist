import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp

# ----------------------------------------------------------
# 👇 방금 진단기에서 성공했던 그 API 키를 여기에 넣으세요!
GOOGLE_API_KEY = "AIzaSyAOyVgnmN-3qnGt53ftiS8NmCfkfKvx7LI" 
# ----------------------------------------------------------

# API 설정
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# 페이지 설정
st.set_page_config(page_title="Personal AI Stylist Pro", page_icon="✨", layout="centered")

# 스타일 숨기기 (깔끔하게)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- AI 도우미 함수 ---
def ask_gemini(prompt):
    # 💡 방금 성공한 모델 이름 'gemini-1.5-flash' 사용!
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- 분석 로직 (MediaPipe) ---
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
st.write("당신의 퍼스널 컬러와 체형을 분석하고, 맞춤형 스타일링을 제안합니다.")

tab1, tab2 = st.tabs(["🎨 퍼스널 컬러", "👗 체형 코디"])

# 탭 1: 퍼스널 컬러
with tab1:
    img_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        
        if st.button("AI 스타일링 받기", key="btn_face"):
            with st.spinner('AI 제니가 분석 중입니다...✍️'):
                tone, err = analyze_personal_color(image)
                if tone:
                    st.success(f"당신의 톤: **{tone}**")
                    
                    prompt = f"""
                    사용자는 퍼스널 컬러 진단 결과 '{tone}'이 나왔어.
                    너는 10년 차 패션 스타일리스트 '제니'야.
                    사용자에게:
                    1. 어울리는 립/블러셔 메이크업 컬러
                    2. 베스트 옷 색깔 3가지
                    3. 피해야 할 워스트 색깔
                    4. 오늘 날씨나 기분에 맞춘 따뜻한 조언 한마디
                    
                    이 내용을 이모지를 섞어서 친근하고 예쁘게 작성해 줘.
                    """
                    try:
                        ai_advice = ask_gemini(prompt)
                        st.markdown(ai_advice)
                        st.markdown("---")
                        keyword = "웜톤 립스틱" if "웜톤" in tone else "쿨톤 립스틱"
                        st.link_button("🛍️ 추천 아이템 보러가기", f"https://search.shopping.naver.com/search/all?query={keyword}")
                    except Exception as e:
                        st.error(f"AI 연결 오류: {e}")
                else:
                    st.error(err)

# 탭 2: 체형 분석
with tab2:
    img_file = st.file_uploader("전신 사진 업로드", type=["jpg", "png"], key="body")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        
        if st.button("AI 코디 추천 받기", key="btn_body"):
            with st.spinner('AI 제니가 코디를 고민 중입니다...👗'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    st.success(f"체형 타입: **{body_type}**")
                    
                    prompt = f"""
                    사용자의 체형은 '{body_type}'이야.
                    프로 스타일리스트로서:
                    1. 체형을 보완하는 상의 스타일 (넥라인, 핏)
                    2. 다리가 길어 보이는 하의 추천
                    3. 전체적인 밸런스를 위한 팁
                    
                    자신감을 주는 말투로 작성해 줘.
                    """
                    try:
                        ai_advice = ask_gemini(prompt)
                        st.markdown(ai_advice)
                        st.markdown("---")
                        st.link_button("🛍️ 추천 코디 쇼핑하기", f"https://search.shopping.naver.com/search/all?query={body_type} 코디")
                    except Exception as e:
                        st.error(f"AI 연결 오류: {e}")
                else:
                    st.error("전신이 잘 나온 사진을 올려주세요.")









