import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import mediapipe as mp

# ----------------------------------------------------------
# 👇 여기에 아까 복사한 API 키를 붙여넣으세요!
# 예시: GOOGLE_API_KEY = "AIzaSyD..."
GOOGLE_API_KEY = "AIzaSyDDIVKPwLheVt2dey9choqZldlfSG47uQY"
# ----------------------------------------------------------

# API 설정
genai.configure(api_key=GOOGLE_API_KEY)

# 페이지 설정
st.set_page_config(page_title="Personal AI Stylist Pro", page_icon="✨", layout="centered")

# 스타일 숨기기
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- AI 도우미 함수 (Gemini에게 말 걸기) ---
def ask_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- 기존 분석 로직 (눈) ---
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
st.write("단순한 분석이 아닙니다. 생성형 AI가 당신만을 위한 스타일링 조언을 해드립니다.")

tab1, tab2 = st.tabs(["🎨 퍼스널 컬러", "👗 체형 코디"])

# 탭 1: 퍼스널 컬러 + AI 조언
with tab1:
    img_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "png"], key="face")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        
        if st.button("AI 스타일링 받기", key="btn_face"):
            with st.spinner('AI가 얼굴을 분석하고 편지를 쓰는 중입니다...✍️'):
                tone, err = analyze_personal_color(image)
                if tone:
                    st.success(f"당신의 톤: **{tone}**")
                    
                    # 💡 여기가 핵심! AI에게 프롬프트 보내기
                    prompt = f"""
                    사용자는 퍼스널 컬러 진단 결과 '{tone}'이 나왔어.
                    너는 친절하고 센스 있는 10년 차 패션 스타일리스트 '제니'야.
                    사용자에게 이 톤에 어울리는:
                    1. 메이크업 팁 (립, 블러셔 컬러 구체적으로)
                    2. 어울리는 옷 색깔
                    3. 피해야 할 색깔
                    4. 따뜻한 격려의 한마디
                    
                    이 내용을 이모지를 섞어서 보기 편하게, 친근한 말투로 작성해 줘.
                    """
                    
                    # Gemini가 쓴 글 받아오기
                    ai_advice = ask_gemini(prompt)
                    st.markdown(ai_advice) # 화면에 출력
                    
                    st.markdown("---")
                    keyword = "웜톤 립스틱" if "웜톤" in tone else "쿨톤 립스틱"
                    st.link_button("🛍️ 추천 아이템 보러가기", f"https://search.shopping.naver.com/search/all?query={keyword}")
                else:
                    st.error(err)

# 탭 2: 체형 분석 + AI 조언
with tab2:
    img_file = st.file_uploader("전신 사진 업로드", type=["jpg", "png"], key="body")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=200)
        
        if st.button("AI 코디 추천 받기", key="btn_body"):
            with st.spinner('체형 분석 후 코디를 구상 중입니다...👗'):
                ratio, body_type = analyze_body_shape(image)
                if ratio:
                    st.success(f"체형 타입: **{body_type}**")
                    
                    # 💡 AI에게 프롬프트 보내기
                    prompt = f"""
                    사용자의 체형은 '{body_type}'이야. (어깨와 골반 비율: {ratio:.2f})
                    너는 프로 패션 컨설턴트야.
                    이 체형의 장점을 살리고 단점을 보완할 수 있는:
                    1. 상의 스타일 추천 (구체적인 넥라인, 핏)
                    2. 하의 스타일 추천
                    3. 전체적인 스타일링 팁 (액세서리 등)
                    
                    자신감을 심어주는 말투로 예쁘게 작성해 줘.
                    """
                    
                    ai_advice = ask_gemini(prompt)
                    st.markdown(ai_advice)
                    
                    st.markdown("---")
                    keyword = "와이드 팬츠" # 간단히 예시
                    st.link_button("🛍️ 추천 코디 쇼핑하기", f"https://search.shopping.naver.com/search/all?query={body_type} 코디")
                else:
                    st.error("전신 사진을 다시 확인해 주세요.")



