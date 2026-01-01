import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# 페이지 설정
st.set_page_config(page_title="AI 스타일리스트", page_icon="👗")

# --- 💡 로직 (원래 main.py에 있던 두뇌) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def analyze_personal_color(image):
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        # Streamlit은 이미지를 RGB로 읽으므로 바로 사용
        results = face_mesh.process(np.array(image))
        
        if not results.multi_face_landmarks:
            return None, "얼굴을 찾을 수 없습니다."

        landmarks = results.multi_face_landmarks[0].landmark
        img_np = np.array(image)
        h, w, _ = img_np.shape
        
        # 볼 중앙 좌표 (116번)
        cx, cy = int(landmarks[116].x * w), int(landmarks[116].y * h)
        
        # 좌표가 이미지 밖으로 나가는지 확인
        if cx >= w or cy >= h:
            return None, "얼굴이 너무 가장자리에 있습니다."

        pixel = img_np[cy, cx]
        red, green, blue = int(pixel[0]), int(pixel[1]), int(pixel[2])
        
        # 웜/쿨 판단
        tone = "Warm Tone (웜톤)" if red > blue else "Cool Tone (쿨톤)"
        
        # 추천 로직
        if tone == "Warm Tone (웜톤)":
            recommendation = {
                "jewelry": "골드 (Gold)",
                "makeup": {"lip": "코랄, 오렌지 레드, 브릭", "blusher": "피치, 살구색"},
                "avoid": "차가운 핑크, 실버, 푸른끼 회색"
            }
        else:
            recommendation = {
                "jewelry": "실버 (Silver)",
                "makeup": {"lip": "베이비 핑크, 플럼, 쿨레드", "blusher": "딸기우유 핑크, 라벤더"},
                "avoid": "카키, 오렌지, 짙은 브라운"
            }
            
        return tone, recommendation

def analyze_body_shape(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(np.array(image))
        
        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark
        
        # 어깨/골반 너비 (비율 계산)
        shoulder = abs(lm[11].x - lm[12].x)
        hip = abs(lm[23].x - lm[24].x)
        ratio = shoulder / hip if hip > 0 else 1.0
        
        # 체형 추천 로직
        if ratio > 1.05:
            advice = {
                "type": "역삼각형 (어깨 발달형)",
                "desc": "시크하고 멋진 체형! 하체 볼륨을 살려보세요.",
                "styling": {"top": "브이넥, 어두운 컬러", "bottom": "와이드 팬츠, 밝은 컬러", "accessary": "긴 목걸이"}
            }
        elif ratio < 0.95:
            advice = {
                "type": "삼각형 (하체 발달형)",
                "desc": "우아한 라인입니다! 상의에 포인트를 주세요.",
                "styling": {"top": "퍼프 소매, 화려한 패턴", "bottom": "스트레이트 핏, 어두운 컬러", "accessary": "화려한 귀걸이"}
            }
        else:
            advice = {
                "type": "모래시계/직사각형 (균형형)",
                "desc": "비율이 좋습니다! 허리를 강조해보세요.",
                "styling": {"top": "크롭티, 허리 벨트 자켓", "bottom": "하이웨스트, 부츠컷", "accessary": "허리 벨트"}
            }
            
        return ratio, advice

# --- 🖥️ 화면 구성 (Streamlit) ---
st.title("👗 나만의 AI 스타일리스트")
st.write("당신의 사진을 업로드하면 **퍼스널 컬러**와 **체형**을 분석해 드려요!")

tab1, tab2 = st.tabs(["🎨 퍼스널 컬러 진단", "my 체형 분석 & 코디"])

# 탭 1: 퍼스널 컬러
with tab1:
    st.header("나의 퍼스널 컬러는?")
    uploaded_file_face = st.file_uploader("얼굴 사진 업로드", type=["jpg", "png", "jpeg"], key="face")

    if uploaded_file_face is not None:
        image = Image.open(uploaded_file_face)
        st.image(image, caption='업로드한 사진', width=300)
        
        if st.button("분석 시작", key="btn_face"):
            with st.spinner('분석 중...'):
                tone, info = analyze_personal_color(image)
                
                if tone:
                    st.success(f"당신의 톤은 **{tone}** 입니다! 🎉")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"✨ **추천 주얼리**: {info['jewelry']}")
                        st.write(f"💄 **립스틱**: {info['makeup']['lip']}")
                    with col2:
                        st.write(f"😊 **블러셔**: {info['makeup']['blusher']}")
                        st.warning(f"🚫 **피하면 좋은 색**: {info['avoid']}")
                    
                    st.markdown("---")
                    st.link_button(f"💄 {tone} 추천 립스틱 쇼핑하기", f"https://search.shopping.naver.com/search/all?query={tone} 립스틱")
                else:
                    st.error(info)

# 탭 2: 체형 분석
with tab2:
    st.header("나의 체형 비율과 추천 코디")
    uploaded_file_body = st.file_uploader("전신 사진 업로드", type=["jpg", "png", "jpeg"], key="body")

    if uploaded_file_body is not None:
        image = Image.open(uploaded_file_body)
        st.image(image, caption='업로드한 사진', width=300)
        
        if st.button("체형 분석 시작", key="btn_body"):
            with st.spinner('측정 중...'):
                ratio, advice = analyze_body_shape(image)
                
                if ratio:
                    st.success(f"결과: **{advice['type']}**")
                    st.metric("어깨:골반 비율", round(ratio, 2))
                    st.write(f"💡 {advice['desc']}")
                    
                    st.subheader("추천 코디")
                    st.write(f"- 👚 **상의**: {advice['styling']['top']}")
                    st.write(f"- 👖 **하의**: {advice['styling']['bottom']}")
                    
                    st.markdown("---")
                    keyword = advice['styling']['bottom'].split(',')[0]
                    st.link_button(f"🛍️ {keyword} 쇼핑하러 가기", f"https://search.shopping.naver.com/search/all?query={keyword}")
                else:
                    st.error("전신이 잘 나온 사진을 써주세요!")