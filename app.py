import streamlit as st
import google.generativeai as genai

# 👇 여기에 API 키를 넣어주세요
GOOGLE_API_KEY = "AIzaSyAOyVgnmN-3qnGt53ftiS8NmCfkfKvx7LI" 

st.set_page_config(page_title="API 진단기", page_icon="🩺")
st.title("🩺 AI 모델 연결 테스트")

try:
    # 1. 키 설정
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # 2. 사용 가능한 모델 목록 가져오기
    st.write("서버에 연결 중입니다...")
    models = list(genai.list_models())
    
    # 3. '글쓰기(generateContent)'가 가능한 모델만 골라내기
    available_models = []
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            
    # 4. 결과 출력
    if available_models:
        st.success(f"✅ 연결 성공! 사용 가능한 모델 {len(available_models)}개를 찾았습니다.")
        st.write("### 👇 이 이름들 중 하나를 써야 합니다:")
        st.code(available_models)
    else:
        st.error("❌ 연결은 됐는데, 사용할 수 있는 모델이 하나도 없습니다. (프로젝트 권한 문제)")
        
except Exception as e:
    st.error(f"❌ 연결 실패 (에러 메시지): {e}")









