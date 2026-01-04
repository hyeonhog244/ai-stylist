import streamlit as st
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz
import json

# 페이지 설정
st.set_page_config(page_title="AI 스타일리스트 제니", page_icon="✨")

# 비밀 금고에서 API 키만 가져오기
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except:
    st.error("구글 API 키가 없습니다.")
    st.stop()

# 사이드바에서 키 파일 업로드 받기 (이러면 에러 날 일이 없습니다!)
with st.sidebar:
    st.header("⚙️ 설정 (관리자용)")
    key_file = st.file_uploader("구글 시트 키 파일(.json)을 올려주세요", type="json")

def save_to_sheet(category, result):
    if key_file is None:
        st.warning("⚠️ 데이터 저장을 하려면 사이드바에 키 파일을 올려주세요!")
        return
    
    try:
        # 업로드된 파일을 바로 읽어서 씁니다 (복붙 실수 원천 차단!)
        key_data = json.load(key_file)
        scope = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(key_data, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open("ai_stylist_data").sheet1
        now = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, category, result])
        st.toast("✅ 엑셀 저장 완료!")
    except Exception as e:
        st.error(f"저장 실패: {e}")

# ... (나머지 UI 코드는 동일) ...
# (코드가 너무 길어지니, 일단 위 '백스페이스' 방법부터 해보시고 안 되면 이 코드를 드릴게요!)







