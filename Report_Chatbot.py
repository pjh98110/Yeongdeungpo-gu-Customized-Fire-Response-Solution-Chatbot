import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "gpt_api_key" not in st.session_state:
    st.session_state.gpt_api_key = openai.api_key # gpt API Key

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]


# 세션 변수 체크
def check_session_vars():
    required_vars = ['selected_district', 'report_type']
    for var in required_vars:
        if var not in st.session_state:
            st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
            st.stop()


# 사이드바에서 행정동 선택
selected_district = st.sidebar.selectbox(
    "(1) 영등포구 행정동을 선택하세요:",
    ('당산동', '문래동', '영등포동', '신길동', '양평동', '도림동', '대림동', '여의동')
)
st.session_state.selected_district = selected_district

# 사이드바에서 보고서 타입 선택
report_type = st.sidebar.selectbox(
    "(2) 원하는 보고서 타입을 선택하세요:",
    ['영등포구 지구별 취약지역 솔루션', '영등포구 소방차의 원활한 진입로']
)
st.session_state.report_type = report_type


# GPT 프롬프트 엔지니어링 함수
def gpt_prompt(user_input, selected_district):
    base_prompt = f"""
    너는 지금부터 입력된 {st.session_state.report_type} 보고서 타입에 따라 사용자가 요구한 영등포구의 상세한 정보를 작성하는 친절하고 차분한 [보고서 프로그램]이다. 
    사용자는 영등포구의 다양한 정보를 원하는 [영등포구 주민]이며, 사용자에게 정확하고 자세한 영등포구 정보를 전달한다. 
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 답변한다.

    <규칙>
    1) 영등포구 지구별 취약지역 솔루션은 예시를 참고하여 더 발전시켜서 상세하게 보고서 형태로 제공한다.
    2) 영등포구 소방차의 원할한 진입로는 예시를 참고하여 더 발전시켜서 상세하게 보고서 형태로 제공한다.

    이 정보를 바탕으로 <규칙>에 따라서 답변한다.

    예시: 영등포구 지구별 취약지역 솔루션
    [상업지구]
    - 복합쇼핑몰, 시장
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - [상업지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.

    [업무지구]
    - 산업단지, 여의동
    - 여의동
    - [업무지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.
    
    [주거지구]
    - 아파트, 주택 단지, 쪽방 지역
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - [업무지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.

    예시: 영등포구 소방차의 원할한 진입로
    - 분석 결과를 참고해서 영등포구에서 사용자의 행정동에서 소방차가 원할하게 화재 현장에 진입하기 위해서 필요한 내용을 작성한다.
    - 영등포구에서 실현 가능한 내용으로 상세하고 자세하게 작성한다.(예시: 불법주정차 단속 활성화, 주차장 추가 건설 등) 

    사용자 입력: {user_input}
    행정동: {selected_district}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input, selected_district):
    base_prompt = f"""
    너는 지금부터 입력된 {st.session_state.report_type} 보고서 타입에 따라 사용자가 요구한 영등포구의 상세한 정보를 작성하는 친절하고 차분한 [보고서 프로그램]이다. 
    사용자는 영등포구의 다양한 정보를 원하는 [영등포구 주민]이며, 사용자에게 정확하고 자세한 영등포구 정보를 전달한다. 
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 답변한다.

    <규칙>
    1) 영등포구 지구별 취약지역 솔루션은 예시를 참고하여 더 발전시켜서 상세하게 보고서 형태로 제공한다.
    2) 영등포구 소방차의 원할한 진입로는 예시를 참고하여 더 발전시켜서 상세하게 보고서 형태로 제공한다.

    이 정보를 바탕으로 <규칙>에 따라서 답변한다.

    예시: 영등포구 지구별 취약지역 솔루션
    [상업지구]
    - 복합쇼핑몰, 시장
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - [상업지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.

    [업무지구]
    - 산업단지, 여의동
    - 여의동
    - [업무지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.
    
    [주거지구]
    - 아파트, 주택 단지, 쪽방 지역
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - [업무지구]에 대한 특징과 정확하고 상세한 내용을 추가로 작성한다.

    예시: 영등포구 소방차의 원할한 진입로
    - 분석 결과를 참고해서 영등포구에서 사용자의 행정동에서 소방차가 원할하게 화재 현장에 진입하기 위해서 필요한 내용을 작성한다.
    - 영등포구에서 실현 가능한 내용으로 상세하고 자세하게 작성한다.(예시: 불법주정차 단속 활성화, 주차장 추가 건설 등) 

    사용자 입력: {user_input}
    행정동: {selected_district}
    """
    return base_prompt

# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "gpt_messages" not in st.session_state:
    st.session_state.gpt_messages = [
        {"role": "system", "content": "GPT가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}
    ]

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = [
        {"role": "model", "parts": [{"text": "Gemini가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}]}
    ]


selected_chatbot = st.sidebar.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["GPT를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇", "Gemini를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇"],
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)

if selected_chatbot == "GPT를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇":
    colored_header(
        label='GPT를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇',
        description=None,
        color_name="gray-70",
    )

    # 세션 변수 체크
    check_session_vars()

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.gpt_messages = [
            {"role": "system", "content": "GPT가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gpt_messages" not in st.session_state:
        st.session_state.gpt_messages = [
            {"role": "system", "content": "GPT가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}
        ]
        
    for msg in st.session_state.gpt_messages:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.gpt_messages.append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gpt_prompt(prompt, st.session_state.selected_district)

        # 모델 호출 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.gpt_messages,
                max_tokens=1500,
                temperature=0.8,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            text = response.choices[0]['message']['content']

            # 응답 메시지 표시 및 저장
            st.session_state.gpt_messages.append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.write(text)
        except Exception as e:
            st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")

elif selected_chatbot == "Gemini를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇":
    colored_header(
        label='Gemini를 통한 영등포구 맞춤형 화재 대응 솔루션 챗봇',
        description=None,
        color_name="gray-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ["gemini-1.5-pro", 'gemini-1.5-flash']
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=2048, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "gemini_messages": [{"role": "model", "parts": [{"text": "Gemini가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}]}]
    }))

    # 이전 메시지 표시
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = [
            {"role": "model", "parts": [{"text": "Gemini가 사용자에게 영등포구 맞춤형 화재 대응 보고서를 출력해드립니다."}]}
        ]
        
    for msg in st.session_state.gemini_messages:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.gemini_messages.append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt, st.session_state.selected_district)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.gemini_messages)
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.gemini_messages.append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")
