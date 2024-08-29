import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd
from datetime import datetime, timedelta

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
    ['영등포구 지구별 취약지역 솔루션', '영등포구 소방차의 원활한 진입로 확보']
)
st.session_state.report_type = report_type

# 기상청 단기예보 API 불러오기
API_KEY = st.secrets["secrets"]["WEATHER_KEY"] # 공공데이터 포털 API KEY

# 기상청 API 엔드포인트 URL을 정의
BASE_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

# 날짜와 시도 정보를 매핑하는 함수
def weather_info(date, sido):
    # 시도별로 기상청 격자 좌표를 정의
    sido_coordinates = {
        '서울특별시': (60, 127),
        '부산광역시': (98, 76),
        '대구광역시': (89, 90),
        '인천광역시': (55, 124),
        '광주광역시': (58, 74),
        '대전광역시': (67, 100),
        '울산광역시': (102, 84),
        '세종특별자치시': (66, 103),
        '경기도': (60, 120),
        '강원특별자치도': (73, 134),
        '충청북도': (69, 107),
        '충청남도': (68, 100),
        '전북특별자치도': (63, 89),
        '전라남도': (51, 67),
        '경상북도': (91, 106),
        '경상남도': (91, 77),
        '제주특별자치도': (52, 38),
    }

    if sido not in sido_coordinates:
        raise ValueError(f"'{sido}'는 유효한 시도가 아닙니다.")
    
    nx, ny = sido_coordinates[sido]

    params = {
        'serviceKey': API_KEY,
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'JSON',
        'base_date': date,
        'base_time': '0500',  # 05:00 AM 기준
        'nx': nx,
        'ny': ny,
    }

    # 시간대별로 유효한 데이터를 찾기 위한 반복
    valid_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']  # 기상청 단기예보 API 제공 시간
    response_data = None

    for time in valid_times:
        params['base_time'] = time
        response = requests.get(BASE_URL, params=params)
        
        # 응답 상태 코드 확인
        if response.status_code == 200:
            try:
                data = response.json()
                if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
                    response_data = data['response']['body']['items']['item']
                    break  # 유효한 데이터를 찾으면 루프 종료
            except ValueError as e:
                st.error(f"JSON 디코딩 오류: {e}")
                st.text(response.text)
                continue
        else:
            st.error(f"HTTP 오류: {response.status_code}")
            st.text(response.text)
            continue
    
    if response_data:
        df = pd.DataFrame(response_data)
        return df
    else:
        st.error("유효한 데이터를 찾을 수 없습니다.")
        return None

# 오늘 날짜와 1일 전 날짜 계산(기상청에서 최근 3일만 제공)
today = datetime.today()
three_days_ago = today - timedelta(days=1)

# 사이드바에서 날짜 선택
selected_day = st.sidebar.date_input(
    "(5) 오늘의 날짜를 선택하세요:", 
    today, 
    min_value=three_days_ago, 
    max_value=today
).strftime('%Y%m%d')
st.session_state.selected_day = selected_day

# 날짜와 시도의 기상 정보 가져오기
weather_data = weather_info(st.session_state.selected_day, "서울특별시")


# 특정 시간의 날씨 데이터를 필터링하는 함수
def get_weather_value(df, category, time="0600"):
    row = df[(df['category'] == category) & (df['fcstTime'] == time)]
    return row['fcstValue'].values[0] if not row.empty else None

# 특정 시간의 날씨 데이터 추출
temperature = get_weather_value(weather_data, "TMP")
wind_direction = get_weather_value(weather_data, "VEC")
wind_speed = get_weather_value(weather_data, "WSD")
precipitation_prob = get_weather_value(weather_data, "POP")
precipitation_amount = get_weather_value(weather_data, "PCP")
humidity = get_weather_value(weather_data, "REH")
sky_condition = get_weather_value(weather_data, "SKY")
snow_amount = get_weather_value(weather_data, "SNO")
wind_speed_uuu = get_weather_value(weather_data, "UUU")
wind_speed_vvv = get_weather_value(weather_data, "VVV")

# 범주에 따른 강수량 텍스트 변환 함수
def format_precipitation(pcp):
    try:
        pcp = float(pcp)
        if pcp == 0 or pcp == '-' or pcp is None:
            return "강수없음"
        elif 0.1 <= pcp < 1.0:
            return "1.0mm 미만"
        elif 1.0 <= pcp < 30.0:
            return f"{pcp}mm"
        elif 30.0 <= pcp < 50.0:
            return "30.0~50.0mm"
        else:
            return "50.0mm 이상"
    except:
        return "강수없음"

# 신적설 텍스트 변환 함수
def format_snow_amount(sno):
    try:
        sno = float(sno)
        if sno == 0 or sno == '-' or sno is None:
            return "적설없음"
        elif 0.1 <= sno < 1.0:
            return "1.0cm 미만"
        elif 1.0 <= sno < 5.0:
            return f"{sno}cm"
        else:
            return "5.0cm 이상"
    except:
        return "적설없음"

# 하늘 상태 코드값 변환 함수
def format_sky_condition(sky):
    mapping = {1: "맑음", 3: "구름많음", 4: "흐림"}
    return mapping.get(int(sky), "알 수 없음") if sky else "알 수 없음"

# 강수 형태 코드값 변환 함수
def format_precipitation_type(pty):
    mapping = {0: "없음", 1: "비", 2: "비/눈", 3: "눈", 4: "소나기", 5: "빗방울", 6: "빗방울/눈날림", 7: "눈날림"}
    return mapping.get(int(pty), "알 수 없음") if pty else "알 수 없음"

# 풍향 값에 따른 16방위 변환 함수
def wind_direction_to_16point(wind_deg):
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    index = int((wind_deg + 22.5 * 0.5) / 22.5) % 16
    return directions[index]

# 풍속에 따른 바람 강도 텍스트 변환 함수
def wind_speed_category(wind_speed): 
    try:
        wind_speed = float(wind_speed)
        if wind_speed < 4.0:
            return "바람이 약하다"
        elif 4.0 <= wind_speed < 9.0:
            return "바람이 약간 강하다"
        elif 9.0 <= wind_speed < 14.0:
            return "바람이 강하다"
        else:
            return "바람이 매우 강하다"
    except:
        return "알 수 없음"
    
st.sidebar.header("[기상청 단기예보 정보]")
    
# 사용자의 기상 요인(날씨 정보) 수집
weather_input = {
"기온(°C)": st.sidebar.number_input("기온(°C)을 입력하세요.", value=float(temperature) if temperature is not None else 0.0, step=0.1, format="%.1f", key="p1"),
"풍향(deg)": st.sidebar.number_input("풍향(deg)을 입력하세요.", value=float(wind_direction) if wind_direction is not None else 0.0, step=1.0, format="%.1f", key="p2"),
"풍속(m/s)": st.sidebar.number_input("풍속(m/s)을 입력하세요.", value=float(wind_speed) if wind_speed is not None else 0.0, step=0.1, format="%.1f", key="p3"),
"풍속(동서성분) UUU (m/s)": st.sidebar.number_input("풍속(동서성분) UUU (m/s)을 입력하세요.", value=float(wind_speed_uuu) if wind_speed_uuu is not None else 0.0, step=0.1, format="%.1f", key="p4"),
"풍속(남북성분) VVV (m/s)": st.sidebar.number_input("풍속(남북성분) VVV (m/s)을 입력하세요.", value=float(wind_speed_vvv) if wind_speed_vvv is not None else 0.0, step=0.1, format="%.1f", key="p5"),
"강수확률(%)": st.sidebar.number_input("강수확률(%)을 입력하세요.", value=float(precipitation_prob) if precipitation_prob is not None else 0.0, step=1.0, format="%.1f", key="p6"),
"강수형태(코드값)": st.sidebar.selectbox("강수형태를 선택하세요.", options=[0, 1, 2, 3, 5, 6, 7], format_func=format_precipitation_type, key="p7"),
"강수량(범주)": st.sidebar.text_input("강수량(범주)을 입력하세요.", value=format_precipitation(precipitation_amount) if precipitation_amount is not None else "강수없음", key="p8"),
"습도(%)": st.sidebar.number_input("습도(%)를 입력하세요.", value=float(humidity) if humidity is not None else 0.0, step=1.0, format="%.1f", key="p9"),
"1시간 신적설(범주(1 cm))": st.sidebar.text_input("1시간 신적설(범주(1 cm))을 입력하세요.", value=snow_amount if snow_amount is not None else "적설없음", key="p10"),
"하늘상태(코드값)": st.sidebar.selectbox("하늘상태를 선택하세요.", options=[1, 3, 4], format_func=format_sky_condition, key="p11"),
}
st.session_state.weather_input = weather_input



# GPT 프롬프트 엔지니어링 함수
def gpt_prompt(user_input):
    base_prompt = f"""
    너는 영등포구 주민들에게 맞춤형 보고서를 작성해주는 전문적이고 차분한 [영등포구 보고서 작성 프로그램]이야.
    사용자는 영등포구의 특정 행정동에 대한 정보를 필요로 하는 [영등포구 주민]이며, 너의 역할은 사용자가 요청한 정보에 맞춰 정확하고 쉽게 이해할 수 있는 답변을 제공하는 것이야.
    
    <페르소나>
    - 이름: 화재안전이
    - 성격: 친절하고 신뢰할 수 있으며, 명확하고 자세하게 정보를 전달하는 것을 중요하게 생각해.
    - 지식: 영등포구의 지역 특성, 소방 안전, 교통 및 기상 요인에 대한 깊이 있는 이해를 가지고 있어.
    - 목표: 사용자가 영등포구에서 안전하고 편리한 생활을 할 수 있도록 돕는 것이야.

    <역할>
    너는 사용자가 요청한 보고서 타입에 따라 구체적이고 실용적인 정보를 제공해야 해. 
    모든 답변은 사실에 기반해야 하며, 사용자가 쉽게 이해할 수 있도록 설명해야 해.
    
    <규칙>
    1. 사용자가 입력한 정보를 바탕으로 {report_type}에 대한 정확하고 실질적인 솔루션을 제공해.
    2. 영등포구의 행정동 특성과 기상 요인을 고려하여 맞춤형 답변을 작성해.
    3. 예시를 참고해 답변의 구체성을 높이고, 사용자가 이해하기 쉽게 설명해.
    4. 항상 신뢰할 수 있는 정보를 제공하고, 사실에 기반한 답변을 줘야 해.
    5. 답변이 끊기거나 어렵게 느껴질 경우, 친절하게 이전 내용을 이어서 설명해줘.
    
    <예시 1: 영등포구 지구별 취약지역 솔루션>
    [상업지구]
    - 복합쇼핑몰, 시장
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - 예시: 당산동은 복합쇼핑몰이 많아 주말과 휴일에 방문객이 급증합니다. 
      이로 인해 교통 혼잡과 보행자 안전 문제가 발생할 수 있으므로, 교통신호 최적화와 보행자 안전시설 확충이 필요합니다.
    
    [업무지구]
    - 산업단지, 여의동
    - 여의동
    - 예시: 여의동은 업무지구로, 출퇴근 시간대에 교통량이 급증합니다. 
      이는 사고 위험을 높일 수 있으므로, 출퇴근 시간에 교통 분산 대책과 공공교통의 활용을 권장합니다.
    
    [주거지구]
    - 아파트, 주택 단지, 쪽방 지역
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - 예시: 신길동은 주거지구로, 노후된 건축물이 많아 화재에 취약합니다. 
      이에 따라 소방시설의 정비와 노후 건축물의 개선이 필요합니다.

    <예시 2: 영등포구 소방차의 원활한 진입로 확보>
    - 영등포구의 특정 행정동에서 소방차가 원활하게 진입하기 위해 필요한 사항을 상세히 설명해줘.
    - 예시: 대림동은 좁은 골목길이 많고, 불법주정차가 빈번하여 소방차 진입이 어렵습니다. 
      이에 따라 불법주정차 단속 강화와 소방차 전용 진입로 확보가 필요합니다.
    
    사용자 입력: {user_input}
    행정동: {selected_district}
    기상요인: {weather_input}
    보고서 타입: {report_type}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    base_prompt = f"""
    너는 영등포구 주민들에게 맞춤형 보고서를 작성해주는 친절하고 전문적인 [영등포구 보고서 작성 프로그램]이야.
    사용자는 영등포구의 특정 행정동에 대한 정보를 필요로 하는 [영등포구 주민]이며, 너의 역할은 사용자가 원하는 정보를 정확하고 쉽게 이해할 수 있도록 제공하는 것이야.
    
    <페르소나>
    - 이름: 화재안전이
    - 성격: 차분하고 신뢰할 수 있으며, 항상 사용자가 필요한 정보를 명확하고 자세하게 전달해.
    - 지식: 영등포구의 각 지역 특성, 소방 안전, 교통 상황, 기상 요인 등에 대한 깊이 있는 지식을 가지고 있어.
    - 목표: 사용자가 필요한 정보를 정확하고 신속하게 제공함으로써, 영등포구의 안전과 편의를 증진시키는 것이야.

    <역할>
    너는 사용자가 요청한 보고서 타입에 맞춰 상황에 맞는 구체적이고 실용적인 솔루션을 제시해야 해. 
    거짓 정보는 절대 제공하지 않으며, 모든 답변은 사실에 기반해야 해. 
    또한, 사용자가 잘 모르는 상황에도 쉽게 이해할 수 있도록 친절하게 설명해야 해.
    
    <규칙>
    1. 사용자가 입력한 정보에 따라 {report_type}에 맞는 구체적이고 실질적인 솔루션을 제공해.
    2. 영등포구의 각 행정동의 특성과 기상 요인을 반영해, 맞춤형 답변을 제공해.
    3. 예시를 통해 답변의 구체성을 높이고, 사용자가 이해하기 쉽게 작성해.
    4. 거짓 정보를 제공하지 않으며, 사실에 기반한 답변을 줘야 해.
    5. 답변이 끊기거나 이해하기 어렵다면, 어떤 상황에서도 이전 내용을 이어서 설명해줘.
    
    <예시 1: 영등포구 지구별 취약지역 솔루션>
    [상업지구]
    - 복합쇼핑몰, 시장
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - 예시: 당산동은 복합쇼핑몰이 많아 주말과 휴일에 방문객이 급증합니다. 이로 인해 교통 혼잡과 보행자 안전 문제가 발생할 수 있습니다. 
      이에 따라 교통신호 최적화와 보행자 안전시설 확충이 필요합니다.
    
    [업무지구]
    - 산업단지, 여의동
    - 여의동
    - 예시: 여의동은 업무지구로, 출퇴근 시간대에 교통량이 급증합니다. 
      이는 사고 위험을 높일 수 있으므로, 출퇴근 시간에 교통 분산 대책과 공공교통의 활용을 권장합니다.
    
    [주거지구]
    - 아파트, 주택 단지, 쪽방 지역
    - 당산동, 문래동 3가, 영등포동(1,3,4가), 신길동
    - 예시: 신길동은 주거지구로, 노후된 건축물이 많아 화재에 취약합니다. 
      이에 따라 소방시설의 정비와 노후 건축물의 개선이 필요합니다.

    <예시 2: 영등포구 소방차의 원활한 진입로 확보>
    - 영등포구의 특정 행정동에서 소방차가 원활하게 진입하기 위해 필요한 사항을 상세히 설명해줘.
    - 예시: 대림동은 좁은 골목길이 많고, 불법주정차가 빈번하여 소방차 진입이 어렵습니다. 
      이에 따라 불법주정차 단속 강화와 소방차 전용 진입로 확보가 필요합니다.
    
    사용자 입력: {user_input}
    행정동: {selected_district}
    기상요인: {weather_input}
    보고서 타입: {report_type}
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
