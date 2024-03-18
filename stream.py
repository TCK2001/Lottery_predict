import pandas as pd
import json
from urllib.request import urlopen
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np

start = 1
end = 1112
st.title("Lottery Prediction App :money_with_wings:")

def load_lotto_data(draw_no):
    url = f"http://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    result_data = urlopen(url)
    result = result_data.read()
    data = json.loads(result)
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.transpose()
    return df

selected_draw = st.selectbox("Select a draw", range(start, end))

if selected_draw:
    df_selected_draw = load_lotto_data(selected_draw)
    winning_numbers = df_selected_draw[['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo']].iloc[0]
    winning_prize = [int(prize) for prize in df_selected_draw[['firstWinamnt']].iloc[0]]
    st.subheader(f'Selected :blue[{selected_draw}] :sunglasses:')
    st.write("Winning numbers:")
    st.code(f"""{list(winning_numbers)}""")
    st.write("Winning Prize:")
    st.code(f"{winning_prize} KRW", language='text')

def regression_analysis_per_number(sorted_win_number):
    models = []
    X = np.arange(1, len(sorted_win_number) + 1).reshape(-1, 1)

    for number_position in range(6):
        y = np.array([numbers[number_position] for numbers in sorted_win_number])
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        models.append(model)

    return models


df = pd.read_csv("lotto_1111.csv")

df = df[["drwtNo1", "drwtNo2", "drwtNo3", "drwtNo4", "drwtNo5", "drwtNo6"]]
win_numbers = df.apply(lambda x:list(x), axis=1)
sorted_win_number = win_numbers.apply(lambda x:sorted(x))

# 각 번호별 모델 학습
models_per_number = regression_analysis_per_number(sorted_win_number)

if st.button("Predict the next winning numbers"):
    prediction = []

    for i, model in enumerate(models_per_number):
        y = model.predict(np.array([end]).reshape(-1, 1))
        prediction.append(y.item())
        
    st.write(f"{end} Winning numbers:")
    st.code(f"""{list(prediction)}""")
    st.code(f"""Ceiling : {np.ceil(prediction).astype(int)}""")
    st.code(f"""Round : {np.ceil(prediction).astype(int)}""")
    st.code(f"""Floor : {np.floor(prediction).astype(int)}""")