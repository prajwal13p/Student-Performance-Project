import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import pickle

@st.cache_resource(show_spinner="loading the model...")
def load_model():
    with open('/modelle.pkl','rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le


def preprocessing_input(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transform = scaler.transform(df)
    return df_transform

@st.cache_resource(show_spinner="loading the model...")
def predict_data(data):
    model, scalar, le = load_model()
    processed_data = preprocessing_input(data, scalar, le)
    pred = model.predict(processed_data)
    return pred

def main():
    st.title("Student Performance Prediction")
    st.write("Enter the data")
    hours = st.number_input("Hours Stuied",min_value=1, max_value=10, value=5)
    pre_score = st.number_input("Previous Score", min_value=40, max_value=100,value=70)
    extra_act = st.selectbox("extra curri activity", ['Yes','No'])
    sleep_hr = st.number_input("sleepling Hours", min_value=4, max_value=10, value=7)
    num_of_papers = st.number_input('number of question paper solved', min_value=0, max_value=10, value=7)

    btn = st.button("Predict")

    if btn :
        user_data = {
            'Hours Studied': hours,
            'Previous Scores': pre_score,
            'Extracurricular Activities': extra_act,
            'Sleep Hours':sleep_hr,
            'Sample Question Papers Practiced':num_of_papers
        }
        pred = predict_data(user_data)
        st.success(f"Student predicted Performance Index {round(pred[0][0],2)}")

if __name__ == '__main__':
    main()

