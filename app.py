import streamlit as st
import joblib
import pandas as pd
import sklearn

st.title('Insurance Prediction')

model = joblib.load('reg_model.joblib')
enc = joblib.load('encoder.joblib')
le_sex = joblib.load('le_sex.joblib')
le_smk = joblib.load('le_smk.joblib')

age = st.number_input('Enter age', min_value=18)

sex = st.selectbox('Enter sex', ['male', 'female'])

bmi = st.number_input('Enter BMI', min_value=0)

children = st.number_input('Enter number of children', min_value=0)

smoker = st.selectbox('Smoker?', ['yes', 'no'])

region = st.selectbox('Enter region', ['northeast', 'northwest', 'southeast', 'southwest'])

data = {"age":age,
        "sex":sex,
        "bmi":bmi,
        "children":children,
        "smoker":smoker,
        "region":region}

df = pd.DataFrame(data, index=[0])

one_hot = enc.transform(df[['region']]).toarray()
df[["northeast","northwest","southeast","southwest"]] = one_hot
df['sex'] = le_sex.transform(df[['sex']])
df['smoker'] = le_smk.transform(df[['smoker']])
df = df.drop(columns='region')

button = st.button('Predict')

if button == True:
    prediction = model.predict(df.head(1))

    if prediction < 0:
        st.success(f'${0:0,.2f}')    
    st.success(f'${prediction[0]:0,.2f}')
