
# Install tensorflow 
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

# Libraries
import streamlit as st
import tensorflow as tf 
import pandas as pd
import pickle

model = tf.keras.models.load_model("model.h5", compile=False)

with open("geo_encoder.pkl", "rb") as file:
    label_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)


st.title("Customer Churn Prediction")

# user input

geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
# st.write(label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
# st.write(gender)
age = st.slider("Age", 18, 95)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 1, 10)
number_of_product = st.slider("Number of Product", 1, 4)
has_cr_card = st.selectbox("Has credit card", [0,1])
is_active_member = st.selectbox("Is active number", [0,1])

#Prepare input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_product],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoder= label_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoder, columns=label_encoder_geo.get_feature_names_out(["Geography"]))

# st.write(geo_encoded_df)
# st.write(input_data)

input_data_df = pd.concat([input_data.reset_index(drop= True), geo_encoded_df], axis=1)

# st.write(input_data_df)

scaled_input_df = scalar.transform(input_data_df)

prediction = model.predict(scaled_input_df)

prediction_probability = prediction[0][0]

st.write(f"Churn Probability is : {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write(f"Customer is likey to churn.\nWhich percentage is {prediction_probability:.2%}")

else:
    st.write(f"Customer is not likely to churn\nWhich percentage is {prediction_probability:.2%}")

