import numpy as np
import pickle
import pandas as pd
# from flasggar import swagger
import streamlit as st


# app = Flask(__name__)
# Swagger(app)

pickle_in = open("model_svc_pickle","rb")
classifier = pickle.load(pickle_in)

# @app.route('/')
#def welcome():
# return "Welcome all"

# app.route('/predict',methods=["Get"])
def diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
  prediction = classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
  print(prediction)
  return prediction

def main():
  st.title('Diabetes Predictor')
  html_temp = """
  <div style="background-color:#546beb; padding:10px">
  <h2 style="color:#93f50a;text-align:center;">Streamlit Diabetes Predictor </h2>
  </div>
  """

  st.markdown(html_temp, unsafe_allow_html=True)
  #years = st.text_input("Years")  #years = st.text_input("Years","Type Here")
  Pregnancies = st.text_input("Pregnancies")
  Glucose = st.text_input("Glucose")
  BloodPressure = st.text_input("BloodPressure")
  SkinThickness = st.text_input("SkinThickness")
  Insulin = st.text_input("Insulin")
  BMI = st.text_input("BMI")
  DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
  Age = st.text_input("Age")

  result = ""
  if st.button("Predict"):
    result = diabetes_prediction(float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age))
  st.success('The output is {}'.format(result))
  if st.button("About"):
    st.text("Lets Learn")
    st.text("Built with Streamlit")


#if __name__ == '__main__':
main()
