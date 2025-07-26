# import tensorflow as tf
# import keras
# from keras.models import load_model

# print("TensorFlow version:", tf.__version__)
# print("Keras version:", keras.__version__)


# import pickle
# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
# model=load_model('model.h5')
# with open('onehot_encode_geo.pkl','rb') as file:
#     onehot_encoder_geo=pickle.load(file)
# with open('lable_encoder_gender.pkl','rb') as file:
#     label_encoder_gender=pickle.load(file)
# with open('scaler.pkl','rb') as file:
#     scaler=pickle.load(file)
# # streamlit app
# st.title('Customer churn prediction')
# geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
# gender=st.selectbox('Gender',label_encoder_gender.classes_)
# age=st.slider('Age',18,92)
# balance=st.number_input('Balance')
# credit_score=st.number_input('Credit Score')
# estimated_salary=st.number_input('Estimated Salary')
# tenure=st.slider('Tenure',0,10)
# num_of_products=st.slider('Num of Products',1,4)
# has_cr_card=st.selectbox('Has Credit Card',[0,1])
# is_active_member=st.selectbox('Is Active Member',[0,1])

# input_data={
#     'CreditScore':[credit_score],
#     'Gender':[label_encoder_gender.transform([gender][0])],
#     'Age':[age],
#     'Tenure':[tenure],
#     'Balance':[balance],
#     'NumOfProducts':[num_of_products],
#     'HasCrCard':[has_cr_card],
#     'IsActiveMember':[is_active_member],
#     'EstimatedSalary':[estimated_salary]
# }
# geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
# geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# input_data=pd.concat([input_data.reset_index(drop-True),geo_encoded_df],axis=1)
# input_data_scaled=scaler.transform(input_data)
# prediction=model.predict(input_data_scaled)
# prediction_proba=prediction[0][0]

# if prediction[0][0]>0.5:
#     print('customer is likely to churn')
# else:
#     print('customer is not likely to churn')
import tensorflow as tf
from tensorflow.keras.models import load_model

       # ✅ should work too


import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



# Load model and encoders
model = load_model('model.h5')
with open('onehot_encode_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('lable_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Num of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Data processing
input_data = {
    'CreditScore': [credit_score],
    'Gender': label_encoder_gender.transform([gender]),
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

input_df = pd.DataFrame(input_data)
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(final_input)
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba > 0.5:
    st.error(f"❌ Customer is likely to churn. (Confidence: {prediction_proba:.2f})")
else:
    st.success(f"✅ Customer is not likely to churn. (Confidence: {prediction_proba:.2f})")
