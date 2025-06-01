import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf  
import pickle

model = tf.keras.models.load_model('model.h5')

# Load the model and encoders
@st.cache_resource
def load_models():
    
    with open('gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    with open('one_geo.pkl', 'rb') as f:
        geo_encoder = pickle.load(f)
    with open('scalr.pkl', 'rb') as f:
        scalar = pickle.load(f)
    return model, gender_encoder, geo_encoder, scalar

model, gender_encoder, geo_encoder, scalar = load_models()

# Set up the Streamlit page
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict if they are likely to leave the bank.')

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
        geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=18, max_value=100, value=40)
        tenure = st.number_input('Tenure (years)', min_value=0, max_value=20, value=3)
        
    with col2:
        balance = st.number_input('Balance', min_value=0.0, value=600.0)
        num_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
        has_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
        is_active = st.selectbox('Is Active Member?', ['Yes', 'No'])
        salary = st.number_input('Estimated Salary', min_value=0.0, value=500.0)
    
    submitted = st.form_submit_button("Predict Churn")

# Make prediction when form is submitted
if submitted:
    # Prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender.lower(),
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active == 'Yes' else 0,
        'EstimatedSalary': salary
    }
    
    # Transform the data
    geo_encoded = geo_encoder.transform([[input_data['Geography']]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
    
    input_df = pd.DataFrame([input_data])
    input_df = pd.concat([input_df.drop(['Geography'], axis=1), geo_encoded_df], axis=1)
    input_df['Gender'] = gender_encoder.fit_transform([input_data['Gender']])
    
    # Scale the features
    input_scaled = scalar.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]
    
    # Show results
    st.write('---')
    st.subheader('Prediction Results')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")
    with col2:
        result = "Customer is likely to leave" if probability > 0.5 else "Customer is likely to stay"
        # Fix the conditional display
        if probability <= 0.5:
            st.success(result)
        else:
            st.error(result)
    
    # Add a gauge chart
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig)
