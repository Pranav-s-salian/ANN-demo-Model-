# This file is for testing the model with sample data
# Do not use in production environment

from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler

model = load_model('model.h5')

with open('gender_encoder.pkl', 'rb') as f:
    geo_label_encoder = pickle.load(f)
    
with open('one_geo.pkl', 'rb') as f:
    one = pickle.load(f)    
    
with open('scalr.pkl', 'rb') as f:
    scalar = pickle.load(f)
    
    
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender':'male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 600,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 500
}

geo_encoder = one.transform([[input_data['Geography']]])
geo_encode_df = pd.DataFrame(geo_encoder, columns=one.get_feature_names_out(['Geography']))

input_data_df = pd.DataFrame([input_data])
input_data_df = pd.concat([input_data_df.drop(['Geography'], axis=1), geo_encode_df], axis=1)
input_data_df['Gender'] = geo_label_encoder.fit_transform([input_data['Gender']])

input_scalar = scalar.transform(input_data_df)  
print(input_scalar)


predict = model.predict(input_scalar)
print(f"Prediction: {predict[0][0]}")
if predict[0][0] > 0.5:
    print("Customer is likely to exit.")
else:
    print("Customer is likely to stay.")
