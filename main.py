import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import datetime
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

geo_label_encoder = LabelEncoder()
data['Gender'] = geo_label_encoder.fit_transform(data['Gender'])

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
one = OneHotEncoder(sparse_output=False)

geo_encoder = one.fit_transform(data[['Geography']])

df = pd.DataFrame(geo_encoder, columns=one.get_feature_names_out(['Geography']))

full_df = pd.concat([data.drop(['Geography'],axis = 1) ,df], axis=1)


import pickle

with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(geo_label_encoder, f)
    
with open('one_geo.pkl', 'wb') as f:
    pickle.dump(one, f)
    


from sklearn.model_selection import train_test_split
X = full_df.drop(['Exited'], axis=1)
Y = full_df['Exited']


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

print(X_train,X_test)

with open('scalr.pkl','wb') as f:
    pickle.dump(scalar, f)
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

model = Sequential([
    Dense(64, activation='relu', input_dim = X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation = 'sigmoid')
])

print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    restore_best_weights = True
)

history = model.fit(X_train,Y_train, validation_data=(X_test, Y_test),callbacks = [tensorflow_callback, early_stopping_callback], epochs=100)

model.save('model.h5')

# To visualize training metrics in TensorBoard:
# 1. Open terminal (View -> Terminal in VS Code)
# here logs/fit willl be generated as you run this file(for visualization)
# 2. Run: tensorboard --logdir=logs/fit
# 3. Open browser and go to: http://localhost:6006













