# entrenamiento de MLP, LSTM, CNN
# guarda modelos serializados
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dl_models import MLPModel, LSTMModel, CNNModel

project_root = os.path.dirname(os.getcwd())
data_path = os.path.join(project_root,'finalprojectTS', 'data')
processed_path = data_path + '/processed'

features_temporal = pd.read_csv(processed_path + '/features_temporales.csv')
features_espectrales = pd.read_csv(processed_path + '/features_spectral.csv')
# los labels estan en features_temporal en la ultima columna
X_temp = features_temporal.iloc[:, :-1]  # Todas las columnas excepto la última
y_temp = features_temporal.iloc[:, -1]   # Solo la última columna

X_spec = features_espectrales.iloc[:, :-1]  # Todas las columnas excepto la última
y_spec = features_espectrales.iloc[:, -1]   # Solo la última columna

# Dividir el dataset en entrenamiento y prueba
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.2, random_state=42)

if 'Unnamed: 0' in X_temp_train.columns:
    X_temp_train = X_temp_train.drop(columns=['Unnamed: 0'])
    X_temp_test = X_temp_test.drop(columns=['Unnamed: 0'])

if 'Unnamed: 0' in X_spec_train.columns:
    X_spec_train = X_spec_train.drop(columns=['Unnamed: 0'])
    X_spec_test = X_spec_test.drop(columns=['Unnamed: 0'])
    
X_temp_train = X_temp_train.astype('float32')
X_temp_test = X_temp_test.astype('float32')
y_temp_train = y_temp_train.astype('category').cat.codes
y_temp_test = y_temp_test.astype('category').cat.codes

X_spec_train = X_spec_train.astype('float32')
X_spec_test = X_spec_test.astype('float32')
y_spec_train = y_spec_train.astype('category').cat.codes
y_spec_test = y_spec_test.astype('category').cat.codes    



## training and evaluation of MLP
#mlp_temp_model = MLPModel(input_shape=(X_temp_train.shape[1],), num_classes=len(y_temp.unique()))
#mlp_temp_model.fit(X_temp_train, y_temp_train, epochs=10, batch_size=32)
#mlp_temp_predictions = mlp_temp_model.predict(X_temp_test)
#print("MLP Model - Features Temporales")
#print("Accuracy:", (mlp_temp_predictions.argmax(axis=1) == y_temp_test).mean())
## Save the MLP model
save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
#joblib.dump(mlp_temp_model.model, os.path.join(save_models_path, 'mlp_features_temporales.pkl'))
#print("\n" + "="*50 + "\n")
#
## training and evaluation of MLP with spectral features
#mlp_spec_model = MLPModel(input_shape=(X_spec_train.shape[1],), num_classes=len(y_spec.unique()))
#mlp_spec_model.fit(X_spec_train, y_spec_train, epochs=10, batch_size=32)
#mlp_spec_predictions = mlp_spec_model.predict(X_spec_test)
#print("MLP Model - Features Espectrales")
#print("Accuracy:", (mlp_spec_predictions.argmax(axis=1) == y_spec_test).mean())
## Save the MLP model
#joblib.dump(mlp_spec_model.model, os.path.join(save_models_path, 'mlp_features_espectrales.pkl'))


##training and cross-validation of LSTM
#lstm_temp_model = LSTMModel(input_shape=(X_temp_train.shape[1], 1), num_classes=len(y_temp.unique()))
#lstm_temp_model.fit(X_temp_train.values.reshape(-1, X_temp_train.shape[1], 1), y_temp_train, epochs=10, batch_size=32)
#lstm_temp_predictions = lstm_temp_model.predict(X_temp_test.values.reshape(-1, X_temp_test.shape[1], 1))
#print("LSTM Model - Features Temporales")
#print("Accuracy:", (lstm_temp_predictions.argmax(axis=1) == y_temp_test).mean())
## Save the LSTM model
#joblib.dump(lstm_temp_model.model, os.path.join(save_models_path, 'lstm_features_temporales.pkl'))
## training and cross-validation of LSTM with spectral features
#lstm_spec_model = LSTMModel(input_shape=(X_spec_train.shape[1], 1), num_classes=len(y_spec.unique()))
#lstm_spec_model.fit(X_spec_train.values.reshape(-1, X_spec_train.shape[1], 1), y_spec_train, epochs=10, batch_size=32)
#lstm_spec_predictions = lstm_spec_model.predict(X_spec_test.values.reshape(-1, X_spec_test.shape[1], 1))
#print("LSTM Model - Features Espectrales")
#print("Accuracy:", (lstm_spec_predictions.argmax(axis=1) == y_spec_test).mean())
## Save the LSTM model
#joblib.dump(lstm_spec_model.model, os.path.join(save_models_path, 'lstm_features_espectrales.pkl'))


#training and cross-validation of CNN
cnn_temp_model = CNNModel(input_shape=(X_temp_train.shape[1], 1), num_classes=len(y_temp.unique()))
cnn_temp_model.fit(X_temp_train.values.reshape(-1, X_temp_train.shape[1], 1), y_temp_train, epochs=10, batch_size=32)
cnn_temp_predictions = cnn_temp_model.predict(X_temp_test.values.reshape(-1, X_temp_test.shape[1], 1))
print("CNN Model - Features Temporales")
print("Accuracy:", (cnn_temp_predictions.argmax(axis=1) == y_temp_test).mean())
# Save the CNN model
joblib.dump(cnn_temp_model.model, os.path.join(save_models_path, 'cnn_features_temporales.pkl'))
# training and cross-validation of CNN with spectral features
cnn_spec_model = CNNModel(input_shape=(X_spec_train.shape[1], 1), num_classes=len(y_spec.unique()))
cnn_spec_model.fit(X_spec_train.values.reshape(-1, X_spec_train.shape[1], 1), y_spec_train, epochs=10, batch_size=32)
cnn_spec_predictions = cnn_spec_model.predict(X_spec_test.values.reshape(-1, X_spec_test.shape[1], 1))
print("CNN Model - Features Espectrales")
print("Accuracy:", (cnn_spec_predictions.argmax(axis=1) == y_spec_test).mean())
# Save the CNN model
joblib.dump(cnn_spec_model.model, os.path.join(save_models_path, 'cnn_features_espectrales.pkl'))
