# entrenamiento de random forest, KNN y SVM con el codigo de classic_models.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.classic_models import RandomForestModel, KNNModel, SVMModel

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

# Definir los modelos y sus nombres
#models = [
#    ("RandomForest", RandomForestModel(n_estimators=100, random_state=42)),
#    ("KNN", KNNModel(n_neighbors=6)),
#    ("SVM", SVMModel(kernel='rbf', C=1.0))
#]
#
## Definir los conjuntos de datos y sus nombres
#datasets = [
#    ("Features_Temporales", X_temp_train, X_temp_test, y_temp_train, y_temp_test),
#    ("Features_Espectrales", X_spec_train, X_spec_test, y_spec_train, y_spec_test)
#]
#
#save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
#
#for model_name, model in models:
#    for data_name, X_train, X_test, y_train, y_test in datasets:
#        model.fit(X_train, y_train)
#        predictions = model.predict(X_test)
#        print(f"{model_name} Model - {data_name}")
#        print("Accuracy:", accuracy_score(y_test, predictions))
#        print(classification_report(y_test, predictions))
#        # Guardar el modelo
#        filename = f"{model_name.lower()}_{data_name.lower()}.pkl"
#        joblib.dump(model, os.path.join(save_models_path, filename))
#        print("\n" + "="*50 + "\n")


## Entrenamiento y evaluación de Random Forest para features temporales
#rf_temp_model = RandomForestModel(n_estimators=100, random_state=42)
#rf_temp_model.fit(X_temp_train, y_temp_train)
#rf_temp_predictions = rf_temp_model.predict(X_temp_test)
#
#print("Random Forest Model - Features Temporales")
#print("Accuracy:", accuracy_score(y_temp_test, rf_temp_predictions))
#print(classification_report(y_temp_test, rf_temp_predictions))
#
## guardar el modelo de Random Forest para features temporales
#save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
#joblib.dump(rf_temp_model, os.path.join(save_models_path, 'rf_temp_model.pkl'))
#
#
#print("\n" + "="*50 + "\n")
#
## Entrenamiento y evaluación de Random Forest para features espectrales
#rf_spec_model = RandomForestModel(n_estimators=100, random_state=42)
#rf_spec_model.fit(X_spec_train, y_spec_train)
#rf_spec_predictions = rf_spec_model.predict(X_spec_test)
#
#print("Random Forest Model - Features Espectrales")
#print("Accuracy:", accuracy_score(y_spec_test, rf_spec_predictions))
#print(classification_report(y_spec_test, rf_spec_predictions))
#
## guardar el modelo de Random Forest para features espectrales
#joblib.dump(rf_spec_model, os.path.join(save_models_path, 'rf_spec_model.pkl'))


## Entrenamiento y evaluación de KNN para features temporales
#knn_temp_model = KNNModel(n_neighbors=6)
#knn_temp_model.fit(X_temp_train, y_temp_train)
#knn_temp_predictions = knn_temp_model.predict(X_temp_test)
#
#print("KNN Model - Features Temporales")
#print("Accuracy:", accuracy_score(y_temp_test, knn_temp_predictions))
#print(classification_report(y_temp_test, knn_temp_predictions))
#
## guardar el modelo de KNN para features temporales
#save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
#joblib.dump(knn_temp_model, os.path.join(save_models_path, 'knn_temp_model.pkl'))
#
#print("\n" + "="*50 + "\n")
#
## Entrenamiento y evaluación de KNN para features espectrales
#knn_spec_model = KNNModel(n_neighbors=6)
#knn_spec_model.fit(X_spec_train, y_spec_train)
#knn_spec_predictions = knn_spec_model.predict(X_spec_test)
#
#print("KNN Model - Features Espectrales")
#print("Accuracy:", accuracy_score(y_spec_test, knn_spec_predictions))
#print(classification_report(y_spec_test, knn_spec_predictions))
#
## guardar el modelo de KNN para features espectrales
#joblib.dump(knn_spec_model, os.path.join(save_models_path, 'knn_spec_model.pkl'))


# PROBLEMA SVM, DESBALANCE DE CLASES
# Entrenamiento y evaluación de SVM para features temporales
svm_temp_model = SVMModel(kernel='rbf', C=1.0)
svm_temp_model.fit(X_temp_train, y_temp_train)
svm_temp_predictions = svm_temp_model.predict(X_temp_test)

print("SVM Model - Features Temporales")
print("Accuracy:", accuracy_score(y_temp_test, svm_temp_predictions))
print(classification_report(y_temp_test, svm_temp_predictions))

# guardar el modelo de SVM para features temporales
save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
joblib.dump(svm_temp_model, os.path.join(save_models_path, 'svm_temp_model.pkl'))

print("\n" + "="*50 + "\n")

# Entrenamiento y evaluación de SVM para features espectrales
svm_spec_model = SVMModel(kernel='rbf', C=1.0)
svm_spec_model.fit(X_spec_train, y_spec_train)
svm_spec_predictions = svm_spec_model.predict(X_spec_test)

print("SVM Model - Features Espectrales")
print("Accuracy:", accuracy_score(y_spec_test, svm_spec_predictions))
print(classification_report(y_spec_test, svm_spec_predictions))

# guardar el modelo de SVM para features espectrales
joblib.dump(svm_spec_model, os.path.join(save_models_path, 'svm_spec_model.pkl'))
