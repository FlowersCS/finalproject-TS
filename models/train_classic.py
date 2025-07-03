# entrenamiento de random forest, KNN y SVM con el codigo de classic_models.py
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score, balanced_accuracy_score
import sys
import os
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.classic_models import RandomForestModel, KNNModel, SVMModel

project_root = os.path.dirname(os.getcwd())
data_path = os.path.join(project_root,'finalprojectTS', 'data')
processed_path = data_path + '/processed'

#features_temporal = pd.read_csv(processed_path + '/features_temporales_labelNum.csv')
#features_espectrales = pd.read_csv(processed_path + '/features_espectrales_labelNum.csv')
features_temporal = pd.read_csv(processed_path + '/features_temporales_labelNum_overlap50.csv')
features_espectrales = pd.read_csv(processed_path + '/features_espectrales_labelNum_overlap50.csv')
# los labels estan en features_temporal en la ultima columna
X_temp = features_temporal.iloc[:, :-1]  # Todas las columnas excepto la última
y_temp = features_temporal.iloc[:, -1]   # Solo la última columna

X_spec = features_espectrales.iloc[:, :-1]  # Todas las columnas excepto la última
y_spec = features_espectrales.iloc[:, -1]   # Solo la última columna

# Dividir el dataset en entrenamiento y prueba
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.2, random_state=42)

models = [
    ("RandomForest", RandomForestModel(n_estimators=100, random_state=42)),
    ("KNN", KNNModel(n_neighbors=6)),
    ("SVM", SVMModel(kernel='rbf', C=1.0))
]

datasets = [
    ("Features_temporales", X_temp, y_temp),
    ("Features_espectrales", X_spec, y_temp)
]

save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
save_results_path = os.path.join(project_root, 'finalprojectTS', 'models', 'results')
os.makedirs(save_models_path, exist_ok=True)

# Separación de kfold y training con testing
def crossval_models(models, X, y, n_splits=7):
    results = []
    for model_name, model in models:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = ['f1_weighted', 'balanced_accuracy', 'accuracy']
        cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)
        results.append({
            'Modelo': model_name,
            'F1-score': np.mean(cv_results['test_f1_weighted']),
            'Balanced accuracy': np.mean(cv_results['test_balanced_accuracy']),
            'Accuracy': np.mean(cv_results['test_accuracy'])
        })
    return pd.DataFrame(results)

cross_validation_temp = crossval_models(models, X_temp_train, y_temp_train)
cross_validation_spec = crossval_models(models, X_spec_train, y_spec_train)

print("Resultados validación cruzada - Features Temporales")
print(cross_validation_temp)
print("\nResultados validación cruzada - Features Espectrales")
print(cross_validation_spec)

def train_and_test_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    filename = f"{model_name.lower()}_{dataset_name.lower()}.pkl"
    joblib.dump(model, os.path.join(save_models_path, filename))
    return {
        'F1-score': f1_score(y_test, preds, average='weighted'),
        'Balanced accuracy': balanced_accuracy_score(y_test, preds),
        'Accuracy': accuracy_score(y_test, preds),
        'Classification report': classification_report(y_test, preds),
        'Archivo': filename
    }
    
test_results_temp = []
test_results_spec = []
    
for model_name, model in models:
    if model_name in ['RandomForest', 'KNN', 'SVM']:
        # Temporales
        res_temp = train_and_test_model(model, X_temp_train, y_temp_train, X_temp_test, y_temp_test, model_name, "features_temporales")
        test_results_temp.append({'Modelo': model_name, **{k: v for k, v in res_temp.items() if k != 'Classification report'}})
        #print(f"{model_name} - Test Results (Temporales):", res_temp)
        # Espectrales
        res_spec = train_and_test_model(model, X_spec_train, y_spec_train, X_spec_test, y_spec_test, model_name, "features_espectrales")
        test_results_spec.append({'Modelo': model_name, **{k: v for k, v in res_spec.items() if k != 'Classification report'}})
        #print(f"{model_name} - Test Results (Espectrales):", res_spec)

df_test_temp = pd.DataFrame(test_results_temp)
df_test_spec = pd.DataFrame(test_results_spec)

df_test_temp.to_csv(os.path.join(save_results_path, 'resultados_test_temporales.csv'), index=False)
df_test_spec.to_csv(os.path.join(save_results_path, 'resultados_test_espectrales.csv'), index=False)

cross_validation_temp.to_csv(os.path.join(save_results_path, 'resultados_cv_temporales.csv'), index=False)
cross_validation_spec.to_csv(os.path.join(save_results_path, 'resultados_cv_espectrales.csv'), index=False)
