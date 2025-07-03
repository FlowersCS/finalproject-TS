from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score #classification_report
import sys
import os
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import kfold_trad_models

import wandb
wandb.init(project="Final-Project-TimeSeries-UTEC", name="SVM-temp-spec", config={
    "n_splits": 7
})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.classic_models import SVMModel

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
    ("SVM-rbf-1-scale", SVMModel(kernel='rbf', C=1.0, gamma='scale')),
    ("SVM-linear-1-scale", SVMModel(kernel='linear', C=1.0, gamma='scale')),
    ("SVM-poly-1-scale", SVMModel(kernel='poly', C=1.0, gamma='scale'))
]

datasets = [
    ("Features_temporales", X_temp, y_temp),
    ("Features_espectrales", X_spec, y_spec)
]

save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
save_results_path = os.path.join(project_root, 'finalprojectTS', 'models', 'results')
os.makedirs(save_models_path, exist_ok=True)
os.makedirs(save_results_path, exist_ok=True)

# crear el archivo csv de resultados cv, test tanto para temporal como espectral
results_cv = pd.DataFrame(columns=["Model", "Dataset", "F1_Score", "Balanced_Accuracy", "Accuracy"])
results_test = pd.DataFrame(columns=["Model", "Dataset", "F1_Score", "Balanced_Accuracy", "Accuracy"])

# guardarlo en la carpeta results solo sino existe el archivo csv
if not os.path.isfile(os.path.join(save_results_path, 'resultados_cv_temporales.csv')):
    results_cv.to_csv(os.path.join(save_results_path, 'resultados_cv_temporales.csv'), index=False)
if not os.path.isfile(os.path.join(save_results_path, 'resultados_test_temporales.csv')):
    results_test.to_csv(os.path.join(save_results_path, 'resultados_test_temporales.csv'), index=False)
    

# K-fold
kfold_temp = kfold_trad_models(models, X_temp_train, y_temp_train)
kfold_spec = kfold_trad_models(models, X_spec_train, y_spec_train)

print("Resultados validación cruzada - Features Temporales")
print(kfold_temp)
print("\nResultados validación cruzada - Features Espectrales")
print(kfold_spec)

# Guardar resultados de KFold temporales
csv_cv_temp = os.path.join(save_results_path, 'resultados_cv_temporales.csv')
if os.path.isfile(csv_cv_temp):
    df_cv_temp = pd.read_csv(csv_cv_temp)
    kfold_temp = pd.concat([df_cv_temp, kfold_temp], ignore_index=True)
kfold_temp.to_csv(csv_cv_temp, index=False)

# Guardar resultados de KFold espectrales
csv_cv_spec = os.path.join(save_results_path, 'resultados_cv_espectrales.csv')
if os.path.isfile(csv_cv_spec):
    df_cv_spec = pd.read_csv(csv_cv_spec)
    kfold_spec = pd.concat([df_cv_spec, kfold_spec], ignore_index=True)
kfold_spec.to_csv(csv_cv_spec, index=False)




def train_and_test_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    filename = f"{model_name.lower()}_{dataset_name.lower()}.pkl"
    joblib.dump(model, os.path.join(save_models_path, filename))
    return {
        'F1-score': f1_score(y_test, preds, average='weighted'),
        'Balanced accuracy': balanced_accuracy_score(y_test, preds),
        'Accuracy': accuracy_score(y_test, preds),
        #'Classification report': classification_report(y_test, preds),
        'Archivo': filename
    }
    
test_results_temp = []
test_results_spec = []

for model_name, model in models:
    # Temporales
    res_temp = train_and_test_model(model, X_temp_train, y_temp_train, X_temp_test, y_temp_test, model_name, "features_temporales")
    test_results_temp.append({'Modelo': model_name, **{k: v for k, v in res_temp.items() if k != 'Classification report'}})
    print(f"{model_name} - Test Results (Temporales):", res_temp)
    # Espectrales
    res_spec = train_and_test_model(model, X_spec_train, y_spec_train, X_spec_test, y_spec_test, model_name, "features_espectrales")
    test_results_spec.append({'Modelo': model_name, **{k: v for k, v in res_spec.items() if k != 'Classification report'}})
    print(f"{model_name} - Test Results (Espectrales):", res_spec)
    
df_test_temp = pd.DataFrame(test_results_temp)
df_test_spec = pd.DataFrame(test_results_spec)

# Guardar resultados de test temporales
csv_test_temp = os.path.join(save_results_path, 'resultados_test_temporales.csv')
if os.path.isfile(csv_test_temp):
    df_test_temp_old = pd.read_csv(csv_test_temp)
    df_test_temp = pd.concat([df_test_temp_old, df_test_temp], ignore_index=True)
df_test_temp.to_csv(csv_test_temp, index=False)

# Guardar resultados de test espectrales
csv_test_spec = os.path.join(save_results_path, 'resultados_test_espectrales.csv')
if os.path.isfile(csv_test_spec):
    df_test_spec_old = pd.read_csv(csv_test_spec)
    df_test_spec = pd.concat([df_test_spec_old, df_test_spec], ignore_index=True)
df_test_spec.to_csv(csv_test_spec, index=False)




# Log de KFold (después de crossval_models)
for idx, row in kfold_temp.iterrows():
    wandb.log({
        "model": row["Modelo"] + "_temp",
        "cv_f1": row["F1-score"],
        "cv_balanced_accuracy": row["Balanced accuracy"],
        "cv_accuracy": row["Accuracy"],
        "dataset": "temporales"
    })

for idx, row in kfold_spec.iterrows():
    wandb.log({
        "model": row["Modelo"] + "_spec",
        "cv_f1": row["F1-score"],
        "cv_balanced_accuracy": row["Balanced accuracy"],
        "cv_accuracy": row["Accuracy"],
        "dataset": "espectrales"
    })

# Log de test (después de test_results_temp y test_results_spec)
for idx, row in df_test_temp.iterrows():
    wandb.log({
        "model": row["Modelo"] + "_temp",
        "test_f1": row["F1-score"],
        "test_balanced_accuracy": row["Balanced accuracy"],
        "test_accuracy": row["Accuracy"],
        "dataset": "temporales"
    })

for idx, row in df_test_spec.iterrows():
    wandb.log({
        "model": row["Modelo"] + "_spec",
        "test_f1": row["F1-score"],
        "test_balanced_accuracy": row["Balanced accuracy"],
        "test_accuracy": row["Accuracy"],
        "dataset": "espectrales"
    })