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
from src.models.classic_models import KNNModel#, SVMModel
from models.utils import train_and_test_model
from copy import deepcopy
import wandb
wandb.init(project="Final-Project-TimeSeries-UTEC", name="KNN-temp-spec", config={
    "n_splits": 7
})

project_root = os.path.dirname(os.getcwd())
data_path = os.path.join(project_root,'finalprojectTS', 'data')
processed_path = data_path + '/processed'

save_models_path = os.path.join(project_root, 'finalprojectTS','models', 'save_models')
save_results_path = os.path.join(project_root, 'finalprojectTS','models', 'results')

features_temporal = pd.read_csv(processed_path + '/features_temporales_labelNum_overlap50.csv')
features_espectrales = pd.read_csv(processed_path + '/features_espectrales_labelNum_overlap50.csv')

X_temp = features_temporal.iloc[:, 1:-1] 
y_temp = features_temporal.iloc[:, -1]  
X_spec = features_espectrales.iloc[:, 1:-1] 
y_spec = features_espectrales.iloc[:, -1]  

# Dividir el dataset en entrenamiento y prueba
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.2, random_state=42)

#----------KFOLD-------------------
knn_model = ("KNN", KNNModel(n_neighbors=6))
models = [knn_model]

kfold_temp = kfold_trad_models([knn_model], X_temp_train, y_temp_train)
kfold_spec = kfold_trad_models([knn_model], X_spec_train, y_spec_train)

print(kfold_temp)
print(kfold_spec)

# Guardar resultados de KFold temporales
csv_cv_temp = os.path.join(save_results_path, 'resultados_cv_temporales.csv')
if os.path.isfile(csv_cv_temp):
    df_cv_temp_old = pd.read_csv(csv_cv_temp)
    kfold_temp = pd.concat([df_cv_temp_old, kfold_temp], ignore_index=True)
kfold_temp.to_csv(csv_cv_temp, index=False)

# Guardar resultados de KFold espectrales
csv_cv_spec = os.path.join(save_results_path, 'resultados_cv_espectrales.csv')
if os.path.isfile(csv_cv_spec):
    df_cv_spec_old = pd.read_csv(csv_cv_spec)
    kfold_spec = pd.concat([df_cv_spec_old, kfold_spec], ignore_index=True)
kfold_spec.to_csv(csv_cv_spec, index=False)

#---------Traning-test-------------------
test_results_temp = []
test_results_spec = []

for model_name, model in models:
    # Clonar el modelo para cada ejecución
    current_model = deepcopy(model)
    
    # Temporales
    res_temp = train_and_test_model(current_model, X_temp_train, y_temp_train, X_temp_test, y_temp_test, 
                                  model_name, "features_temporales", save_models_path)
    test_results_temp.append({'Modelo': model_name, **res_temp})
    print(f"{model_name} - Test Results (Temporales):")
    print(pd.Series({k: v for k, v in res_temp.items() if k != 'Archivo'}))
    
    # Espectrales
    current_model = deepcopy(model)  # Nuevo clone para espectrales
    res_spec = train_and_test_model(current_model, X_spec_train, y_spec_train, X_spec_test, y_spec_test, 
                                  model_name, "features_espectrales", save_models_path)
    test_results_spec.append({'Modelo': model_name, **res_spec})
    print(f"\n{model_name} - Test Results (Espectrales):")
    print(pd.Series({k: v for k, v in res_spec.items() if k != 'Archivo'}))
    print("\n" + "="*80 + "\n")
    
# Convertir a DataFrames
df_test_temp = pd.DataFrame(test_results_temp)
df_test_spec = pd.DataFrame(test_results_spec)

# Reordenar columnas para mejor presentación
column_order = ['Modelo', 'F1-score', 'Balanced accuracy', 'Accuracy', 'ROC-AUC',
                'Tiempo entrenamiento (s)', 'Tiempo predicción (s)', 'Tiempo total (s)', 'Archivo']

df_test_temp = df_test_temp[column_order]
df_test_spec = df_test_spec[column_order]
print(df_test_temp)
print(df_test_spec)

# Guardar resultados de KFold temporales
csv_test_temp = os.path.join(save_results_path, 'resultados_test_temporales.csv')
if os.path.isfile(csv_test_temp):
    df_test_temp_old = pd.read_csv(csv_test_temp)
    df_test_temp = pd.concat([df_test_temp_old, df_test_temp], ignore_index=True)
df_test_temp.to_csv(csv_test_temp, index=False)

# Guardar resultados de KFold espectrales
csv_test_spec = os.path.join(save_results_path, 'resultados_test_espectrales.csv')
if os.path.isfile(csv_test_spec):
    df_test_spec_old = pd.read_csv(csv_test_spec)
    df_test_spec = pd.concat([df_test_spec_old, df_test_spec], ignore_index=True)
df_test_spec.to_csv(csv_test_spec, index=False)


#----------------WADBN----------------------------------

# 1. Log de KFold (después de crossval_models)
for idx, row in kfold_temp.iterrows():
    wandb.log({
        "Modelo": row["Modelo"],
        "Dataset": "temporales",
        "Tipo": "validación_cruzada",
        "F1-score": row["F1-score"],
        "Balanced_accuracy": row["Balanced accuracy"],
        "Accuracy": row["Accuracy"],
        "ROC-AUC": row["ROC-AUC"],
        "Tiempo_promedio(s)": row["Tiempo promedio (s)"],
        "Tiempo_total(s)": row["Tiempo total (s)"]
    })

for idx, row in kfold_spec.iterrows():
    wandb.log({
        "Modelo": row["Modelo"],
        "Dataset": "espectrales",
        "Tipo": "validación_cruzada",
        "F1-score": row["F1-score"],
        "Balanced_accuracy": row["Balanced accuracy"],
        "Accuracy": row["Accuracy"],
        "ROC-AUC": row["ROC-AUC"],
        "Tiempo_promedio(s)": row["Tiempo promedio (s)"],
        "Tiempo_total(s)": row["Tiempo total (s)"]
    })

# 2. Log de test (después de test_results_temp y test_results_spec)
for idx, row in df_test_temp.iterrows():
    wandb.log({
        "Modelo": row["Modelo"],
        "Dataset": "temporales",
        "Tipo": "evaluación_test",
        "F1-score": row["F1-score"],
        "Balanced_accuracy": row["Balanced accuracy"],
        "Accuracy": row["Accuracy"],
        "ROC-AUC": row["ROC-AUC"],
        "Tiempo_entrenamiento(s)": row["Tiempo entrenamiento (s)"],
        "Tiempo_predicción(s)": row["Tiempo predicción (s)"],
        "Tiempo_total(s)": row["Tiempo total (s)"]
    })

for idx, row in df_test_spec.iterrows():
    wandb.log({
        "Modelo": row["Modelo"],
        "Dataset": "espectrales",
        "Tipo": "evaluación_test",
        "F1-score": row["F1-score"],
        "Balanced_accuracy": row["Balanced accuracy"],
        "Accuracy": row["Accuracy"],
        "ROC-AUC": row["ROC-AUC"],
        "Tiempo_entrenamiento(s)": row["Tiempo entrenamiento (s)"],
        "Tiempo_predicción(s)": row["Tiempo predicción (s)"],
        "Tiempo_total(s)": row["Tiempo total (s)"]
    })
    
    
# Opcional: Crear tablas resumen
wandb.log({
    "Resumen_KFold_Temporales": wandb.Table(dataframe=kfold_temp),
    "Resumen_KFold_Espectrales": wandb.Table(dataframe=kfold_spec),
    "Resumen_Test_Temporales": wandb.Table(dataframe=df_test_temp),
    "Resumen_Test_Espectrales": wandb.Table(dataframe=df_test_spec),
    #"Curva_ROC": wandb.plot.roc_curve(y_true, y_probs, labels=class_names)
})

wandb.finish()