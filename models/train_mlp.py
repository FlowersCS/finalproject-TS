from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score #classification_report
import sys
import os
import joblib
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import kfold_trad_models
from src.models.dl_models import MLPModel
from models.utils import kfold_mlp, train_and_test_deepmodel,setup_wandb,log_results_to_wandb
from copy import deepcopy
import wandb
#wandb.init(project="Final-Project-TimeSeries-UTEC", name="MLP-temp-spec", config={
#    "n_splits": 7
#})
setup_wandb(project_name="Final-Project-TimeSeries-UTEC", model_type="deep", name="MLP-temp-spec")

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

epochs = 10
n_splits = 7

#----------KFOLD-------------------
df_kfold_temp = kfold_mlp(X_temp_train.values, y_temp_train.values, 
                         "MLP",MLPModel, 
                         {'input_size': X_temp_train.shape[1], 'num_classes': len(np.unique(y_temp_train))},
                         epochs=epochs, device='cuda' if torch.cuda.is_available() else 'cpu')

df_kfold_spec = kfold_mlp(X_spec_train.values, y_spec_train.values, 
                         "MLP",MLPModel, 
                         {'input_size': X_spec_train.shape[1], 'num_classes': len(np.unique(y_spec_train))},
                         epochs=epochs, device='cuda' if torch.cuda.is_available() else 'cpu')

print(df_kfold_temp)
print(df_kfold_spec)

csv_cv_temp = os.path.join(save_results_path, 'resultados_cv_temporales.csv')
if os.path.isfile(csv_cv_temp):
    df_cv_temp_old = pd.read_csv(csv_cv_temp)
    df_temp_kfold = pd.concat([df_cv_temp_old, df_kfold_temp], ignore_index=True)
df_temp_kfold.to_csv(csv_cv_temp, index=False)

csv_cv_spec = os.path.join(save_results_path, 'resultados_cv_espectrales.csv')
if os.path.isfile(csv_cv_spec):
    df_cv_spec_old = pd.read_csv(csv_cv_spec)
    df_spec_kfold = pd.concat([df_cv_spec_old, df_kfold_spec], ignore_index=True)
df_spec_kfold.to_csv(csv_cv_spec, index=False)

#/
log_results_to_wandb(df_kfold_temp, "temporales", "validación_cruzada")
log_results_to_wandb(df_kfold_spec, "espectrales", "validación_cruzada")


#---------------Training-test-------------------
dict_test_results_temp = train_and_test_deepmodel(
    model=MLPModel(input_size=X_temp_train.shape[1], num_classes=6),
    X_train=X_temp_train,
    y_train=y_temp_train,
    X_test=X_temp_test,
    y_test=y_temp_test,
    model_name="MLP",
    dataset_name="temporales",
    save_path=save_models_path,
    epochs=epochs
)

dict_test_results_spec = train_and_test_deepmodel(
    model=MLPModel(input_size=X_spec_train.shape[1], num_classes=6),
    X_train=X_spec_train,
    y_train=y_spec_train,
    X_test=X_spec_test,
    y_test=y_spec_test,
    model_name="MLP",
    dataset_name="espectrales",
    save_path=save_models_path,
    epochs=epochs
)

print(dict_test_results_temp)
print(dict_test_results_spec)


csv_test_temp = os.path.join(save_results_path, "resultados_test_temporales.csv")
csv_test_spec = os.path.join(save_results_path, "resultados_test_espectrales.csv")
# Crear DataFrame con los resultados del MLP
#mlp_test_results_temp = {
#    "Modelo": "MLP",
#    "F1-score": dict_test_results_temp["F1-score"],
#    "Balanced accuracy": dict_test_results_temp["Balanced accuracy"],
#    "Accuracy": dict_test_results_temp["Accuracy"],
#    "ROC-AUC": dict_test_results_temp["ROC-AUC"],
#    "Tiempo entrenamiento (s)": dict_test_results_temp["Tiempo entrenamiento (s)"],
#    "Tiempo predicción (s)": dict_test_results_temp["Tiempo predicción (s)"],
#    "Tiempo total (s)": dict_test_results_temp["Tiempo total (s)"],
#    "Archivo": dict_test_results_temp["Archivo"]
#}

#df_test_mlp_temp = pd.DataFrame([dict_test_results_temp])
## eliminar de fg_test_mlp_temp la columna "Tiempo promedio (s)" si existe
#if "Tiempo promedio (s)" in df_test_mlp_temp.columns:
#    df_test_mlp_temp.drop(columns=["Tiempo promedio (s)"], inplace=True)

df_test_mlp_temp = pd.DataFrame([dict_test_results_temp])
# Leer archivo existente o crear nuevo DataFrame
if os.path.isfile(csv_test_temp):
    df_test_temp = pd.read_csv(csv_test_temp)
    # Eliminar entrada previa de MLP si existe (para evitar duplicados)
    df_test_temp = df_test_temp[df_test_temp['Modelo'] != 'MLP']
else:
    df_test_temp = pd.DataFrame()

# Concatenar los nuevos resultados
df_test_temp = pd.concat([df_test_temp, df_test_mlp_temp], ignore_index=True)

# Guardar el archivo actualizado
df_test_temp.to_csv(csv_test_temp, index=False)

log_results_to_wandb(df_test_mlp_temp, "temporales", "evaluación_test")



df_test_mlp_spec = pd.DataFrame([dict_test_results_spec])
# Leer archivo existente o crear nuevo DataFrame
if os.path.isfile(csv_test_spec):
    df_test_spec = pd.read_csv(csv_test_spec)
    # Eliminar entrada previa de MLP si existe (para evitar duplicados)
    df_test_spec = df_test_spec[df_test_spec['Modelo'] != 'MLP']
else:
    df_test_spec = pd.DataFrame()

# Concatenar los nuevos resultados
df_test_spec = pd.concat([df_test_spec, df_test_mlp_spec], ignore_index=True)

# Guardar el archivo actualizado
df_test_spec.to_csv(csv_test_spec, index=False)

log_results_to_wandb(df_test_mlp_spec, "espectrales", "evaluación_test")

wandb.finish()