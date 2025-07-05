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
from src.models.dl_models import LSTMModel
from models.utils import kfold_lstm, train_and_test_lstm, setup_wandb, log_results_to_wandb
from copy import deepcopy
import wandb
setup_wandb(project_name="Final-Project-TimeSeries-UTEC", model_type="deep", name="LSTM-temp-spec")


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
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp,stratify=y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, stratify=y_spec, test_size=0.2, random_state=42)

epochs = 10
n_splits = 7

#----------KFOLD-------------------
df_kfold_temp = kfold_lstm(X_temp_train.values, y_temp_train.values, 
                         "LSTM",LSTMModel, 
                         {'input_size': X_temp_train.shape[1], 'num_classes': len(np.unique(y_temp_train))},
                         epochs=epochs, device='cuda' if torch.cuda.is_available() else 'cpu')

df_kfold_spec = kfold_lstm(X_spec_train.values, y_spec_train.values, 
                         "LSTM",LSTMModel, 
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

log_results_to_wandb(df_kfold_temp, "temporales", "validación_cruzada")
log_results_to_wandb(df_kfold_spec, "espectrales", "validación_cruzada")

#---------------Training-test-------------------
dict_test_results_temp = train_and_test_lstm(
    model=LSTMModel(input_size=X_temp_train.shape[1], num_classes=6),
    X_train=X_temp_train,
    y_train=y_temp_train,
    X_test=X_temp_test,
    y_test=y_temp_test,
    model_name="LSTM",
    dataset_name="temporales",
    save_path=save_models_path,
    epochs=epochs
)

dict_test_results_spec = train_and_test_lstm(
    model=LSTMModel(input_size=X_spec_train.shape[1], num_classes=6),
    X_train=X_spec_train,
    y_train=y_spec_train,
    X_test=X_spec_test,
    y_test=y_spec_test,
    model_name="LSTM",
    dataset_name="espectrales",
    save_path=save_models_path,
    epochs=epochs
)

print(dict_test_results_temp)
print(dict_test_results_spec)


csv_test_temp = os.path.join(save_results_path, "resultados_test_temporales.csv")
csv_test_spec = os.path.join(save_results_path, "resultados_test_espectrales.csv")

df_test_lstm_temp = pd.DataFrame([dict_test_results_temp])
# Leer archivo existente o crear nuevo DataFrame
if os.path.isfile(csv_test_temp):
    df_test_temp = pd.read_csv(csv_test_temp)
    # Eliminar entrada previa de LSTM si existe (para evitar duplicados)
    df_test_temp = df_test_temp[df_test_temp['Modelo'] != 'LSTM']
else:
    df_test_temp = pd.DataFrame()

# Concatenar los nuevos resultados
df_test_temp = pd.concat([df_test_temp, df_test_lstm_temp], ignore_index=True)

# Guardar el archivo actualizado
df_test_temp.to_csv(csv_test_temp, index=False)

log_results_to_wandb(df_test_lstm_temp, "temporales", "evaluación_test")



df_test_lstm_spec = pd.DataFrame([dict_test_results_spec])
# Leer archivo existente o crear nuevo DataFrame
if os.path.isfile(csv_test_spec):
    df_test_spec = pd.read_csv(csv_test_spec)
    # Eliminar entrada previa de LSTM si existe (para evitar duplicados)
    df_test_spec = df_test_spec[df_test_spec['Modelo'] != 'LSTM']
else:
    df_test_spec = pd.DataFrame()

# Concatenar los nuevos resultados
df_test_spec = pd.concat([df_test_spec, df_test_lstm_spec], ignore_index=True)

# Guardar el archivo actualizado
df_test_spec.to_csv(csv_test_spec, index=False)

log_results_to_wandb(df_test_lstm_spec, "espectrales", "evaluación_test")


wandb.finish()




'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = os.path.dirname(os.getcwd())
data_path = os.path.join(project_root,'finalprojectTS', 'data')
processed_path = data_path + '/processed'

save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')
save_results_path = os.path.join(project_root, 'finalprojectTS', 'models', 'results')
os.makedirs(save_models_path, exist_ok=True)
os.makedirs(save_results_path, exist_ok=True)

features_temporal = pd.read_csv(processed_path + '/features_temporales_labelNum_overlap50.csv')
features_espectrales = pd.read_csv(processed_path + '/features_espectrales_labelNum_overlap50.csv')

# los labels estan en features_temporal en la ultima columna
X_temp = features_temporal.iloc[:, 1:-1]  # Todas las columnas excepto la primera y la última
y_temp = features_temporal.iloc[:, -1]   # Solo la última columna

X_spec = features_espectrales.iloc[:, 1:-1]  # Todas las columnas excepto la última
y_spec = features_espectrales.iloc[:, -1]   # Solo la última columna

# Dividir el dataset en entrenamiento y prueba
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.2, random_state=42)

# Convierte a numpy y reshape para LSTM: (batch, seq_len, input_size)
X_temp_train_np = X_temp_train.to_numpy().reshape(-1, 1, X_temp_train.shape[1])
X_temp_test_np  = X_temp_test.to_numpy().reshape(-1, 1, X_temp_test.shape[1])
y_temp_train_np = y_temp_train.to_numpy()
y_temp_test_np  = y_temp_test.to_numpy()

X_spec_train_np = X_spec_train.to_numpy().reshape(-1, 1, X_spec_train.shape[1])
X_spec_test_np  = X_spec_test.to_numpy().reshape(-1, 1, X_spec_test.shape[1])
y_spec_train_np = y_spec_train.to_numpy()
y_spec_test_np  = y_spec_test.to_numpy()


print("KFold en train LSTM temporales")
f1_scores_temp, bal_acc_temp, acc_temp = kfold(
    X_temp_train_np, y_temp_train_np,
    model_class=LSTMModel,
    model_kwargs={'input_size': X_temp_train.shape[1], 'num_classes': len(np.unique(y_temp_train))},
    epochs=epochs,
    batch_size=batch_size,
    n_splits=7,
    device=device
)

print("KFold en train LSTM espectrales")
f1_scores_spec, bal_acc_spec, acc_spec = kfold(
    X_spec_train_np, y_spec_train_np,
    model_class=LSTMModel,
    model_kwargs={'input_size': X_spec_train.shape[1], 'num_classes': len(np.unique(y_spec_train))},
    epochs=epochs,
    batch_size=batch_size,
    n_splits=7,
    device=device
)

# Guardar los resultados de K-Fold en csv (temporales)
csv_path_cvTemp = os.path.join(save_results_path,'resultados_cv_temporales.csv')
df_cvTemp = pd.read_csv(csv_path_cvTemp)

df_temp = pd.concat([df_cvTemp, pd.DataFrame([{
    "Modelo": "LSTM",
    "F1-score": np.mean(f1_scores_temp),
    "Balanced accuracy": np.mean(bal_acc_temp),
    "Accuracy": np.mean(acc_temp)
}])], ignore_index=True)

df_temp.to_csv(csv_path_cvTemp, index=False)

# Guardar los resultados de K-Fold en csv (espectrales)
csv_path_cvSpec = os.path.join(save_results_path,'resultados_cv_espectrales.csv')
df_cvSpec = pd.read_csv(csv_path_cvSpec)

df_spec = pd.concat([df_cvSpec, pd.DataFrame([{
    "Modelo": "LSTM",
    "F1-score": np.mean(f1_scores_spec),
    "Balanced accuracy": np.mean(bal_acc_spec),
    "Accuracy": np.mean(acc_spec)
}])], ignore_index=True)

df_spec.to_csv(csv_path_cvSpec, index=False)


# --- Entrenamiento final y evaluación en test (temporales)---
print("\nEntrenamiento final y evaluación en test LSTM (temporales):")
train_ds = TensorDataset(torch.FloatTensor(X_temp_train_np), torch.LongTensor(y_temp_train_np))
test_ds  = TensorDataset(torch.FloatTensor(X_temp_test_np),  torch.LongTensor(y_temp_test_np))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size*2)

model = LSTMModel(input_size=X_temp_train.shape[1], num_classes=len(np.unique(y_temp_train))).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
acc = accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
print("Test Accuracy:", acc)
wandb.log({"test_f1": f1, "test_bal_acc": bal_acc, "test_acc": acc})

# Guardar el modelo entrenado
model_save_path = os.path.join(save_models_path, "lstm_features_temporales.pkl")
torch.save(model.state_dict(), model_save_path)

csv_test_temp = os.path.join(save_results_path, "resultados_test_temporales.csv")
df_test_temp = pd.read_csv(csv_test_temp)
df_test_temp = pd.concat([df_test_temp, pd.DataFrame([{
    "Modelo": "LSTM",
    "F1-score": f1,
    "Balanced accuracy": bal_acc,
    "Accuracy": acc,
    "Archivo": "lstm_features_temporales.pkl"
}])], ignore_index=True)
df_test_temp.to_csv(csv_test_temp, index=False)

# --- Entrenamiento final y evaluación en test (espectrales) ---
print("\nEntrenamiento final y evaluación en test LSTM (espectrales):")
train_ds = TensorDataset(torch.FloatTensor(X_spec_train_np), torch.LongTensor(y_spec_train_np))
test_ds  = TensorDataset(torch.FloatTensor(X_spec_test_np),  torch.LongTensor(y_spec_test_np))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size*2)

model = LSTMModel(input_size=X_spec_train.shape[1], num_classes=len(np.unique(y_spec_train))).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
acc = accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
print("Test Accuracy:", acc)
wandb.log({"test_f1": f1, "test_balanced_accuracy": bal_acc, "test_accuracy": acc})

# Guardar el modelo entrenado
model_save_path = os.path.join(save_models_path, "lstm_features_espectrales.pkl")
torch.save(model.state_dict(), model_save_path)

csv_test_spec = os.path.join(save_results_path, "resultados_test_espectrales.csv")
df_test_spec = pd.read_csv(csv_test_spec)
df_test_spec = pd.concat([df_test_spec, pd.DataFrame([{
    "Modelo": "LSTM",
    "F1-score": f1,
    "Balanced accuracy": bal_acc,
    "Accuracy": acc,
    "Archivo": "lstm_features_espectrales.pkl"
}])], ignore_index=True)
df_test_spec.to_csv(csv_test_spec, index=False)

wandb.finish()

'''