import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import joblib
from datetime import datetime

from copy import deepcopy
# TRADITIONAL MODELS UTILS

def kfold_trad_models(models, X, y, n_splits=7):
    results = []
    classes = sorted(y.unique())
    y_bin = label_binarize(y, classes=classes)
    
    for model_name, model in models:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Métricas a recolectar
        f1_scores = []
        bal_acc_scores = []
        acc_scores = []
        roc_auc_scores = []
        fold_times = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_test_bin = label_binarize(y_test, classes=classes)
            
            # Clonar el modelo personalizado
            current_model = deepcopy(model)
            
            # Medir tiempo de entrenamiento y predicción
            start_time = time.time()
            
            current_model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = current_model.predict(X_test)
            
            # Para ROC-AUC necesitamos probabilidades (ajustar SVM)
            try:
                y_prob = current_model.predict_proba(X_test)
            except AttributeError:
                if hasattr(current_model, 'model') and hasattr(current_model.model, 'probability'):
                    current_model.model.probability = True
                    current_model.fit(X_train, y_train)
                    y_prob = current_model.model.predict_proba(X_test)
                else:
                    raise
            
            # Calcular métricas
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            bal_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
            acc_scores.append(accuracy_score(y_test, y_pred))
            
            # ROC-AUC (para multiclase)
            if len(classes) > 2:
                roc_auc_scores.append(roc_auc_score(y_test_bin, y_prob, multi_class='ovr'))
            else:
                roc_auc_scores.append(roc_auc_score(y_test, y_prob[:, 1]))
            
            end_time = time.time()
            fold_times.append(end_time - start_time)
        
        # Guardar resultados para este modelo
        results.append({
            'Modelo': model_name,
            'F1-score': np.mean(f1_scores),
            'Balanced accuracy': np.mean(bal_acc_scores),
            'Accuracy': np.mean(acc_scores),
            'ROC-AUC': np.mean(roc_auc_scores),
            'Tiempo promedio (s)': np.mean(fold_times),
            'Tiempo total (s)': np.sum(fold_times)
        })
    
    return pd.DataFrame(results)

def train_and_test_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name, save_models_path):
    # Medición de tiempo total
    start_time = time.time()
    
    # Entrenamiento con medición de tiempo
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    
    # Predicción con medición de tiempo
    pred_start = time.time()
    preds = model.predict(X_test)
    
    # Cálculo de probabilidades para ROC-AUC
    classes = sorted(y_train.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    
    try:
        probas = model.predict_proba(X_test)
    except AttributeError:
        # Manejo especial para SVM
        if hasattr(model, 'model') and hasattr(model.model, 'probability'):
            model.model.probability = True
            model.fit(X_train, y_train)  # Reentrenar con probability=True
            probas = model.model.predict_proba(X_test)
        else:
            probas = None
    
    pred_end = time.time()
    
    # Cálculo de ROC-AUC
    roc_auc = None
    if probas is not None:
        if len(classes) > 2:
            roc_auc = roc_auc_score(y_test_bin, probas, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, probas[:, 1])
    
    end_time = time.time()
    
    # Guardar modelo
    filename = f"{model_name.lower()}_{dataset_name.lower()}.pkl"
    joblib.dump(model, os.path.join(save_models_path, filename))
    
    return {
        'F1-score': f1_score(y_test, preds, average='weighted'),
        'Balanced accuracy': balanced_accuracy_score(y_test, preds),
        'Accuracy': accuracy_score(y_test, preds),
        'ROC-AUC': roc_auc,
        'Tiempo entrenamiento (s)': train_end - train_start,
        'Tiempo predicción (s)': pred_end - pred_start,
        'Tiempo total (s)': end_time - start_time,
        'Archivo': filename
    }

# DEEP MODELS UTILS

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    #if epoch is not None:
    #    wandb.log({"train_loss": epoch_loss, "epoch": epoch})
    return epoch_loss

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = nn.functional.softmax(outputs, dim=1)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    eval_time = time.time() - start_time
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calcular métricas
    metrics = {
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'bal_acc': balanced_accuracy_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(
            label_binarize(all_labels, classes=np.unique(all_labels)),
            all_probs,
            multi_class='ovr'
        ) if len(np.unique(all_labels)) > 2 else roc_auc_score(all_labels, all_probs[:, 1]),
        'eval_time': eval_time
    }
    
    return metrics, all_preds, all_probs, all_labels



def train_and_test_deepmodel(model, X_train, y_train, X_test, y_test, model_name, dataset_name, save_path, epochs):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preparar DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train.values), torch.LongTensor(y_train.values))
    test_ds = TensorDataset(torch.FloatTensor(X_test.values), torch.LongTensor(y_test.values))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    # Entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_start = time.time()
    for epoch in range(epochs):  # 10 épocas por defecto
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_time = time.time() - train_start
    
    # Evaluación
    eval_start = time.time()
    metrics, _, _, _ = evaluate(model, test_loader, device)
    eval_time = time.time() - eval_start
    
    # Guardar modelo
    filename = f"{model_name.lower()}_features_{dataset_name.lower()}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    
    return {
        'Modelo': model_name,
        'F1-score': metrics['f1'],
        'Balanced accuracy': metrics['bal_acc'],
        'Accuracy': metrics['accuracy'],
        'ROC-AUC': metrics['roc_auc'],
        'Tiempo entrenamiento (s)': train_time,
        'Tiempo predicción (s)': eval_time,
        'Tiempo total (s)': train_time + eval_time,
        'Archivo': filename
    }

def kfold_mlp(X, y, name_model ,model_class, model_kwargs, epochs=10, batch_size=128, n_splits=7, device='cpu'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Resultados a recolectar (formato consistente con modelos clásicos)
    results = {
        'Modelo': name_model,
        'F1-score': [],
        'Balanced accuracy': [],
        'Accuracy': [],
        'ROC-AUC': [],
        'Tiempo fold (s)': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        
        # Inicializar modelo
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Preparar DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx]))
        val_ds = TensorDataset(torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2)
        
        # Entrenamiento
        train_time = 0
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_time += time.time() - epoch_start
            
        # Evaluación
        eval_start = time.time()
        metrics, _, _, _ = evaluate(model, val_loader, device)
        eval_time = time.time() - eval_start
        
        # Calcular tiempo total del fold (entrenamiento + evaluación)
        fold_time = time.time() - fold_start_time
        
        # Guardar resultados
        results['F1-score'].append(metrics['f1'])
        results['Balanced accuracy'].append(metrics['bal_acc'])
        results['Accuracy'].append(metrics['accuracy'])
        results['ROC-AUC'].append(metrics['roc_auc'])
        results['Tiempo fold (s)'].append(fold_time)
    
    # Crear DataFrame en formato consistente con modelos clásicos
    df_results = pd.DataFrame({
        'Modelo': results['Modelo'],
        'F1-score': np.mean(results['F1-score']),
        'Balanced accuracy': np.mean(results['Balanced accuracy']),
        'Accuracy': np.mean(results['Accuracy']),
        'ROC-AUC': np.mean(results['ROC-AUC']),
        'Tiempo promedio (s)': np.mean(results['Tiempo fold (s)']),
        'Tiempo total (s)': np.sum(results['Tiempo fold (s)'])
    }, index=[0])
    
    return df_results

def setup_wandb(project_name="Final-Project-TimeSeries-UTEC", model_type="traditional", name="name model"):
    """Configuración inicial de wandb para cualquier tipo de modelo"""
    wandb.init(
        project=project_name,
        config={
            "model_type": model_type,
            "model_name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
    )

def log_results_to_wandb(results_df, dataset_type, evaluation_type):
    """
    Loggear resultados a wandb de forma consistente para todos los modelos
    
    Args:
        results_df: DataFrame con columnas estandarizadas
        dataset_type: "temporales" o "espectrales"
        evaluation_type: "validación_cruzada" o "evaluación_test"
    """
    required_columns = {
        'Modelo', 'F1-score', 'Balanced accuracy', 'Accuracy', 'ROC-AUC',
        'Tiempo total (s)' #'Tiempo promedio (s)'
    }
    
    # Verificar columnas requeridas
    assert required_columns.issubset(results_df.columns), f"Faltan columnas requeridas: {required_columns - set(results_df.columns)}"
    
    # Loggear cada fila del DataFrame
    for _, row in results_df.iterrows():
        log_data = {
            "Modelo": row["Modelo"],
            "Dataset": dataset_type,
            "Tipo": evaluation_type,
            "F1-score": row["F1-score"],
            "Balanced_accuracy": row["Balanced accuracy"],
            "Accuracy": row["Accuracy"],
            "ROC-AUC": row["ROC-AUC"]
        }
        
        # Añadir tiempos según el tipo de evaluación
        if evaluation_type == "validación_cruzada":
            log_data.update({
                "Tiempo_promedio(s)": row["Tiempo promedio (s)"],
                "Tiempo_total(s)": row["Tiempo total (s)"]
            })
        else:  # evaluación_test
            log_data.update({
                "Tiempo_entrenamiento(s)": row["Tiempo entrenamiento (s)"],
                "Tiempo_predicción(s)": row["Tiempo predicción (s)"],
                "Tiempo_total(s)": row["Tiempo total (s)"]
            })
        
        wandb.log(log_data)
    
    # Loggear tabla resumen
    wandb.log({
        f"Resumen_{evaluation_type}_{dataset_type}": wandb.Table(dataframe=results_df)
    })
    
    
    
def kfold_lstm(X, y, name_model, model_class, model_kwargs, epochs=10, batch_size=128, n_splits=7, device='cpu'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Resultados a recolectar
    results = {
        'Modelo': name_model,
        'F1-score': [],
        'Balanced accuracy': [],
        'Accuracy': [],
        'ROC-AUC': [],
        'Tiempo fold (s)': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        
        # Inicializar modelo LSTM
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Preparar DataLoaders con formato adecuado para LSTM (añadir dimensión secuencial)
        train_ds = TensorDataset(torch.FloatTensor(X[train_idx]).unsqueeze(1),  # Añade dimensión de secuencia
                                torch.LongTensor(y[train_idx]))
        val_ds = TensorDataset(torch.FloatTensor(X[val_idx]).unsqueeze(1),
                              torch.LongTensor(y[val_idx]))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2)
        
        # Entrenamiento
        train_time = 0
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_time += time.time() - epoch_start
            
        # Evaluación
        eval_start = time.time()
        metrics, _, _, _ = evaluate(model, val_loader, device)
        eval_time = time.time() - eval_start
        
        # Calcular tiempo total del fold
        fold_time = time.time() - fold_start_time
        
        # Guardar resultados
        results['F1-score'].append(metrics['f1'])
        results['Balanced accuracy'].append(metrics['bal_acc'])
        results['Accuracy'].append(metrics['accuracy'])
        results['ROC-AUC'].append(metrics['roc_auc'])
        results['Tiempo fold (s)'].append(fold_time)
    
    # Crear DataFrame
    df_results = pd.DataFrame({
        'Modelo': results['Modelo'],
        'F1-score': np.mean(results['F1-score']),
        'Balanced accuracy': np.mean(results['Balanced accuracy']),
        'Accuracy': np.mean(results['Accuracy']),
        'ROC-AUC': np.mean(results['ROC-AUC']),
        'Tiempo promedio (s)': np.mean(results['Tiempo fold (s)']),
        'Tiempo total (s)': np.sum(results['Tiempo fold (s)'])
    }, index=[0])
    
    return df_results


def train_and_test_lstm(model, X_train, y_train, X_test, y_test, model_name, dataset_name, save_path, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preparar DataLoaders con formato adecuado para LSTM
    train_ds = TensorDataset(torch.FloatTensor(X_train.values).unsqueeze(1), 
                            torch.LongTensor(y_train.values))
    test_ds = TensorDataset(torch.FloatTensor(X_test.values).unsqueeze(1),
                           torch.LongTensor(y_test.values))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    # Entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_start = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_time = time.time() - train_start
    
    # Evaluación
    eval_start = time.time()
    metrics, _, _, _ = evaluate(model, test_loader, device)
    eval_time = time.time() - eval_start
    
    # Guardar modelo
    filename = f"{model_name.lower()}_features_{dataset_name.lower()}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    
    return {
        'Modelo': model_name,
        'F1-score': metrics['f1'],
        'Balanced accuracy': metrics['bal_acc'],
        'Accuracy': metrics['accuracy'],
        'ROC-AUC': metrics['roc_auc'],
        'Tiempo entrenamiento (s)': train_time,
        'Tiempo predicción (s)': eval_time,
        'Tiempo total (s)': train_time + eval_time,
        'Archivo': filename
    }
    
def kfold_cnn(X, y, name_model, model_class, model_kwargs, epochs=10, batch_size=128, n_splits=7, device='cpu'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Resultados a recolectar
    results = {
        'Modelo': name_model,
        'F1-score': [],
        'Balanced accuracy': [],
        'Accuracy': [],
        'ROC-AUC': [],
        'Tiempo fold (s)': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        
        # Inicializar modelo CNN
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Preparar DataLoaders con formato adecuado para CNN (añadir dimensión de canal)
        train_ds = TensorDataset(torch.FloatTensor(X[train_idx]).unsqueeze(1),  # Añade dimensión de canal
                               torch.LongTensor(y[train_idx]))
        val_ds = TensorDataset(torch.FloatTensor(X[val_idx]).unsqueeze(1),
                             torch.LongTensor(y[val_idx]))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2)
        
        # Entrenamiento
        train_time = 0
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch_cnn(model, train_loader, criterion, optimizer, device)
            train_time += time.time() - epoch_start
            
        # Evaluación
        eval_start = time.time()
        metrics, _, _, _ = evaluate(model, val_loader, device)
        eval_time = time.time() - eval_start
        
        # Calcular tiempo total del fold
        fold_time = time.time() - fold_start_time
        
        # Guardar resultados
        results['F1-score'].append(metrics['f1'])
        results['Balanced accuracy'].append(metrics['bal_acc'])
        results['Accuracy'].append(metrics['accuracy'])
        results['ROC-AUC'].append(metrics['roc_auc'])
        results['Tiempo fold (s)'].append(fold_time)
    
    # Crear DataFrame
    df_results = pd.DataFrame({
        'Modelo': results['Modelo'],
        'F1-score': np.mean(results['F1-score']),
        'Balanced accuracy': np.mean(results['Balanced accuracy']),
        'Accuracy': np.mean(results['Accuracy']),
        'ROC-AUC': np.mean(results['ROC-AUC']),
        'Tiempo promedio (s)': np.mean(results['Tiempo fold (s)']),
        'Tiempo total (s)': np.sum(results['Tiempo fold (s)'])
    }, index=[0])
    
    return df_results


def train_and_test_cnn(model, X_train, y_train, X_test, y_test, model_name, dataset_name, save_path, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preparar DataLoaders con formato adecuado para CNN
    train_ds = TensorDataset(torch.FloatTensor(X_train.values).unsqueeze(1), 
                           torch.LongTensor(y_train.values))
    test_ds = TensorDataset(torch.FloatTensor(X_test.values).unsqueeze(1),
                          torch.LongTensor(y_test.values))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    # Entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_start = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_time = time.time() - train_start
    
    # Evaluación
    eval_start = time.time()
    metrics, _, _, _ = evaluate(model, test_loader, device)
    eval_time = time.time() - eval_start
    
    # Guardar modelo
    filename = f"{model_name.lower()}_features_{dataset_name.lower()}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    
    return {
        'Modelo': model_name,
        'F1-score': metrics['f1'],
        'Balanced accuracy': metrics['bal_acc'],
        'Accuracy': metrics['accuracy'],
        'ROC-AUC': metrics['roc_auc'],
        'Tiempo entrenamiento (s)': train_time,
        'Tiempo predicción (s)': eval_time,
        'Tiempo total (s)': train_time + eval_time,
        'Archivo': filename
    }


def train_one_epoch_cnn(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Resetear gradientes
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass y optimización
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss