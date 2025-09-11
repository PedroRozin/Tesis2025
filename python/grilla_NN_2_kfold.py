import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import joblib
import csv
from tqdm import tqdm
from funciones_tesis import RegressionNN
import os
import matplotlib.pyplot as plt

path_folder = '/home/pedrorozin/scripts/outputs_pedro/neural_networks/'
n = 'red_kfold_cv'

if os.path.exists(path_folder + n):
    raise FileExistsError(f"El directorio {path_folder}/{n} ya existe.")

if not os.path.exists(path_folder + n):
    os.makedirs(path_folder + n)

# ===========================
# 1. Load data 
# ===========================
path_grilla = '/home/pedrorozin/scripts/outputs_pedro/grillas/sin_As/grilla_results_no_As.csv'
df = pd.read_csv(path_grilla)

# Features y targets
features = df[["a", "k h", "h", "Omega_m"]].values
targets = df[["delta_m", "delta_prime_m"]].values

print(f"📊 Dataset: {features.shape[0]} samples, {features.shape[1]} features, {targets.shape[1]} targets")

# ===========================
# 2. Configuración del entrenamiento
# ===========================
# Early stopping class
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Learning rate scheduler
def get_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=20, 
        factor=0.5, 
        min_lr=1e-6
    )

# ===========================
# 3. K-Fold Cross Validation
# ===========================
def train_fold(X_train, X_val, y_train, y_val, fold_num):
    """Entrena un fold específico"""
    
    # Escalamos usando SOLO los datos de entrenamiento
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    # datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Modelo
    model = RegressionNN()
    
    # Loss function y optimizador con weight decay (regularización L2)
    criterion = nn.MSELoss()
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer)
    early_stopping = EarlyStopping(patience=50, min_delta=1e-6)

    # Training
    max_epochs = 400
    train_losses, val_losses = [], []
    
    if fold_num >= 0:
        print(f"\n🔄 Entrenando Fold {fold_num + 1}...")
    else:
        print(f"\n🎯 Entrenando modelo final...")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_dataset)

        # Guardar historial
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"⏹️  Early stopping en epoch {epoch + 1}")
            break

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # Evaluación final
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds.numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Desescalar para métricas
    y_true_phys = scaler_y.inverse_transform(y_true)
    y_pred_phys = scaler_y.inverse_transform(y_pred)

    # Métricas
    mae_targets = mean_absolute_error(y_true_phys, y_pred_phys, multioutput="raw_values")
    r2_targets = r2_score(y_true_phys, y_pred_phys, multioutput="raw_values")
    
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'mae_targets': mae_targets,
        'r2_targets': r2_targets,
        'epochs_trained': len(train_losses)
    }

# K-Fold Cross Validation
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []
all_maes = []
all_r2s = []

print(f"🔬 Iniciando {k_folds}-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
    X_train_fold = features[train_idx]
    X_val_fold = features[val_idx]
    y_train_fold = targets[train_idx]
    y_val_fold = targets[val_idx]
    
    result = train_fold(X_train_fold, X_val_fold, y_train_fold, y_val_fold, fold)
    fold_results.append(result)
    all_maes.append(result['mae_targets'])
    all_r2s.append(result['r2_targets'])
    
    print(f"✅ Fold {fold + 1} completado:")
    print(f"   📉 Val Loss: {result['final_val_loss']:.6f}")
    print(f"   📊 MAE delta_m: {result['mae_targets'][0]:.6f}")
    print(f"   📊 R² delta_m: {result['r2_targets'][0]:.6f}")

# ===========================
# 4. Resultados del Cross Validation
# ===========================
all_maes = np.array(all_maes)
all_r2s = np.array(all_r2s)

mean_maes = np.mean(all_maes, axis=0)
std_maes = np.std(all_maes, axis=0)
mean_r2s = np.mean(all_r2s, axis=0)
std_r2s = np.std(all_r2s, axis=0)

print(f"\n📈 RESULTADOS K-FOLD CROSS VALIDATION:")
print(f"=" * 50)
target_names = ["delta_m", "delta_prime_m"]
for i, target in enumerate(target_names):
    print(f"{target}:")
    print(f"  MAE: {mean_maes[i]:.6f} ± {std_maes[i]:.6f}")
    print(f"  R²:  {mean_r2s[i]:.6f} ± {std_r2s[i]:.6f}")

# Encontrar el mejor fold (menor validation loss)
best_fold_idx = np.argmin([result['final_val_loss'] for result in fold_results])
best_model_result = fold_results[best_fold_idx]

print(f"\n🏆 Mejor modelo: Fold {best_fold_idx + 1}")
print(f"   Val Loss: {best_model_result['final_val_loss']:.6f}")

# ===========================
# 5. Entrenar modelo final con todos los datos
# ===========================
print(f"\n🎯 Entrenando modelo final con todos los datos...")

# División simple para el modelo final
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    features, targets, 
    test_size=0.2, 
    random_state=100, 
    shuffle=True
)

final_result = train_fold(X_train_final, X_val_final, y_train_final, y_val_final, -1)

print(f"✅ Modelo final entrenado:")
print(f"   📉 Val Loss: {final_result['final_val_loss']:.6f}")
print(f"   📊 MAE delta_m: {final_result['mae_targets'][0]:.6f}")
print(f"   📊 R² delta_m: {final_result['r2_targets'][0]:.6f}")

# ===========================
# 6. Guardar resultados
# ===========================

# Guardar resultados del cross validation
with open(f"{path_folder}/{n}/cross_validation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Val_Loss", "MAE_delta_m", "MAE_delta_prime_m", "R2_delta_m", "R2_delta_prime_m", "Epochs"])
    for i, result in enumerate(fold_results):
        writer.writerow([
            i + 1, 
            result['final_val_loss'],
            result['mae_targets'][0],
            result['mae_targets'][1],
            result['r2_targets'][0],
            result['r2_targets'][1],
            result['epochs_trained']
        ])
    
    # Agregar resumen estadístico
    writer.writerow([])
    writer.writerow(["Estadísticas", "", "", "", "", "", ""])
    writer.writerow(["Mean", np.mean([r['final_val_loss'] for r in fold_results]), 
                    mean_maes[0], mean_maes[1], mean_r2s[0], mean_r2s[1], ""])
    writer.writerow(["Std", np.std([r['final_val_loss'] for r in fold_results]), 
                    std_maes[0], std_maes[1], std_r2s[0], std_r2s[1], ""])

# Guardar el mejor modelo
torch.save(best_model_result['model'].state_dict(), f"{path_folder}/{n}/best_model_fold_{best_fold_idx + 1}.pth")
joblib.dump(best_model_result['scaler_X'], f"{path_folder}/{n}/best_scaler_X_fold_{best_fold_idx + 1}.pkl")
joblib.dump(best_model_result['scaler_y'], f"{path_folder}/{n}/best_scaler_y_fold_{best_fold_idx + 1}.pkl")

# Guardar modelo final
torch.save(final_result['model'].state_dict(), f"{path_folder}/{n}/final_model.pth")
joblib.dump(final_result['scaler_X'], f"{path_folder}/{n}/final_scaler_X.pkl")
joblib.dump(final_result['scaler_y'], f"{path_folder}/{n}/final_scaler_y.pkl")

# Guardar historial del modelo final
with open(f"{path_folder}/{n}/final_training_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE"])
    for epoch, (tr, val) in enumerate(zip(final_result['train_losses'], final_result['val_losses']), 1):
        writer.writerow([epoch, tr, val])

# Guardar información detallada
with open(f"{path_folder}/{n}/detailed_info.txt", "w") as f:
    f.write("K-FOLD CROSS VALIDATION RESULTS\n")
    f.write("=" * 40 + "\n\n")
    
    f.write(f"Dataset: {features.shape[0]} samples\n")
    f.write(f"Features: {features.shape[1]} (a, k h, h, Omega_m)\n")
    f.write(f"Targets: {targets.shape[1]} (delta_m, delta_prime_m)\n\n")
    
    f.write(f"K-Fold Configuration:\n")
    f.write(f"  - Number of folds: {k_folds}\n")
    f.write(f"  - Shuffle: True\n")
    f.write(f"  - Random state: 42\n\n")
    
    f.write(f"Training Configuration:\n")
    f.write(f"  - Max epochs: 400\n")
    f.write(f"  - Batch size: 128\n")
    f.write(f"  - Learning rate: 5e-4\n")
    f.write(f"  - Optimizer: Adam with weight_decay=1e-4\n")
    f.write(f"  - Early stopping: patience=50, min_delta=1e-6\n")
    f.write(f"  - LR scheduler: ReduceLROnPlateau\n\n")
    
    f.write("Cross Validation Results:\n")
    for i, target in enumerate(target_names):
        f.write(f"  {target}:\n")
        f.write(f"    MAE: {mean_maes[i]:.6f} ± {std_maes[i]:.6f}\n")
        f.write(f"    R²:  {mean_r2s[i]:.6f} ± {std_r2s[i]:.6f}\n")
    
    f.write(f"\nBest Fold: {best_fold_idx + 1}\n")
    f.write(f"Best Val Loss: {best_model_result['final_val_loss']:.6f}\n\n")
    
    f.write(f"Final Model (trained on 80% of data):\n")
    f.write(f"  Val Loss: {final_result['final_val_loss']:.6f}\n")
    f.write(f"  MAE delta_m: {final_result['mae_targets'][0]:.6f}\n")
    f.write(f"  R² delta_m: {final_result['r2_targets'][0]:.6f}\n")
    f.write(f"  Epochs trained: {final_result['epochs_trained']}\n")

# Crear gráfico de comparación
plt.figure(figsize=(15, 5))

# Subplot 1: Validation losses por fold
plt.subplot(1, 3, 1)
val_losses_all_folds = [result['final_val_loss'] for result in fold_results]
plt.bar(range(1, k_folds + 1), val_losses_all_folds)
plt.xlabel('Fold')
plt.ylabel('Validation Loss')
plt.title('Validation Loss por Fold')
plt.xticks(range(1, k_folds + 1))

# Subplot 2: MAE comparison
plt.subplot(1, 3, 2)
x = np.arange(len(target_names))
width = 0.35
plt.bar(x - width/2, mean_maes, width, yerr=std_maes, label='MAE', capsize=5)
plt.xlabel('Target')
plt.ylabel('MAE')
plt.title('MAE por Target (Cross Validation)')
plt.xticks(x, target_names)
plt.legend()

# Subplot 3: R² comparison
plt.subplot(1, 3, 3)
plt.bar(x - width/2, mean_r2s, width, yerr=std_r2s, label='R²', capsize=5)
plt.xlabel('Target')
plt.ylabel('R²')
plt.title('R² por Target (Cross Validation)')
plt.xticks(x, target_names)
plt.legend()

plt.tight_layout()
plt.savefig(f"{path_folder}/{n}/cross_validation_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
print(f"📁 Resultados guardados en: {path_folder}/{n}/")
print(f"✅ Cross validation results: cross_validation_results.csv")
print(f"✅ Mejor modelo: best_model_fold_{best_fold_idx + 1}.pth")
print(f"✅ Modelo final: final_model.pth")
print(f"✅ Información detallada: detailed_info.txt")
print(f"✅ Gráfico comparativo: cross_validation_comparison.png")
