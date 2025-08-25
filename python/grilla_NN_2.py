import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import csv
from tqdm import tqdm


n= '1'

# ===========================
# 1. Cargar y preparar datos
# ===========================
path = '/home/pedrorozin/scripts/grilla_results_x11_2.csv'
df = pd.read_csv(path)

# Features y targets
features = df[["A_s", "k h", "h", "Omega_m", "sigma8"]].values
targets = df[["delta_m", "delta_prime_m", "sigma8"]].values

# División aleatoria ANTES de escalar para evitar data leakage
# Usamos un random_state fijo para reproducibilidad
X_train, X_val, y_train, y_val = train_test_split(
    features, targets, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

# Escalamos usando SOLO los datos de entrenamiento para evitar data leakage
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)  # Solo transform, no fit

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)  # Solo transform, no fit

# Convertir a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# Crear datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# ===========================
# 2. Definir la red neuronal
# ===========================
class RegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # salida de 3 targets
        )
        
    def forward(self, x):
        return self.network(x)

model = RegressionNN()

# ===========================
# 3. Función de pérdida y optimizador
# ===========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===========================
# 4. Loop de entrenamiento
# ===========================
epochs = 200
train_losses, val_losses = [], []

for epoch in range(epochs):
    # --- Entrenamiento ---
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # --- Validación ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    # Guardar historial
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ===========================
# 5. Guardar historial y métricas
# ===========================

# Guardar historial en CSV


# Evaluamos métricas finales en validación
model.eval()
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        preds = model(X_batch)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds.numpy())

y_true = np.vstack(y_true_list)
y_pred = np.vstack(y_pred_list)

# Desescalar
y_true_phys = scaler_y.inverse_transform(y_true)
y_pred_phys = scaler_y.inverse_transform(y_pred)

# Métricas por target
mae_targets = mean_absolute_error(y_true_phys, y_pred_phys, multioutput="raw_values")
r2_targets = r2_score(y_true_phys, y_pred_phys, multioutput="raw_values")



# ===========================
# 6. Guardar modelo y escaladores
# ===========================


path_folder = '/home/pedrorozin/scripts/outputs_pedro/neural_networks'
n = '1'
with open(f"{path_folder}/final_metrics_{n}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Target", "MAE", "R2"])
    for name, mae, r2 in zip(["delta_m", "delta_prime_m", "sigma8"], mae_targets, r2_targets):
        writer.writerow([name, mae, r2])

with open(f"{path_folder}/training_history_{n}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE"])
    for epoch, (tr, val) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([epoch, tr, val])
        
torch.save(model.state_dict(), f"{path_folder}/regression_model_{n}.pth")
joblib.dump(scaler_X, f"{path_folder}/scaler_X_{n}.pkl")
joblib.dump(scaler_y, f"{path_folder}/scaler_y_{n}.pkl")

print(f"✅ Modelo guardado en regression_model_{n}.pth")
print(f"✅ Escaladores guardados en scaler_X_{n}.pkl y scaler_y_{n}.pkl")
print(f"✅ Historial en training_history_{n}.csv")
print(f"✅ Métricas finales en final_metrics_{n}.csv")
