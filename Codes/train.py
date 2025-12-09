"""
train.py

Este script se encarga del proceso de aprendizaje del modelo híbrido.
Gestiona la carga de datos, el ciclo de optimización (backpropagation) y
la validación del rendimiento utilizando métricas especializadas.

Flujo de Ejecución:
1. Carga de Splits: Recupera los índices de entrenamiento/validación/test.
2. Inicialización: Configura el modelo PowerPredictor y el optimizador Adam.
3. Bucle de Entrenamiento (Epochs):
   - Forward Pass: Predicción del modelo.
   - Cálculo de Loss (MSE): Comparación con el Ground Truth.
   - Backward Pass: Ajuste de pesos (aprendizaje).
4. Validación Continua: Monitoreo de MSE y DTW para guardar el mejor modelo.
5. Test Final: Evaluación definitiva sobre datos nunca vistos.
"""

import os
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from tslearn.metrics import dtw

# Importamos el modelo y el dataset definidos en utils.py
from utils import PowerTraceDataset, PowerPredictor


# --- Paths ---
BASE_DIR = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose\Sayid Tests\New Model'
DATASET_DIR = os.path.join(BASE_DIR, 'processed_dataset_retrain')
SPLITS_PATH = os.path.join(DATASET_DIR, 'splits.json')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_model_retrain.pth')

# Hiperparámetros del Experimento
NUM_EPOCHS = 15       # Ciclos completos sobre el dataset
BATCH_SIZE = 16       # Muestras procesadas simultáneamente 
LEARNING_RATE = 0.001 # Velocidad de convergencia del optimizador
HIDDEN_DIM_GNN = 64   # Capacidad de representación estructural
HIDDEN_DIM_LSTM = 128 # Capacidad de memoria temporal

# =============================================================================
# 1. FUNCIONES AUXILIARES DE ENTRENAMIENTO
# =============================================================================

def collate_fn(batch):
    """
    Función de colapso personalizada para PyTorch Geometric.
    
    PyG utiliza un objeto `Batch` especial que concatena todos los grafos pequeños en un "super-grafo"
    disconexo para procesarlos en paralelo en la GPU.
    """
    # Filtramos muestras corruptas (None) para robustez
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Ejecuta una época de entrenamiento (Fase de Aprendizaje).
    """
    model.train() # Activa Dropout y BatchNormalization
    total_loss = 0
    num_samples_processed = 0
    
    for data in tqdm(loader, desc="Entrenando"):
        if data is None: continue
            
        data = data.to(device)
        optimizer.zero_grad() # Limpieza de gradientes acumulados
        
        # 1. Forward Pass: El modelo predice la traza de potencia
        prediction = model(data)
        
        # 2. Loss Calculation: Error Cuadrático Medio
        # Redimensionamos el target para coincidir con [Batch, 400]
        target = data.power_trace.reshape(data.num_graphs, -1)
        loss = criterion(prediction, target)
        
        # 3. Backward Pass: Cálculo de gradientes y actualización de pesos
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_samples_processed += data.num_graphs
        
    return total_loss / num_samples_processed if num_samples_processed > 0 else 0

def evaluate(model, loader, device, desc="Evaluando"):
    """
    Evalúa el modelo sin actualizar pesos (Inferencia).
    Calcula métricas clave: MSE (precisión puntual) y DTW (similitud de forma).
    """
    model.eval() # Desactiva Dropout
    all_predictions, all_actuals = [], []
    
    with torch.no_grad(): # Ahorra memoria y cómputo al no rastrear gradientes
        for data in tqdm(loader, desc=desc):
            if data is None: continue
            data = data.to(device)
            
            prediction = model(data)
            target = data.power_trace.reshape(data.num_graphs, -1)
            
            all_predictions.append(prediction.cpu().numpy())
            all_actuals.append(target.cpu().numpy())
    
    if not all_predictions:
        return float('nan'), float('nan')

    # Concatenamos todo para métricas globales
    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)
    
    # Métrica 1: MSE (Estándar para regresión)
    mse = np.mean((all_predictions - all_actuals)**2)
    
    # Métrica 2: DTW (Dynamic Time Warping)
    try:
        dtw_scores = [dtw(pred, act) for pred, act in zip(all_predictions, all_actuals)]
        dtw_score = np.mean(dtw_scores)
    except Exception:
        dtw_score = float('nan')
    
    return mse, dtw_score


# =============================================================================
# 2. BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================

def main():
    print("--- INICIANDO PROTOCOLO DE ENTRENAMIENTO ---")
    
    # 1. Carga de Metadatos de Splits
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    
    # 2. Instanciación de Datasets
    # Usamos la clase personalizada que soporta Lazy Loading
    train_dataset = PowerTraceDataset(DATASET_DIR, splits['train'])
    val_dataset = PowerTraceDataset(DATASET_DIR, splits['validation'])
    test_dataset = PowerTraceDataset(DATASET_DIR, splits['test'])
    
    # 3. Configuración de DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Configuración de Hardware (CUDA/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de cómputo: {device}")
    
    # 4. Inicialización del Modelo
    # Necesitamos una muestra para inferir dimensiones de entrada dinámicamente
    sample = next(item for item in train_dataset if item is not None)
    if sample is None:
        print("Error: Dataset de entrenamiento vacío o corrupto.")
        return
        
    print("Arquitectura del Modelo:")
    print(f" -> Entrada GNN (Features Nodos): {sample.num_node_features}")
    print(f" -> Entrada LSTM (Nodos Activos): {sample.activity.shape[1]}")
    print(f" -> Salida (Puntos Temporales): {len(sample.power_trace)}")
    
    model = PowerPredictor(
        gnn_in=sample.num_node_features,
        gnn_hidden=HIDDEN_DIM_GNN,
        num_nodes=sample.activity.shape[1],
        cnn_out=32,
        lstm_hidden=HIDDEN_DIM_LSTM,
        common_embedding_dim=128,
        output_len=len(sample.power_trace)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 5. Bucle de Entrenamiento
    best_val_mse = float('inf') 

    print("\n--- Comenzando optimización ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Fase de Entrenamiento
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Fase de Validación
        val_mse, val_dtw = evaluate(model, val_loader, device, desc="Validando")
        
        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_mse:.6f} | Val DTW: {val_dtw:.4f}")

        # Guardado de Checkpoint (Model Checkpointing)
        # Solo guardamos si el modelo mejora en el set de validación
        if not np.isnan(val_mse) and val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Nuevo mejor modelo guardado! (MSE: {best_val_mse:.6f})")

    # 6. Evaluación Final
    print("\n--- Entrenamiento finalizado. Evaluando en Test Set ---")
    # Cargamos los pesos del mejor modelo obtenido, no el de la última época
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    test_mse, test_dtw = evaluate(model, test_loader, device, desc="Test Final")
    
    print("\nRESULTADOS FINALES:")
    print(f" -> Test MSE (Precisión Puntual): {test_mse:.6f}")
    print(f" -> Test DTW (Similitud de Forma): {test_dtw:.4f}")

if __name__ == "__main__":
    main()
