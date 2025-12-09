"""
predict.py

Este script utiliza el modelo entrenado (PowerPredictor) para estimar el consumo
de potencia de nuevas simulaciones (archivos .vcd) que nunca ha visto.

Flujo de Inferencia:
1. Carga de Contexto: Recupera la estructura del grafo estático del circuito.
2. Carga del Modelo: Restaura los pesos sinápticos aprendidos.
3. Preprocesamiento al Vuelo: Convierte VCDs crudos en Tensores de Actividad.
4. Forward Pass: Ejecuta la red neuronal en modo evaluación.
5. Reporte: Genera gráficas y archivos de datos con la predicción.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from vcdvcd import VCDVCD
from tqdm import tqdm

# Importamos la arquitectura y las herramientas de procesamiento
from utils import PowerPredictor, crear_matriz_actividad

# --- PATHS ---
BASE_DIR = r'C:\Users\amesa\OneDrive\Delaware\Power Traces'
MODEL_PATH = os.path.join(BASE_DIR, 'Data', 'Power_trace_jose', 'best_model_retrain.pth')

# Rutas de metadatos necesarios para reconstruir el grafo
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'Data', 'Power_trace_jose', '4_processed_dataset')
ORIGINAL_JSON_NETLIST = os.path.join(BASE_DIR, 'Data', 'Power_trace_jose', '1_source_files', 'aes_netlist.json')

# Carpeta de Entrada (Nuevas simulaciones a evaluar)
INPUT_VCD_DIR = os.path.join(BASE_DIR, 'Data', 'Power_trace_jose', 'Sayid Tests', 'Third Batch')

# Carpeta de Salida (Resultados)
OUTPUT_PREDICTIONS_DIR = os.path.join(INPUT_VCD_DIR, 'predictions')
os.makedirs(OUTPUT_PREDICTIONS_DIR, exist_ok=True)

# Hiperparámetros (Deben coincidir exactamente con los usados en train.py)
TRACE_LENGTH = 400
TIME_STEP = 650.0  # Paso de tiempo de la simulación original


# =============================================================================
# 1. FUNCIONES AUXILIARES DE EXPORTACIÓN
# =============================================================================

def save_power_trace_data(predicted_trace, output_path, start_time=325.0, time_step=650.0):
    """
    Guarda la predicción en formato .data estándar de la industria.
    Permite comparar fácilmente con herramientas comerciales de EDA.
    """
    with open(output_path, 'w') as f:
        f.write("# Predicted Power Profile using Deep Learning Model\n")
        current_time = start_time
        for power_value in predicted_trace:
            # Formato: [Tiempo] [Potencia]
            f.write(f"{current_time:.1f} {power_value:.12f}\n")
            current_time += time_step


# =============================================================================
# 2. BLOQUE PRINCIPAL DE INFERENCIA
# =============================================================================

def main():
    print("--- INICIANDO PIPELINE DE INFERENCIA ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de inferencia: {device}")

    # ---------------------------------------------------------
    # PASO 1: RECONSTRUCCIÓN DEL GRAFO ESTÁTICO
    # ---------------------------------------------------------
    # La GNN necesita la topología (x, edge_index). Como el circuito físico (netlist)
    # no cambia entre simulaciones, cargamos un grafo "plantilla" preprocesado.
    print("Cargando estructura topológica del circuito...")
    try:
        # Recuperamos el mapeo de nombres de Nets para parsear el VCD
        with open(ORIGINAL_JSON_NETLIST, 'r') as f:
            module_data = next(iter(json.load(f)['modules'].values()))
        netnames_json = list(module_data['netnames'].keys())
        net_map_original = {i: name.split('.')[-1] for i, name in enumerate(netnames_json)}
        
        # Cargamos el grafo físico (ignorando su actividad antigua)
        # Usamos sample_000.pt como "molde"
        template_path = os.path.join(PROCESSED_DATA_DIR, 'sample_000.pt')
        data_template = torch.load(template_path)
        
        # Creamos una copia limpia solo con la estructura estática
        graph_base = data_template.clone()
        if hasattr(graph_base, 'activity'): del graph_base.activity
        if hasattr(graph_base, 'power_trace'): del graph_base.power_trace
        
        print(" -> Grafo base cargado correctamente.")
        
    except FileNotFoundError as e:
        print(f"Error crítico: Falta archivo de metadatos. {e}")
        return

    # ---------------------------------------------------------
    # PASO 2: CARGA DEL MODELO ENTRENADO
    # ---------------------------------------------------------
    print(f"Cargando pesos del modelo desde: {MODEL_PATH}")
    
    # Instanciamos la arquitectura con las mismas dimensiones que en el entrenamiento
    model = PowerPredictor(
        gnn_in=graph_base.num_node_features,
        gnn_hidden=64,
        num_nodes=graph_base.num_nodes,
        cnn_out=32,
        lstm_hidden=128,
        common_embedding_dim=128,
        output_len=TRACE_LENGTH
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Modo evaluación (congela Dropout/BatchNorm)
        print(" -> Modelo listo para inferencia.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # ---------------------------------------------------------
    # PASO 3: PROCESAMIENTO DE NUEVAS SIMULACIONES
    # ---------------------------------------------------------
    vcds_to_process = [f for f in os.listdir(INPUT_VCD_DIR) if f.endswith('.vcd')]
    if not vcds_to_process:
        print(f"No se encontraron archivos .vcd en {INPUT_VCD_DIR}")
        return

    print(f"\nProcesando {len(vcds_to_process)} nuevos archivos de actividad...")

    for vcd_filename in tqdm(vcds_to_process, desc="Generando predicciones"):
        vcd_path = os.path.join(INPUT_VCD_DIR, vcd_filename)
        
        # A. Preprocesamiento "Al Vuelo"
        # Convertimos el archivo VCD (eventos) en Matriz de Actividad (Tensor)
        dummy_time = np.linspace(0, (TRACE_LENGTH-1)*TIME_STEP, TRACE_LENGTH)
        dummy_df_power = pd.DataFrame({'Time': dummy_time})
        
        # Calculamos número de celdas reales
        # Aquí asumimos que graph_base es el grafo puro sin padding
        num_cells_real = (graph_base.x.sum(dim=1) > 0).sum().item()
        
        vcd = VCDVCD(vcd_path)
        activity_matrix = crear_matriz_actividad(
            vcd, 
            dummy_df_power, 
            graph_base.num_nodes, 
            num_cells_real, 
            net_map_original
        )
        
        # B. Ensamblaje del Batch
        # Combinamos el grafo estático (graph_base) con la nueva actividad dinámica
        inference_packet = graph_base.clone()
        inference_packet.activity = activity_matrix
        
        # DataLoader crea el batch correctamente para PyG
        loader = DataLoader([inference_packet], batch_size=1)
        batch = next(iter(loader)).to(device)

        # C. Inferencia (Forward Pass)
        with torch.no_grad():
            # El modelo devuelve [1, 400], aplanamos a [400]
            predicted_trace = model(batch).cpu().numpy().flatten()
            
        # -----------------------------------------------------
        # PASO 4: GUARDADO DE RESULTADOS
        # -----------------------------------------------------
        base_name = os.path.splitext(vcd_filename)[0]
        
        # 1. Gráfica PNG
        plt.figure(figsize=(12, 6))
        plt.plot(predicted_trace, label='Predicción IA', color='#2ca02c', linewidth=2)
        plt.title(f'Predicción de Potencia: {vcd_filename}')
        plt.xlabel('Pasos de Tiempo')
        plt.ylabel('Potencia Normalizada')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_PREDICTIONS_DIR, f'{base_name}.png'))
        plt.close()

        # 2. Archivo de Datos Numéricos
        data_out_path = os.path.join(OUTPUT_PREDICTIONS_DIR, f'{base_name}.data')
        save_power_trace_data(predicted_trace, data_out_path)

    print(f"\n--- Inferencia completada exitosamente ---")
    print(f"Resultados guardados en: {OUTPUT_PREDICTIONS_DIR}")

if __name__ == "__main__":
    main()
