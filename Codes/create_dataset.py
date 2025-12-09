"""
create_dataset.py

Este script es el encargado de la transformación "Hardware-to-Tensor".
Su función es convertir los archivos originales de diseño de circuitos (Verilog) y simulaciones (VCD)
en objetos geométricos (Grafos) y series temporales compatibles con PyTorch Geometric.

Flujo de Trabajo:
1. Parsing Estructural: Verilog -> JSON -> Grafo (Nodos y Aristas).
2. Parsing Dinámico: VCD -> Matriz de Actividad (Nodos x Tiempo).
3. Normalización: Ajuste de escalas para estabilidad numérica del modelo.
4. Serialización: Guardado de objetos .pt para carga eficiente.
"""

import os
import json
import subprocess
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vcdvcd import VCDVCD

# Importamos las utilidades que contienen la lógica de extracción física de la topología y la actividad
from utils import construir_grafo_desde_json, crear_matriz_actividad

# --- PATHS ---
BASE_DIR = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose'
RAW_FILES_DIR = os.path.join(BASE_DIR, '1_source_files')
SAMPLES_DIR = os.path.join(BASE_DIR, '3_dataset_samples')
NETLIST_JSON_PATH = os.path.join(RAW_FILES_DIR, 'aes_netlist.json')
NEW_BATCH_DIR = os.path.join(BASE_DIR, 'Sayid Tests', 'Third Batch')

# Dataset procesado
OUTPUT_DIR = os.path.join(BASE_DIR, 'Sayid Tests', 'New Model', 'processed_dataset_retrain')


def run_yosys(verilog_path, json_output_path):
    """
    Convierte el Netlist Verilog (.v) a una estructura jerárquica JSON con un parsing estructural
  
    """
    print(f"\n--- Ejecutando síntesis Yosys: {verilog_path} -> JSON ---")
    yosys_script = f'read_verilog "{verilog_path}"; proc; write_json "{json_output_path}"'
    
    try:
        subprocess.run(['yosys', '-p', yosys_script], check=True, capture_output=True, text=True)
        print(" -> Conversión estructural exitosa.")
        return True
    except Exception as e:
        print(f"ERROR CRÍTICO: Yosys falló. Verifique la instalación. Detalles: {e}")
        return False

def main():
    print("--- INICIANDO PIPELINE DE PREPROCESAMIENTO (HW -> TENSORES) ---")
    
    # ---------------------------------------------------------
    # PASO 1: CONSTRUCCIÓN DEL GRAFO ESTÁTICO (Base para GNN)
    # ---------------------------------------------------------
    # Si no existe la representación intermedia JSON, la creamos.
    if not os.path.exists(NETLIST_JSON_PATH):
        run_yosys(os.path.join(RAW_FILES_DIR, 'aes_netlist.v'), NETLIST_JSON_PATH)

    # Construimos el grafo base.
    # graph_base: Objeto torch_geometric.data.Data (x, edge_index).
    #             Representa la topología física del circuito.
    # cell_type_map: Diccionario que mapea tipos de celdas (AND, XOR) a enteros.
    graph_base, cell_map, net_map, cell_type_map = construir_grafo_desde_json(NETLIST_JSON_PATH)
    
    if not graph_base:
        print("Error: No se pudo construir el grafo base.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardamos el mapeo de tipos de celdas. Esto es crucial para que el
    # Embedding Layer de la GNN sepa cuántos tipos de nodos existen.
    with open(os.path.join(OUTPUT_DIR, 'cell_types.json'), 'w') as f:
        json.dump(cell_type_map, f)
    print(f"Grafo estático construido. Vocabulario de celdas: {len(cell_type_map)} tipos.")

    # ---------------------------------------------------------
    # PASO 2: AGREGACIÓN DE MUESTRAS (Dinámica Temporal)
    # ---------------------------------------------------------
    master_sample_list = []
    
    # Dataset Original (800 muestras)
    for i in range(800):
        master_sample_list.append({
            'vcd_path': os.path.join(SAMPLES_DIR, f'sample_{i:03d}', 'activity.vcd'),
            'power_path': os.path.join(SAMPLES_DIR, f'sample_{i:03d}', 'power.data'),
            'id': i
        })
    
    # Dataset Nuevo (Third Batch - Generalización)
    new_vcds = [f for f in os.listdir(NEW_BATCH_DIR) if f.endswith('.vcd')]
    for i, vcd_file in enumerate(new_vcds):
        base_name = os.path.splitext(vcd_file)[0]
        master_sample_list.append({
            'vcd_path': os.path.join(NEW_BATCH_DIR, vcd_file),
            'power_path': os.path.join(NEW_BATCH_DIR, f'{base_name}.data'),
            'id': 800 + i
        })

    print(f"\nMuestras totales encontradas: {len(master_sample_list)}")

    # ---------------------------------------------------------
    # PASO 3: CÁLCULO DE ESTADÍSTICAS (Evitar Data Leakage)
    # ---------------------------------------------------------
    # Para normalizar la potencia (Target), calculamos min/max SOLO con datos de TRAIN.
    
    indices = list(range(len(master_sample_list)))
    random.shuffle(indices)
    train_split_idx = int(0.7 * len(indices))
    train_indices = indices[:train_split_idx]

    print("\n--- Calculando estadísticas de normalización (Solo Train Set) ---")
    all_power_values = []
    
    for idx in tqdm(train_indices, desc="Escaneando Ground Truth"):
        p_path = master_sample_list[idx]['power_path']
        if os.path.exists(p_path):
            # Leemos la traza de potencia (Ground Truth)
            df = pd.read_csv(p_path, comment='#', delim_whitespace=True, names=['Time', 'Power'])
            all_power_values.append(df['Power'].values)
    
    all_power_arr = np.concatenate(all_power_values)
    power_min, power_max = all_power_arr.min(), all_power_arr.max()
    
    # Guardamos stats para poder des-normalizar las predicciones después
    norm_stats = {'min': float(power_min), 'max': float(power_max)}
    with open(os.path.join(OUTPUT_DIR, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f)

    # ---------------------------------------------------------
    # PASO 4: EXTRACCIÓN DE CARACTERÍSTICAS Y SERIALIZACIÓN
    # ---------------------------------------------------------
    print("\n--- Generando tensores (GNN + LSTM inputs) ---")
    
    for sample_info in tqdm(master_sample_list, desc="Procesando VCDs"):
        if not os.path.exists(sample_info['power_path']) or not os.path.exists(sample_info['vcd_path']):
            continue
            
        # A. Cargar Ground Truth
        df_potencia = pd.read_csv(sample_info['power_path'], comment='#', delim_whitespace=True, names=['Time', 'Power'])
        
        # B. Procesar Dinámica (VCD -> Matriz de Actividad)
        # Esta matriz será la entrada de la rama LSTM/CNN del modelo.
        # Shape esperado: [Num_Nodos, Time_Steps]
        vcd = VCDVCD(sample_info['vcd_path'])
        activity_matrix = crear_matriz_actividad(vcd, df_potencia, graph_base.num_nodes, len(cell_map), net_map)
        
        # C. Normalización del Target (Escala [-1, 1] para mejor convergencia con Tanh/ReLU)
        power_vals = df_potencia['Power'].values
        norm_power = 2 * (power_vals - power_min) / (power_max - power_min) - 1
        
        # D. Empaquetado en Objeto PyG (Data)
        # Clonamos la estructura estática y le inyectamos la dinámica de esta muestra
        final_packet = graph_base.clone()
        final_packet.activity = activity_matrix  # Input Dinámico
        final_packet.power_trace = torch.tensor(norm_power, dtype=torch.float) # Target
        
        # Guardamos como tensor .pt
        torch.save(final_packet, os.path.join(OUTPUT_DIR, f"sample_{sample_info['id']:03d}.pt"))

    # ---------------------------------------------------------
    # PASO 5: DEFINICIÓN DE SPLITS
    # ---------------------------------------------------------
    # Verificamos qué archivos se crearon realmente y dividimos los índices
    processed_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('sample_') and f.endswith('.pt')]
    valid_indices = []
    for f in processed_files:
        try:
            valid_indices.append(int(f.replace('sample_', '').replace('.pt', '')))
        except: pass

    random.shuffle(valid_indices)
    n_total = len(valid_indices)
    train_end = int(0.7 * n_total)
    val_end = int(0.85 * n_total)
    
    splits = {
        'train': valid_indices[:train_end],
        'validation': valid_indices[train_end:val_end],
        'test': valid_indices[val_end:]
    }
    
    with open(os.path.join(OUTPUT_DIR, 'splits.json'), 'w') as f:
        json.dump(splits, f)
        
    print(f"\nPipeline finalizado. Splits guardados:")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['validation'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()
