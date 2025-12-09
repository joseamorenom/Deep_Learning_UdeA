"""
utils.py

Este script define la arquitectura de Deep Learning Híbrida y las 
herramientas de manipulación de grafos. Es el núcleo computacional del proyecto.

Contenido:
1. Dataset: Manejo eficiente de memoria para grafos grandes.
2. Arquitecturas (Modelos):
   - GNNEncoder: Comprensión espacial/topológica (Estructura del Chip).
   - ActivityEncoder: Comprensión temporal (Dinámica de Señales).
   - PowerPredictor: Fusión multimodal para la predicción final.
3. Preprocesamiento: Conversión de Netlists a Grafos PyG.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from vcdvcd import VCDVCD
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv, global_mean_pool


# ==============================================================================
# 1. CLASE DATASET (Gestión de Datos)
# ==============================================================================

class PowerTraceDataset(Dataset):
    """
    Dataset personalizado para PyTorch Geometric.
    Implementa 'Lazy Loading': los grafos se cargan en RAM solo cuando se necesitan
    para el entrenamiento, evitando desbordamientos de memoria con datasets grandes.
    """
    def __init__(self, root_dir, indices):
        """
        root_dir: Directorio con los archivos .pt procesados.
        indices: Lista de IDs (enteros) que pertenecen a este split (train/test).
        """
        super(PowerTraceDataset, self).__init__(root_dir)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Mapea el índice relativo del DataLoader al ID real del archivo
        real_idx = self.indices[idx]
        return self.get(real_idx)

    def get(self, idx):
        # Carga el tensor desde el disco
        data_path = os.path.join(self.root, f'sample_{int(idx):03d}.pt')
        try:
            data = torch.load(data_path)
            return data
        except FileNotFoundError:
            print(f"Advertencia: Archivo no encontrado {data_path}. Saltando muestra.")
            return None


# ==============================================================================
# 2. DEFINICIÓN DE ARQUITECTURAS 
# ==============================================================================

class GNNEncoder(nn.Module):
    """
    CODIFICADOR ESTRUCTURAL (El 'Ingeniero de Hardware')
    ----------------------------------------------------
    Utiliza Graph Neural Networks (GNN) para procesar el Netlist.
    
    Note: El consumo de potencia depende no solo de una compuerta aislada,
    sino de su vecindario (fan-out, carga capacitiva de vecinos).
    
    Usamos GAT (Graph Attention Networks) para que el modelo aprenda a dar
    más importancia ("atención") a las conexiones críticas que consumen más energía.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        # Capa 1: Agregación de vecindario con múltiples cabezas de atención
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        # Capa 2: Refinamiento del embedding estructural
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, data):
        # x: Características del nodo (Tipo de celda, Área, Leakage)
        # edge_index: Mapa de conexiones (Cables)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Propagación de mensajes a través del grafo
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Global Pooling: Resume todo el grafo en un solo vector latente
        # que representa la "huella digital" estática del circuito.
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding


class ActivityEncoder(nn.Module):
    """
    Procesa la matriz de actividad (VCD) para extraer patrones temporales.
    
    Arquitectura Híbrida:
    1. CNN 1D: Actúa como extractor de características locales. Detecta
       "eventos" de conmutación simultánea (picos de corriente instantáneos).
    2. LSTM Bidireccional: Analiza la secuencia de estos eventos para entender
       la evolución temporal y dependencias de largo plazo en el consumo.
    """
    def __init__(self, num_nodes, cnn_out_channels, lstm_hidden, output_dim):
        super(ActivityEncoder, self).__init__()
        
        # CNN: Escanea la actividad temporal buscando patrones de alto consumo
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=128, stride=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Cálculo dinámico del tamaño de salida de la CNN para conectar la LSTM
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_nodes) # [Batch, Canales, Longitud]
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_flat_size = dummy_output.shape[1] * dummy_output.shape[2]
        
        # LSTM: Procesa la secuencia de características extraídas
        self.lstm = nn.LSTM(
            self.cnn_output_flat_size, 
            lstm_hidden, 
            batch_first=True, 
            num_layers=2, 
            bidirectional=True
        )
        # Proyección final al espacio latente común
        self.fc = nn.Linear(lstm_hidden * 2, output_dim)

    def forward(self, activity_matrix):
        # Entrada: [Batch, Time_Steps, Num_Nodes]
        # La 'imagen' que ve la CNN es la actividad de todos los nodos en un instante.
        batch_size, seq_len, num_nodes = activity_matrix.shape
        
        # Aplanamos el tiempo en el batch para que la CNN procese cada instante independientemente
        activity_reshaped = activity_matrix.view(batch_size * seq_len, 1, num_nodes)
        
        # Extracción de features espaciales (conmutación instantánea)
        cnn_features = self.cnn(activity_reshaped)
        
        # Restauramos la dimensión temporal para la LSTM
        cnn_features_flat = cnn_features.view(batch_size, seq_len, -1)
        
        # Modelado secuencial
        _, (hidden_state, _) = self.lstm(cnn_features_flat)
        
        # Concatenamos los estados finales (forward + backward) para el embedding final
        forward_hidden = hidden_state[-2,:,:]
        backward_hidden = hidden_state[-1,:,:]
        hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.fc(hidden_concat)


class PowerPredictor(nn.Module):
    """
    Modelo Final: hace la fusión de la información Estática (GNN) y Dinámica (ActivityEncoder)
    para realizar la regresión final de la traza de potencia.
    """
    def __init__(self, gnn_in, gnn_hidden, num_nodes, cnn_out, lstm_hidden, common_embedding_dim, output_len):
        super(PowerPredictor, self).__init__()
        
        # Instanciamos los expertos
        self.gnn_encoder = GNNEncoder(gnn_in, gnn_hidden, common_embedding_dim)
        self.activity_encoder = ActivityEncoder(num_nodes, cnn_out, lstm_hidden, common_embedding_dim)
        
        # Cabezal de predicción (MLP)
        self.prediction_head = nn.Sequential(
            nn.Linear(common_embedding_dim * 2, 512), # Fusión por concatenación
            nn.ReLU(),
            nn.Dropout(0.3), # Regularización
            nn.Linear(512, output_len) # Salida: Traza de 400 puntos
        )

    def forward(self, data):
        # 1. Obtener contexto estructural del circuito
        graph_embedding = self.gnn_encoder(data)
        
        # 2. Obtener contexto dinámico de la simulación
        # Necesitamos re-dimensionar la actividad para separar los grafos del batch
        num_graphs = data.num_graphs
        activity_batched = data.activity.reshape(num_graphs, -1, data.activity.shape[-1])
        activity_embedding = self.activity_encoder(activity_batched)
        
        # 3. Fusión y Predicción
        # El modelo decide cuánta potencia se consume combinando QUÉ es el circuito (GNN) con QUÉ está haciendo (ActivityEncoder).
        combined_embedding = torch.cat([graph_embedding, activity_embedding], dim=1)
        predicted_trace = self.prediction_head(combined_embedding)
        
        return predicted_trace


# ==============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO (Ingeniería de Grafos)
# ==============================================================================

def construir_grafo_desde_json(ruta_json_netlist, cell_type_map=None):
    """
    Transforma la jerarquía JSON de Yosys en tensores de PyTorch Geometric.
    Construye un grafo bipartito heterogéneo (Celdas <-> Cables).
    """
    with open(ruta_json_netlist, 'r') as f:
        data = json.load(f)
    module_data = next(iter(data['modules'].values()))
    cells = module_data['cells']
    
    # Manejo del diccionario de tipos de celdas (Vocabulario)
    if cell_type_map is None:
        print("Creando nuevo mapeo de tipos de celdas...")
        cell_types = {cell_type: i for i, cell_type in enumerate(set(cell['type'] for cell in cells.values()))}
        return_new_map = True
    else:
        cell_types = cell_type_map
        return_new_map = False

    # Mapeo de identificadores únicos
    cell_to_id = {name: i for i, name in enumerate(cells)}
    
    # Cálculo robusto del número de nets (cables)
    max_net_index = 0
    for cell_info in cells.values():
        for net_indices in cell_info['connections'].values():
            if net_indices:
                max_net_index = max(max_net_index, max(net_indices))
    num_nets = max_net_index + 1
    num_cells = len(cell_to_id)
    
    netnames_json = list(module_data['netnames'].keys())
    net_id_to_name = {i: name.split('.')[-1] for i, name in enumerate(netnames_json)}
    
    # Construcción de la Lista de Adyacencia (Edge Index)
    # Creamos conexiones bidireccionales entre Celdas y Nets
    edge_list = []
    for cell_name, cell_info in cells.items():
        cell_id = cell_to_id[cell_name]
        for port_name, net_indices in cell_info['connections'].items():
            for net_index in net_indices:
                # Offset para distinguir nodos de tipo 'Net' de nodos tipo 'Cell'
                net_node_id = net_index + num_cells 
                edge_list.append([cell_id, net_node_id])
                edge_list.append([net_node_id, cell_id])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Construcción de la Matriz de Características (X)
    # Usamos One-Hot Encoding para el tipo de celda.
    # NOTA: Aquí es donde se inyectaría información física (Área, Leakage) del archivo .lib para hacer el modelo "Physics-Informed".
    cell_features = torch.zeros(num_cells, len(cell_types))
    for cell_name, cell_info in cells.items():
        cell_id = cell_to_id[cell_name]
        cell_type = cell_info['type']
        if cell_type in cell_types:
            type_id = cell_types[cell_type]
            cell_features[cell_id, type_id] = 1
            
    net_features = torch.zeros(num_nets, len(cell_types)) # Nets tienen vector cero por ahora
    x = torch.cat([cell_features, net_features], dim=0)
    
    graph_data = Data(x=x, edge_index=edge_index)
    
    if return_new_map:
        return graph_data, cell_to_id, net_id_to_name, cell_types
    else:
        return graph_data, cell_to_id, net_id_to_name

def crear_matriz_actividad(vcd, df_potencia, num_total_nodos, num_cells, net_id_to_name):
    """
    Sincroniza el archivo de eventos VCD con la escala de tiempo de la traza de potencia.
    Genera una matriz densa [Tiempo, Nodos] donde 1 indica conmutación.
    """
    num_timesteps = len(df_potencia)
    activity_matrix = np.zeros((num_timesteps, num_total_nodos), dtype=np.uint8)
    time_bins = df_potencia['Time'].values
    
    # Mapa optimizado para búsqueda rápida de señales
    vcd_signal_map = {s.split('.')[-1]: s for s in vcd.signals}
    
    # Vinculación: Señal VCD -> ID de Nodo en el Grafo
    vcd_signal_to_net_id = {}
    for net_id, net_name in net_id_to_name.items():
        if net_name in vcd_signal_map:
            vcd_signal = vcd_signal_map[net_name]
            vcd_signal_to_net_id[vcd_signal] = net_id + num_cells
            
    # Llenado de la matriz (Binning temporal)
    for vcd_signal, net_node_id in vcd_signal_to_net_id.items():
        if net_node_id >= num_total_nodos: continue
        
        activity_signal = vcd[vcd_signal].tv
        switch_times = [t for t, v in activity_signal]
        
        # Digitalizamos los tiempos de conmutación a los bins de la traza de potencia
        time_indices = np.digitize(switch_times, bins=time_bins)
        
        for idx in time_indices:
            if idx < num_timesteps:
                activity_matrix[idx, net_node_id] = 1
                
    return torch.from_numpy(activity_matrix).float()
