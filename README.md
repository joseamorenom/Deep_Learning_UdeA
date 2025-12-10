# Predicción de Trazas de Potencia con Deep Learning Híbrido (GNN + LSTM)

**Universidad de Antioquia - Deep Learning**

Este proyecto implementa un modelo híbrido para predecir el consumo de potencia dinámico en circuitos digitales post-síntesis, combinando la estructura física del chip (Netlist) con su actividad temporal (VCD).


## 🧠 Arquitectura del Modelo
* **Codificador Estático (GNN - GAT):** Procesa el grafo del circuito para aprender la topología.
* **Codificador Dinámico (CNN + LSTM):** Procesa la secuencia de actividad para aprender patrones temporales.
* **Predictor (MLP):** Fusiona ambas representaciones para predecir la serie temporal de potencia.

## 📂 Estructura de Archivos
* `create_dataset.py`: Preprocesamiento (Verilog/VCD $\to$ Tensores `.pt`).
* `utils.py`: Definición de la arquitectura GNN+LSTM y clase Dataset.
* `train.py`: Script de entrenamiento, validación y métricas (MSE/DTW).
* `predict.py`: Script de inferencia para generar trazas de nuevos diseños.

## 🚀 Ejecución

**1. Requisitos:**
`pip install torch torch-geometric pandas numpy matplotlib tqdm vcdvcd tslearn`
*(Requiere Yosys instalado en el sistema para parsear Verilog)*
