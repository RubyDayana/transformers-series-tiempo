# Atención y Transformers para series de tiempo

Implementación desde cero, en **PyTorch**, del mecanismo de atención y de un encoder **Transformer** básico
para predecir la temperatura mínima diaria (Melbourne, 1981-1990) a partir de los 30 días anteriores.

**Estudiante:** Ruby Dayana Cárdenas Gómez · **Curso:** CADI Deep Learning Conceptos · **Docente:** Nidia Stella García Roa · Universidad de Cundinamarca, 2026

| Notebook | Contenido | Abrir |
|---|---|---|
| `01_Atencion_Series_Tiempo.ipynb` | Atención de producto punto escalado desde cero, LSTM vs. LSTM + atención, interpretación de los pesos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/transformers-series-tiempo/blob/main/notebooks/01_Atencion_Series_Tiempo.ipynb) |
| `02_Transformer_Series_Tiempo.ipynb` | Codificación posicional, atención multi-cabeza, capa encoder, Transformer completo, ablaciones y mapas de atención | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/transformers-series-tiempo/blob/main/notebooks/02_Transformer_Series_Tiempo.ipynb) |

## Estructura
```
├── data/daily-min-temperatures.csv   # 3650 registros diarios (°C)
├── notebooks/
│   ├── 01_Atencion_Series_Tiempo.ipynb
│   ├── 02_Transformer_Series_Tiempo.ipynb
│   └── figuras/                      # gráficas generadas por los notebooks
├── requirements.txt
└── README.md
```

## Datos y preprocesamiento
- Partición cronológica: 1981-1988 entrenamiento · 1989 validación · 1990 prueba.
- Normalización z-score con estadísticos del entrenamiento.
- Ventanas deslizantes de 30 días → temperatura del día siguiente.

## Resultados (año de prueba 1990)

| Modelo | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Persistencia (mañana = hoy) | 2,025 | 2,582 | 0,552 |
| LSTM | 1,722 | 2,244 | 0,661 |
| LSTM + Atención | **1,694** | **2,211** | **0,671** |
| Transformer (2 capas, 4 cabezas) | 1,753 | 2,269 | 0,654 |
| Transformer, salida promedio | 1,736 | 2,285 | 0,649 |
| Transformer, salida promedio **sin** cod. posicional | 1,981 | 2,616 | 0,540 |

Hallazgos principales: la atención mejora ligeramente a la LSTM y permite ver qué días usa el modelo; el
Transformer iguala a la LSTM en menos épocas; y sin codificación posicional el Transformer cae por debajo
de la línea base cuando debe razonar sobre el orden de los días.

## Ejecución
En Colab: botón *Open in Colab* → *Entorno de ejecución → Ejecutar todas*. Los datos se descargan solos.
Local: `pip install -r requirements.txt` y abrir los notebooks con Jupyter.

## Referencias
- Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS 30. https://doi.org/10.48550/arXiv.1706.03762
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate*. ICLR. https://doi.org/10.48550/arXiv.1409.0473
- Doshi, K. (2020). *Transformers explained visually (Part 1-3)*. Towards Data Science.
- Brownlee, J. *Daily minimum temperatures in Melbourne* [Dataset]. https://github.com/jbrownlee/Datasets
