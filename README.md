# transformers-series-tiempo

Trabajo de CADI Deep Learning Conceptos (Universidad de Cundinamarca, profe Nidia Stella García Roa).
Hecho por Ruby Dayana Cárdenas Gómez.

La idea fue programar en PyTorch un mecanismo de atención y un Transformer sencillo, y usarlos para predecir
la temperatura mínima de un día con los 30 días anteriores. Los datos son temperaturas diarias de Melbourne
entre 1981 y 1990.

## Notebooks

- `notebooks/01_Atencion_Series_Tiempo.ipynb`: atención hecha a mano, comparación entre una LSTM normal y
  una LSTM con atención, y gráficas de a qué días les presta atención el modelo.
  [Abrir en Colab](https://colab.research.google.com/github/RubyDayana/transformers-series-tiempo/blob/main/notebooks/01_Atencion_Series_Tiempo.ipynb)
- `notebooks/02_Transformer_Series_Tiempo.ipynb`: el encoder del Transformer por partes (codificación
  posicional, atención multi-cabeza, feed-forward), entrenamiento y pruebas quitando la codificación posicional.
  [Abrir en Colab](https://colab.research.google.com/github/RubyDayana/transformers-series-tiempo/blob/main/notebooks/02_Transformer_Series_Tiempo.ipynb)

Para correrlos en Colab: Entorno de ejecución > Ejecutar todas. Los datos se descargan solos.
También están en la carpeta `data/`.

## Cómo se prepararon los datos

Se entrenó con 1981 a 1988, se validó con 1989 y se probó con 1990. Los datos se normalizaron con la media
y la desviación del entrenamiento, y se armaron ventanas de 30 días para predecir el día 31.

## Resultados en 1990

| Modelo | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Mañana = hoy (referencia) | 2,025 | 2,582 | 0,552 |
| LSTM | 1,722 | 2,244 | 0,661 |
| LSTM + atención | 1,694 | 2,211 | 0,671 |
| Transformer | 1,753 | 2,269 | 0,654 |
| Transformer con salida promedio | 1,736 | 2,285 | 0,649 |
| Transformer con salida promedio, sin codificación posicional | 1,981 | 2,616 | 0,540 |

La atención mejoró un poco a la LSTM, y lo más útil fue poder ver qué días usaba para predecir.
El Transformer quedó casi igual que la LSTM, pero necesitó menos épocas. Cuando le quité la codificación
posicional (y la salida era el promedio de los 30 días), quedó peor que la referencia, porque ya no sabía
cuál era el día más reciente.

## Referencias

- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 30. https://doi.org/10.48550/arXiv.1706.03762
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR. https://doi.org/10.48550/arXiv.1409.0473
- Doshi, K. (2020). Transformers explained visually. Towards Data Science.
- Brownlee, J. Daily minimum temperatures in Melbourne [Dataset]. https://github.com/jbrownlee/Datasets
