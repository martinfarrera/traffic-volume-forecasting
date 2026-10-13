Publiqué un dashboard para explorar una evaluación histórica de volumen de tráfico: permite comparar valores observados, predicciones de LightGBM y una referencia sencilla, lag-1.

Demo: https://martinfarrera.github.io/traffic-volume-forecasting/
Código: https://github.com/martinfarrera/traffic-volume-forecasting

Usé el dataset Real-World Traffic Volume Forecasting de Kaggle (licencia CC0-1.0), con 38.373 filas etiquetadas en Train.csv. Tras excluir filas sin características completas, la evaluación utiliza una división cronológica 80/20: 30.564 observaciones para entrenamiento y 7.641 para validación.

Comparación sobre la misma validación (MAE / RMSE; menor es mejor):
• Lag-1, que repite el último valor observado: 595,62 / 826,63.
• HistGradientBoosting: 166,29 / 263,06.
• LightGBM: 164,87 / 262,24.

LightGBM obtuvo el menor error y quedó seleccionado provisionalmente para la demo. Su mejora frente a HistGradientBoosting es modesta: 1,42 puntos de MAE y 0,82 de RMSE.

El alcance importa: es una evaluación histórica de un paso adelante. Los lags se calculan por filas, no por horas transcurridas, y cada predicción de validación utiliza valores objetivo observados previamente, incluidos los de la propia validación. No es un pronóstico en vivo ni multipaso. La selección del modelo usó ese mismo conjunto, sin un test final independiente; Test.csv no contiene valores objetivo.

El repositorio documenta la reproducción del pipeline y una exportación determinista para los mismos datos y entorno. Incluye pruebas automatizadas (23 pasaron en la verificación anterior del dashboard). La publicación es estática en GitHub Pages, con predicciones precalculadas y sin backend de producción. Se verificó disponibilidad HTTP de la página y sus recursos; quedan pendientes las comprobaciones visuales y con lector de pantalla.

Un ejercicio para comparar modelos con una referencia clara y hacer explícito qué significan sus métricas y qué no permiten concluir.

#DataScience #MachineLearning #SeriesTemporales #Python
