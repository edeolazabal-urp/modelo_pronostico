from autots import AutoTS
import pandas as pd

# Supongamos que tu dataset tiene las columnas: 'Fecha', 'a', 'Ciudad', 'Producto', 'Ventas', 'm', 'd'
# Preparar los datos (en AutoTS, la columna de fecha debe estar en el índice)

# Combinamos las columnas categóricas en una sola para simplificar
dataset['categoria_combinada'] = dataset[['a', 'Ciudad', 'Producto']].agg('_'.join, axis=1)

# Formato requerido por AutoTS
dataset['Fecha'] = pd.to_datetime(dataset['Fecha'])
dataset.set_index('Fecha', inplace=True)

# Crear el modelo de AutoTS
model = AutoTS(
    forecast_length=30,  # pronóstico para 30 períodos en el futuro
    frequency='infer',  # inferir la frecuencia temporal de los datos
    ensemble='simple',  # combinación simple de modelos
    model_list="fast",  # lista de modelos a usar; 'fast' incluye los más rápidos
    transformer_list="fast",  # lista de transformadores, similar a 'yeo-johnson'
    max_generations=5,  # número de iteraciones de optimización
    num_validations=2,  # número de validaciones cruzadas
    validation_method='backwards',  # método de validación
    model_interrupt=True,  # permitir la interrupción del modelo si es necesario
    verbose=2,  # nivel de verbosidad
    drop_most_recent=1  # eliminar la observación más reciente para la validación
)

# Entrenar el modelo
model = model.fit(
    dataset,
    date_col=None,  # columna de fechas, si está en el índice poner None
    value_col='Ventas',  # columna objetivo
    id_col='categoria_combinada'  # columna de identificación de series temporales
)

# Obtener el mejor modelo
prediction = model.predict()

# Acceder al pronóstico y otros detalles del modelo
forecasts_df = prediction.forecast  # DataFrame con el pronóstico

# Guardar el modelo para su uso posterior
model.export_model(r'c:\powerbi\modelo_autots.zip')
