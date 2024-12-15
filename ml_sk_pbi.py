import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Suponiendo que 'dataset' es un DataFrame de pandas con las columnas 'Ciudad', 'Producto', 'a' y 'Ventas'
X = dataset.drop(columns=['Ventas'])
y = dataset['Ventas']

# Identificación de columnas categóricas y numéricas
categorical_columns = ['Ciudad', 'Producto']
numerical_columns = ['a','m', 'd']

# Configuración de transformadores:
# - OneHotEncoder para las columnas categóricas
# - StandardScaler para las columnas numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns), 
        ('num', StandardScaler(), numerical_columns)
    ],
    remainder='passthrough'  # El resto de columnas se dejan sin cambios
)

# Transformación del target usando 'yeo-johnson'
pt = PowerTransformer(method='yeo-johnson')
y_trans = pt.fit_transform(y.values.reshape(-1, 1)).flatten()

# Pipeline que incluye el preprocesador y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=100))
])

# División en entrenamiento y prueba usando TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_trans[train_index], y_trans[test_index]
    
    # Entrenamiento del modelo dentro del pipeline
    pipeline.fit(X_train, y_train)

    # Evaluación del modelo (opcional)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Guardar el pipeline (incluyendo preprocesamiento y modelo) y el transformador de target
joblib.dump(pipeline, r'c:\\powerbi\\modelo\\pipeline_model.pkl')
joblib.dump(pt, r'c:\\powerbi\\modelo\\power_transformer.pkl')

# Para la predicción
# Cargar el pipeline y el transformador de potencia
pipeline = joblib.load(r'C:\\PowerBI\\modelo\\pipeline_model.pkl')
pt = joblib.load(r'C:\\PowerBI\\modelo\\power_transformer.pkl')

# Realizar predicciones
y_pred_transformed = pipeline.predict(dataset)
y_pred = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

# Agregar las predicciones al dataset original
dataset['Predicted_Ventas'] = y_pred

