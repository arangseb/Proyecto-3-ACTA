import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import pickle

# Cargar el archivo CSV
df = pd.read_csv('datos_limpios.csv')

# Quitar columna unnamed
df.drop(df.columns[[0]], axis=1, inplace=True)

X = df.drop(['COLE_MCPIO_UBICACION','ESTU_DEPTO_PRESENTACION','ESTU_DEPTO_RESIDE','ESTU_FECHANACIMIENTO',
             'ESTU_MCPIO_PRESENTACION','ESTU_MCPIO_RESIDE','ESTU_NACIONALIDAD','ESTU_PAIS_RESIDE',
             'PUNT_GLOBAL','COLE_NOMBRE_ESTABLECIMIENTO','COLE_NOMBRE_SEDE','PUNT_INGLES',
             'PUNT_MATEMATICAS','PUNT_SOCIALES_CIUDADANAS','PUNT_C_NATURALES','PUNT_LECTURA_CRITICA'], axis=1)
y = df['PUNT_GLOBAL']

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Seleccionar columnas categóricas
cat_columns = X.select_dtypes(include=['object', 'category']).columns

# OneHotEncoding para las categóricas
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Cambiar 'sparse' por 'sparse_output'
X_train_cat = ohe.fit_transform(X_train[cat_columns])
X_val_cat = ohe.transform(X_val[cat_columns])

# Guardar el OneHotEncoder entrenado
with open("one_hot_encoder.pkl", "wb") as f:
    pickle.dump(ohe, f)

# Concatenar todas las características preprocesadas
X_train_processed = tf.concat([X_train_cat], axis=1)
X_val_processed = tf.concat([X_val_cat], axis=1)

# === Definir el modelo ===
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(1)  # Salida para regresión
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train_processed.shape[1])

# === Configurar MLflow ===
mlflow.set_experiment("Modelo Redes Neuronales con TensorFlow")  # Configura el experimento
#mlflow.tensorflow.autolog()  # Activar autologging para TensorFlow

with mlflow.start_run(run_name="Regresión - Redes Neuronales_128_64_32"):
    # Entrenar el modelo
    history = model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluar el modelo y registrar métricas adicionales
    val_mse, val_mae = model.evaluate(X_val_processed, y_val, verbose=0)
    mlflow.log_metric("val_mse", val_mse)
    mlflow.log_metric("val_mae", val_mae)
    
    # Guardar el modelo final
    model.save("modelo_nn.keras")
    mlflow.log_artifact("modelo_nn.keras", artifact_path="modelo")
    
    # Graficar pérdida y guardar el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png", artifact_path="plots")
