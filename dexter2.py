import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import supabase
import time
from datetime import datetime
import schedule
from flask import Flask, jsonify, request

# Configurar Supabase
SUPABASE_URL = "https://qrrlhwbbujhugpynejhi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFycmxod2JidWpodWdweW5lamhpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjcwMzExMjMsImV4cCI6MjA0MjYwNzEyM30.3-TYjjI2jRpMY2_IJu6-gNHjfI-hRkCtXBoVckhFP3o"
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# Iniciar Flask
app = Flask(__name__)

# Función para descargar el archivo CSV desde Supabase
def descargar_csv_desde_supabase():
    try:
        res = supabase_client.storage.from_("modelos").download('data.csv')
        with open('data.csv', 'wb') as f:
            f.write(res)
        return 'data.csv'
    except Exception as e:
        print(f"Error al descargar CSV desde Supabase: {e}")
        return None

# Función para descargar el archivo de predicciones desde Supabase
def descargar_predicciones():
    try:
        res = supabase_client.storage.from_("predicciones").download('predicciones.csv')
        with open('predicciones.csv', 'wb') as f:
            f.write(res)
        return 'predicciones.csv'
    except Exception as e:
        print(f"Error al descargar predicciones: {e}")
        return None

# Función para subir el archivo de predicciones modificado a Supabase
def subir_predicciones():
    try:
        with open('predicciones.csv', 'rb') as file:
            supabase_client.storage.from_("predicciones").upload('predicciones.csv', file)
    except Exception as e:
        print(f"Error al subir predicciones a Supabase: {e}")

# Verificar si la hora actual está entre las 5:00 AM y las 9:55 PM
def dentro_del_horario():
    hora_actual = datetime.now().time()
    hora_inicio = datetime.strptime("05:00", "%H:%M").time()
    hora_fin = datetime.strptime("21:55", "%H:%M").time()
    return hora_inicio <= hora_actual <= hora_fin

# Crear el modelo
def crear_modelo(input_shape):
    modelo = Sequential()
    modelo.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(1))  # Predecir el siguiente número
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

# Función para entrenar el modelo
def entrenar_modelo():
    try:
        # Descargar el archivo CSV desde Supabase
        csv_file = descargar_csv_desde_supabase()
        if not csv_file:
            print("Error: no se pudo descargar el archivo CSV.")
            return
        
        datos = pd.read_csv(csv_file)
        
        # Usamos solo las columnas 'number' y 'color'
        datos['color'] = datos['color'].map({'black': 0, 'red': 1, 'green': 2})  # Mapear colores a números
        X = datos[['number', 'color']].values  # Características (X)
        y = datos['number'].shift(-1).fillna(0).values  # Predicción del siguiente número (y)
        
        # Crear y entrenar el modelo
        modelo = crear_modelo(X.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        modelo.fit(X, y, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        
        # Guardar el modelo localmente y luego subirlo a Supabase
        modelo.save('modelo_entrenado.h5')
        with open('modelo_entrenado.h5', 'rb') as file:
            supabase_client.storage.from_("modelos").upload('modelo_entrenado.h5', file)
        
        print("Entrenamiento completado y modelo guardado en Supabase.")
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")

# Hacer una predicción
def predecir(modelo, ultimo_numero, ultimo_color):
    try:
        X_pred = np.array([[ultimo_numero, ultimo_color]])  # Crear la entrada con el último número y color
        prediccion = modelo.predict(X_pred)
        return prediccion[0][0]
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return None

# Guardar las predicciones en Supabase
def guardar_prediccion(prediccion, real):
    try:
        pred_file = 'predicciones.csv'
        res = supabase_client.storage.from_("predicciones").download(pred_file)
        
        with open(pred_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{prediccion},{real}\n")
        
        # Subir el archivo actualizado de vuelta a Supabase
        subir_predicciones()
    except Exception as e:
        print(f"Error al guardar la predicción: {e}")

# Función para cargar el modelo desde Supabase
def cargar_modelo():
    from tensorflow.keras.models import load_model
    try:
        # Descargar el modelo desde Supabase
        res = supabase_client.storage.from_("modelos").download('modelo_entrenado.h5')
        with open('modelo_entrenado.h5', 'wb') as f:
            f.write(res)
        
        modelo = load_model('modelo_entrenado.h5')
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        modelo = None
    return modelo

# Predicción cada 5 minutos dentro del rango de 5:00 AM a 9:55 PM
def prediccion_cada_5_minutos():
    modelo = cargar_modelo()
    if modelo is None:
        return  # Si no hay modelo, salir de la función

    while True:
        if dentro_del_horario():
            try:
                # Descargar el archivo CSV de Supabase y cargar los datos
                csv_file = descargar_csv_desde_supabase()
                if not csv_file:
                    print("Error al descargar el CSV.")
                    continue
                
                datos = pd.read_csv(csv_file)
                
                # Usar los últimos 9 números y colores para la predicción
                ultimo_numero = datos['number'].iloc[-9:].mean()
                ultimo_color = datos['color'].map({'black': 0, 'red': 1, 'green': 2}).iloc[-9:].mean()
                
                # Hacer la predicción
                prediccion = predecir(modelo, ultimo_numero, ultimo_color)
                if prediccion is None:
                    print("No se pudo realizar la predicción.")
                    continue
                
                print(f"Predicción realizada: {prediccion}")
                
                # Obtener el número real desde el archivo CSV (última fila de la columna 'number')
                numero_real = datos['number'].iloc[-1]
                
                # Guardar la predicción y el resultado real
                guardar_prediccion(prediccion, numero_real)
            except Exception as e:
                print(f"Error durante la predicción: {e}")
        
        # Dormir por 5 minutos
        time.sleep(300)

# Ruta para obtener las predicciones realizadas
@app.route('/predicciones', methods=['GET'])
def obtener_predicciones():
    predicciones_file = descargar_predicciones()
    if not predicciones_file:
        return jsonify({'message': 'No se pudo descargar el archivo de predicciones'}), 500
    
    predicciones = pd.read_csv(predicciones_file)
    predicciones_json = predicciones.to_dict(orient='records')
    
    return jsonify(predicciones_json)

# Ruta para editar una predicción (indicar si fue correcta o incorrecta)
@app.route('/editar-prediccion', methods=['POST'])
def editar_prediccion():
    try:
        data = request.get_json()
        
        # Recibir los datos enviados en el request
        timestamp = data.get('timestamp')
        fue_correcto = data.get('fue_correcto')
        
        # Descargar el archivo de predicciones
        predicciones_file = descargar_predicciones()
        if not predicciones_file:
            return jsonify({'message': 'No se pudo descargar el archivo de predicciones'}), 500
        
        predicciones = pd.read_csv(predicciones_file)
        predicciones['timestamp'] = pd.to_datetime(predicciones['timestamp'])
        
        # Buscar la fila con el timestamp dado
        prediccion_a_editar = predicciones[predicciones['timestamp'] == pd.to_datetime(timestamp)]
        
        if not prediccion_a_editar.empty:
            if 'correcta' not in predicciones.columns:
                predicciones['correcta'] = None
            
            predicciones.loc[predicciones['timestamp'] == pd.to_datetime(timestamp), 'correcta'] = fue_correcto
            predicciones.to_csv('predicciones.csv', index=False)
            subir_predicciones()
            return jsonify({'message': 'Predicción actualizada correctamente'}), 200
        else:
            return jsonify({'message': 'No se encontró una predicción con ese timestamp'}), 404
    except Exception as e:
        return jsonify({'message': f'Error al editar la predicción: {e}'}), 500

# Programar el reentrenamiento diario a las 12:00 AM
def programar_reentrenamiento_diario():
    schedule.every().day.at("00:00").do(entrenar_modelo)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)
