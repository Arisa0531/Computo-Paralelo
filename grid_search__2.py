import pandas as pd
import itertools
import threading
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Cargar y preprocesar la base de datos
file_path = r'C:\Users\Ariadna\Downloads\winequality-white-clean.csv'
df = pd.read_csv(file_path)

# Separar características (X) y la etiqueta (y)
X = df.drop(columns=['quality'])
y = df['quality']

# Escalar los datos para mejorar el rendimiento del SVC
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

# 2. Definir los hiperparámetros para SVC
param_grid_svc = {
    'C': [0.1, 1, 10],         # Parámetro de regularización
    'kernel': ['linear', 'rbf'],  # Tipos de kernel (lineal o no lineal)
    'gamma': ['scale', 'auto']    # Coeficiente para los kernels no lineales
}

# Generar todas las combinaciones posibles de hiperparámetros
keys_svc, values_svc = zip(*param_grid_svc.items())
combinations_svc = [dict(zip(keys_svc, v)) for v in itertools.product(*values_svc)]

# 3. Nueva función para nivelar las cargas
def nivelacion_cargas_mejorada(D, n_p):
    """
    Esta función divide la lista D en n_p partes de manera equilibrada.
    """
    # Determinar el tamaño de cada segmento
    k, m = divmod(len(D), n_p)
    return [D[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_p)]

# 4. Función para evaluar cada conjunto de hiperparámetros en paralelo usando validación cruzada
def evaluate_set(hyperparameter_set, X_train, y_train, results, idx):
    """
    Función que evalúa un conjunto de hiperparámetros utilizando SVC
    con validación cruzada en los datos de entrenamiento.
    """
    best_accuracy = 0
    best_params = None

    # Creamos una copia de los datos en cada hilo para evitar problemas de escritura
    X_train_local = X_train.copy()
    y_train_local = y_train.copy()

    for params in hyperparameter_set:
        clf = SVC(**params)
        # Usar validación cruzada en el conjunto de entrenamiento
        scores = cross_val_score(clf, X_train_local, y_train_local, cv=5)  # 5-fold cross-validation
        accuracy = scores.mean()

        # Guardar la mejor precisión
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    results[idx] = (best_accuracy, best_params)

if __name__ == '__main__':
    # Número de hilos a usar (4 núcleos lógicos)
    N_THREADS = 2
    # Dividir las combinaciones de hiperparámetros en partes para cada hilo usando la nueva función
    splits = nivelacion_cargas_mejorada(combinations_svc, N_THREADS)
    
    # Crear un arreglo para almacenar los resultados de cada hilo
    results = [None] * N_THREADS
    threads = []

    # 5. Iniciar el temporizador
    start_time = time.perf_counter()

    # Crear y comenzar los hilos manualmente
    for i in range(N_THREADS):
        thread = threading.Thread(target=evaluate_set, args=(splits[i], X_train, y_train, results, i))
        threads.append(thread)
        thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    # 6. Obtener el mejor resultado
    best_accuracy = max(results, key=lambda x: x[0])[0]
    best_params = max(results, key=lambda x: x[0])[1]

    finish_time = time.perf_counter()
    
    # 7. Imprimir los resultados
    print(f"Mejor Precisión (cross-validation): {best_accuracy:.4f} con Hiperparámetros: {best_params}")
    print(f"Grid Search paralelizado completado en {finish_time - start_time:.2f} segundos")

    # Evaluar el modelo con los mejores hiperparámetros en el conjunto de prueba
    final_clf = SVC(**best_params)
    final_clf.fit(X_train, y_train)
    y_pred = final_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}")
