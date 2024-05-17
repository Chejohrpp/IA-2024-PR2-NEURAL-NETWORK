## NeuralHRPLib: Una Librería de Redes Neuronales

### Descripción

NeuralHRPLib es una librería simple de redes neuronales diseñada para proporcionar una fácil comprensión y manipulación de los modelos de redes neuronales. La librería permite la creación, entrenamiento y evaluación de modelos de redes neuronales multicapa con diferentes funciones de activación.

### Instalación

Para utilizar NeuralHRPLib, necesitas tener instalado Python junto con algunas dependencias. A continuación se describe el proceso de instalación:

#### Dependencias

- **NumPy**: Biblioteca fundamental para la computación científica en Python.
- **Matplotlib**: Biblioteca para crear gráficos en Python.

#### Instrucciones de Instalación

1. Asegúrate de tener Python instalado. Puedes descargarlo desde [python.org](https://www.python.org/).
2. Instala las dependencias utilizando pip:

    ```bash
    pip install numpy matplotlib
    ```

3. Descarga el código fuente de NeuralHRPLib y guárdalo en tu proyecto.

### Configuración y Uso

A continuación, se detallan las opciones de configuración y uso de la librería NeuralHRPLib.

#### Inicialización del Modelo

Puedes inicializar un modelo de red neuronal especificando las capas y las funciones de activación. Aquí tienes un ejemplo:

```python
from neuralhrplib import NeuralNetwork

# Definir la arquitectura de la red
layers = [2, 3, 1]  # 2 neuronas de entrada, 3 en la capa oculta, 1 en la capa de salida
activations = 'sigmoid'  # Función de activación para las capas ocultas
activations_output = 'step'  # Función de activación para la capa de salida

# Crear el modelo de red neuronal
nn = NeuralNetwork(layers, activations, activations_output)
```

#### Opciones de Configuración

- **`layers`**: Una lista de enteros que especifica el número de neuronas en cada capa.
- **`activations`**: Función de activación para las capas ocultas. Opciones:
  - `'sigmoid'`
  - `'tanh'`
- **`activations_output`**: Función de activación para la capa de salida. Opciones:
  - `'step'`
  - `'identity'`

#### Entrenamiento del Modelo

Puedes entrenar el modelo con tus datos utilizando el método `training`:

```python
# Datos de entrada (X) y etiquetas (targets)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Entrenar el modelo
nn.training(X, targets, epochs=2000, learning_rate=0.1)
```

Parámetros del método `training`:

- **`X`**: Datos de entrada.
- **`targets`**: Etiquetas de los datos de entrada.
- **`epochs`**: Número de épocas para el entrenamiento.
- **`learning_rate`**: Tasa de aprendizaje para el entrenamiento.
- **`log_epoch_rate`**: Muestreo de epochs de la forma de entrenamiento

#### Predicción

Una vez que el modelo está entrenado, puedes realizar predicciones con el método `predict`:

```python
# Realizar una predicción
prediction = nn.predict([1, 0])
print(prediction)
```

#### Visualización de las Pérdidas

Puedes visualizar las pérdidas durante el entrenamiento utilizando `matplotlib`:

```python
import matplotlib.pyplot as plt

# Obtener las pérdidas
losses = nn.get_losses()

# Graficar las pérdidas
plt.plot(range(len(losses)), losses, color='b')
plt.ylim([0, 1])
plt.ylabel('Average Magnitude of Deltas')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
```

### Estructura del Código

1. **Funciones de Activación y sus Derivadas**: Incluye funciones como `sigmoid`, `tanh`, `step_function`, `identity_function` y sus derivadas correspondientes.

2. **Clase `NeuralNetwork`**:
   - **`__init__`**: Inicializa los pesos y sesgos, así como las funciones de activación.
   - **`get_weighted_sum`**: Calcula la suma ponderada de las entradas.
   - **`cross_entropy`**: Calcula la pérdida utilizando la función de entropía cruzada.
   - **`update_weights`**: Actualiza los pesos durante el entrenamiento.
   - **`update_bias`**: Actualiza los sesgos durante el entrenamiento.
   - **`training`**: Método principal para entrenar la red neuronal.
   - **`predict`**: Realiza predicciones con el modelo entrenado.
   - **`get_losses`**: Devuelve las pérdidas almacenadas durante el entrenamiento.

### Ejemplo Completo

Aquí tienes un ejemplo completo que muestra cómo usar NeuralHRPLib desde la inicialización hasta la visualización de las pérdidas:

```python
import numpy as np
import matplotlib.pyplot as plt
from neuralhrplib import NeuralNetwork

# Definir la arquitectura de la red
layers = [2, 3, 1]
activations = 'sigmoid'
activations_output = 'step'

# Crear el modelo de red neuronal
nn = NeuralNetwork(layers, activations, activations_output)

# Datos de entrada (X) y etiquetas (targets)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Entrenar el modelo
nn.training(X, targets, epochs=2000, learning_rate=0.1)

# Realizar una predicción
prediction = nn.predict([1, 0])
print(f'Predicción: {prediction}')

# Obtener las pérdidas
losses = nn.get_losses()

# Graficar las pérdidas
plt.plot(range(len(losses)), losses, color='b')
plt.ylim([0, 1])
plt.ylabel('Average Magnitude of Deltas')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
```

### Conclusión

NeuralHRPLib es una herramienta versátil y fácil de usar para el entrenamiento y la evaluación de redes neuronales. Con una configuración sencilla y funciones bien definidas, permite a los usuarios experimentar y entender los conceptos fundamentales de las redes neuronales.
