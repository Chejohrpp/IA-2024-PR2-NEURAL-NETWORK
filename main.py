# implement a log or something to see the procedure

#arguments
    # cant entry/inputs and exit/outputs
    # cant hidden layers 
    # cant neurons per layer
    # define function activation for the hidden layers (sigmoid and tanh) default activation is sigmoid
    # define function activation for the exit layers (identity and step function) default activation step

import numpy as np
from neuralhrplib.neuralnetwork import NeuralNetwork

#Xor
nn = NeuralNetwork([2,5,1], activations='sigmoid', activations_output='step')
X = np.array([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
y = np.array([0, 1, 1, 0])

nn.training(X, y, epochs=10000, learning_rate=0.1, log_epoch_rate=1000)

for e in X:
    print("Inputs:",e,"Salidas:",nn.predict(e)[0])

import matplotlib.pyplot as plt


# Obtener las pérdidas del modelo entrenado
losses = nn.get_losses()
# Asegurarse de que las pérdidas sean una lista de valores unidimensionales
valores = [loss for loss in losses if np.isscalar(loss)]


plt.plot(range(len(valores)), valores, color='b')
plt.ylim([0, 1])
plt.ylabel('Average Magnitude of Deltas')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()


