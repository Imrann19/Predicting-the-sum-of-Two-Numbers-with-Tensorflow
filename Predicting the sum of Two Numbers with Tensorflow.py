import tensorflow as tf 
import numpy as np
import random

#Generation des donnes d'entrainement/Generating training data
x = np.random.randint(0, 100, (1000, 2))
y = np.sum(x, axis=1)

#Normalisation des donnes d'entrainement/Normalizing training data
x = x/100.0
y = y/100.0

#Construction du modele/Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation=None),
])

#Compilation du modele/Compiling the model
model.compile(optimizer='adam', loss='mse')

#L'entrainement du modele/Training of the model
print("L'entrainement commence")
model.fit(x, y, epochs=35, batch_size=32)
print("L'entrainement est termine")

#Generation des donnes d'entrainement/Generating training data
x_test = np.random.randint(0, 100, (1000, 2))
y_test = x_test[:, 0] + x_test[:, 0]

#Normalisation des donnes de test/Normalizing test data
x_test = x_test/100.0
y_test = y_test/100.0

#Le modele fait des prédictions a partir des donnés de test/Making predictions on the test data
y_pred = model.predict(x_test)
y_pred = y_pred.flatten()

#Denormalization de la reponse et la prediction du modele/Denormalizing of the awnsers and the model prediction
y_pred = y_pred*100
y_test = y_test*100

diff = y_test-y_pred #La difference entre la vrai reponse et ce que le modele a predit/Calculating the difference between the awnsers and the model predictions 
mse = np.mean(y_pred-y_test)**2 #Moyenne de la difference entre la prediction du modele et la vrai reponse au carre/Mean of the difference between the model prdictions and the awnsers squared 

#Resultats/Results
print(f"Le modele a predit: {y_pred}")
print(f"La vrai reponse est: {y_test}")
print(f"La difference est {diff}")
print(f"La moyenne est: {mse}")
