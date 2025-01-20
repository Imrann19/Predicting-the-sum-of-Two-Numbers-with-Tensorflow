import tensorflow as tf
import random

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

model.fit(x_train, y_train, epochs=10, verbose=0)

X = random.randint(1, 1000)

print(model.predict([X])) 
print(f"\nLe nombre réel est {X} et le nombre à prédire était : {X*2}" )