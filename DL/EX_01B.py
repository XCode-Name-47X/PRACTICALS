import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[]
)

class LossLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/1000, Loss: {logs['loss']:4f}")

print(model.summary())

model.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=0, callbacks=[LossLogger()])

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")

new_samples = np.array([[6.7, 3.3, 5.7, 2.5]])
predictions = model.predict(new_samples, verbose=0)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted Class: {predicted_class}")
