import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_sepal = X[:, :2]
X_train, X_test, y_train, y_test, X_train_sepal, X_test_sepal = train_test_split(X, y, X[:, :2], test_size=0.3)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
X_train_tf = tf.convert_to_tensor(X_train_std, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train)

tfd = tfp.distributions
classes, features = 3, 4
means = tf.Variable(tf.random.normal([classes, features]))
scales = tf.Variable(tf.ones([classes, features]))
priors = tf.constant([np.mean(y_train == i) for i in range(classes)], dtype=tf.float32)
opt = tf.optimizers.Adam(0.05)

losses, means_hist, scales_hist = [], [], []

for _ in range(5000):
    with tf.GradientTape() as tape:
        loss = 0
        for i in range(classes):
            dist = tfd.Normal(means[i], tf.math.softplus(scales[i]))
            logp = tf.reduce_sum(dist.log_prob(X_train_tf), axis=1)
            mask = tf.cast(y_train_tf == i, tf.float32)
            loss += -tf.reduce_sum(mask * (logp + tf.math.log(priors[i])))
    grads = tape.gradient(loss, [means, scales])
    opt.apply_gradients(zip(grads, [means, scales]))
    losses.append(loss.numpy())
    means_hist.append(means.numpy())
    scales_hist.append(tf.math.softplus(scales).numpy())

def predict(X):
    probs = []
    for i in range(classes):
        dist = tfd.Normal(means[i], tf.math.softplus(scales[i]))
        logp = tf.reduce_sum(dist.log_prob(X), axis=1) + tf.math.log(priors[i])
        probs.append(logp)
    return tf.argmax(tf.stack(probs, axis=1), axis=1).numpy()

y_pred = predict(tf.convert_to_tensor(X_test_std, dtype=tf.float32))

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
for i, label in enumerate(['Setosa', 'Versicolour', 'Virginica']):
    plt.scatter(X_train_sepal[y_train == i, 0], X_train_sepal[y_train == i, 1], label=label)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Training set")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Negative Log Likelihood")

plt.subplot(2, 2, 3)
for c in range(3):
    [plt.plot([m[c, f] for m in means_hist]) for f in range(features)]
plt.title("Means vs Epoch")
plt.xlabel("Epoch")

plt.subplot(2, 2, 4)
for c in range(3):
    [plt.plot([s[c, f] for s in scales_hist]) for f in range(features)]
plt.title("Scales vs Epoch")
plt.xlabel("Epoch")

plt.tight_layout()
plt.show()

for i in range(5):
    print(f"Sample {i+1}: True = {y_test[i]}, Pred = {y_pred[i]}")
