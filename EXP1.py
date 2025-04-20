import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Only use Sepal Length and Sepal Width for scatter plot
X_sepal = X[:, :2]

# Train-test split
X_train, X_test, y_train, y_test, X_train_sepal, X_test_sepal = train_test_split(
    X, y, X_sepal, test_size=0.3, random_state=42)

# Standardize features for training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TensorFlow Probability setup
tfd = tfp.distributions
num_classes = 3
num_features = X.shape[1]

# Initialize parameters
means = tf.Variable(tf.random.normal([num_classes, num_features]))
scales = tf.Variable(tf.random.normal([num_classes, num_features], mean=1.0, stddev=0.5))
priors = tf.constant([np.mean(y_train == i) for i in range(num_classes)], dtype=tf.float32)

# Convert to tensor
X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)

# Tracking metrics
epochs = 10000
loss_history = []
means_history = []
scales_history = []

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.05)

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = 0
        for i in range(num_classes):
            dist = tfd.Normal(loc=means[i], scale=tf.math.softplus(scales[i]))
            log_prob = tf.reduce_sum(dist.log_prob(X_train_tf), axis=1)
            mask = tf.cast(tf.equal(y_train_tf, i), tf.float32)
            loss += -tf.reduce_sum(mask * (log_prob + tf.math.log(priors[i] + 1e-8)))

    gradients = tape.gradient(loss, [means, scales])
    optimizer.apply_gradients(zip(gradients, [means, scales]))

    # Save history
    loss_history.append(loss.numpy())
    means_history.append(means.numpy())
    scales_history.append(tf.math.softplus(scales).numpy())

# Convert to numpy arrays
means_history = np.array(means_history)
scales_history = np.array(scales_history)

# Prediction function
def predict(X_input):
    log_probs = []
    for i in range(num_classes):
        dist = tfd.Normal(loc=means[i], scale=tf.math.softplus(scales[i]))
        log_likelihood = tf.reduce_sum(dist.log_prob(X_input), axis=1)
        log_prior = tf.math.log(priors[i] + 1e-8)
        log_probs.append(log_likelihood + log_prior)
    log_probs = tf.stack(log_probs, axis=1)
    return tf.argmax(log_probs, axis=1).numpy()

# Predict on test set
y_pred = predict(tf.convert_to_tensor(X_test_scaled, dtype=tf.float32))

# =======================================
# Plot all graphs (as shown in your image)
# =======================================

plt.figure(figsize=(14, 12))

# Top plot: Sepal Length vs Sepal Width
plt.subplot(2, 2, 1)
colors = ['blue', 'red', 'green']
labels = ['Setosa', 'Versicolour', 'Virginica']
for i in range(num_classes):
    idx = y_train == i
    plt.scatter(X_train_sepal[idx, 0], X_train_sepal[idx, 1], c=colors[i], label=labels[i])
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Training set')
plt.legend()

# Bottom left: Loss vs Epoch
plt.subplot(2, 2, 2)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Negative log likelihood')
plt.title('Loss vs. epoch')

# Bottom middle: Means vs Epoch
plt.subplot(2, 2, 3)
for c in range(num_classes):
    for f in range(num_features):
        plt.plot([m[c, f] for m in means_history])
plt.xlabel('Epoch')
plt.ylabel('Means')
plt.title("ML estimates for model's means vs. epoch")

# Bottom right: Scales vs Epoch
plt.subplot(2, 2, 4)
for c in range(num_classes):
    for f in range(num_features):
        plt.plot([s[c, f] for s in scales_history])
plt.xlabel('Epoch')
plt.ylabel('Scales')
plt.title("ML estimates for model's scales vs. epoch")

plt.tight_layout()
plt.show()

# =======================================
# Sample Output
# =======================================
print("\nSample Predictions:")
for i in range(5):
    print(f"Sample {i+1}: True = {class_names[y_test[i]]}, Predicted = {class_names[y_pred[i]]}")
