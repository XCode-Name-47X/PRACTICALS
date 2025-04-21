import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
X_sepal = X[:, :2]  # Only use sepal features for plotting

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test, X_train_sepal, X_test_sepal = train_test_split(
    X, y, X_sepal, test_size=0.3, random_state=42
)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Prepare for iterative training and evaluation
epochs = 100
subset_sizes = np.linspace(10, len(X_train_std), epochs, dtype=int)

# Store results
losses = []

# Train the model incrementally (subset sizes increase with each epoch)
for size in subset_sizes:
    gnb = GaussianNB()  # Create Gaussian Naive Bayes model
    gnb.fit(X_train_std[:size], y_train[:size])  # Fit on a subset of the data
    y_prob = gnb.predict_proba(X_test_std)  # Get predicted probabilities
    losses.append(log_loss(y_test, y_prob))  # Compute log loss

# Plotting results
plt.figure(figsize=(14, 6))

# 1. Sepal feature scatter plot (train data)
plt.subplot(1, 2, 1)
colors = ['blue', 'red', 'green']
for i, color in enumerate(colors):  # Loop through classes and colors
    plt.scatter(
        X_train_sepal[y_train == i, 0], X_train_sepal[y_train == i, 1],
        c=color, label=iris.target_names[i]
    )
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Training Set (Sepal Features)")
plt.legend()

# 2. Plot: Loss vs. Epoch
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Negative Log-Likelihood")
plt.title("Loss vs. Epoch")

# Adjust layout and show plots
plt.tight_layout()
plt.show()
