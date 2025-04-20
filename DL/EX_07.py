import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown dog jumps over the lazy cat.",
    "The speedy black cat jumps over the lazy dog."
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
window = 2
contexts, targets = [], []
for seq in sequences:
    for i in range(window, len(seq) - window):
        ctx = seq[i - window:i] + seq[i + 1:i + window + 1]
        tgt = seq[i]
        contexts.append(ctx)
        targets.append(tgt)
X = np.array(contexts)
y = tf.keras.utils.to_categorical(targets, num_classes=len(tokenizer.word_index) + 1)
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=2 * window),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=100, verbose=0)
embeddings = model.get_weights()[0]
reduced = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(6, 6))
for word, idx in tokenizer.word_index.items():
    x, y = reduced[idx]
    plt.scatter(x, y)
    plt.annotate(word, (x, y))
plt.title("CBOW Word Embeddings (2D)")
plt.show()
