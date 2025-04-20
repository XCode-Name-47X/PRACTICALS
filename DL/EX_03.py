import tensorflow as tf
import numpy as np

corpus = [
    "I like playing football with my friends",
    "I enjoy playing tennis",
    "I hate swimming",
    "I love basketball"
]

window_size = 3
embedding_dim = 50
batch_size = 16
epochs = 100
learning_rate = 0.01

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(corpus)
data = []
for seq in sequences:
    for i, target in enumerate(seq):
        for j in range(max(0, i - window_size), min(len(seq), i + window_size + 1)):
            if i != j:
                context = seq[j]
                data.append([target, context])

data = np.array(data)
x_train, y_train = data[:, 0], data[:, 1]

inputs = tf.keras.layers.Input(shape=(1,))
embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
output = tf.keras.layers.Dense(vocab_size, activation='softmax')(tf.keras.layers.Flatten()(embeddings))
model = tf.keras.models.Model(inputs=inputs, outputs=output)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

word_embeddings = model.get_layer(index=1).get_weights()[0]

def get_vector(word):
    return word_embeddings[tokenizer.word_index[word]]

word = "football"
print(f"Vector representation of '{word}': {get_vector(word)}")

def get_context_words(word):
    idx = tokenizer.word_index[word]
    context_indices = list(range(max(0, idx - window_size), min(vocab_size, idx + window_size + 1)))
    return [w for w, i in tokenizer.word_index.items() if i in context_indices]

focus_word = "playing"
print(f"Context words for '{focus_word}': {get_context_words(focus_word)}")
