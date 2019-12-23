import tensorflow as tf
import os
import re
import numpy as np

from tensorflow import keras

print(tf.__version__)
path_name = "linux_kernel/"

## concat all files into training file
with open("linux_kernel_training.txt", "w") as a:
    for file in os.listdir(path_name):
        f = os.path.join(path_name, file)
        current_file = open(f).read()
        a.write(current_file)

text = open("linux_kernel_training.txt", 'r').read()
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
print(f"Length of file: {len(text)}")
print(f"Total vocab length: {VOCAB_SIZE}")

# **Mapping of unique chars to integers and a reverse mapping**
char_to_int = {c:i for i, c in enumerate(chars)}
int_to_char = {i:c for i, c in enumerate(chars)}

# **In this example we would use character based model**
remove_comments_patterns = r'((\s?\*(\s+|\t+)\w+\s)|(\*\/)|(\/\/)|(\/\*)).*'
remove_comments_patterns = r'(\/\s?\*.*(\n|\s+|\t+)\s+\*\s+\w+.*\n\s\*\/)|(\/\/.*)'
text_without_comment = re.sub(remove_comments_patterns, '', text)

SEQ_LENGTH = 200
EPOCHES = 10
BATCH_SIZE = 256

X = []
y = []
for i in range(len(text_without_comment) - SEQ_LENGTH):
    seq_in = text[i:i + SEQ_LENGTH]
    seq_out = text[i + SEQ_LENGTH] # we try to predict next character
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])

print(''.join([int_to_char[x] for x in X[12345]]))
print(f"Next character: {int_to_char[y[12345]]}")

samples = len(X)
print(f"Total samples: {samples}")

X = np.reshape(X, (samples, SEQ_LENGTH, 1))
X = X / float(VOCAB_SIZE) # normalization
y = keras.utils.to_categorical(y)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# **Create simple model with 2 LSTMs**
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    keras.layers.Dropout(0.3),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(y.shape[1], activation='softmax') # output size => character dictionary
])

### Save model ararchitecture
with open('model_architecture.txt', 'w') as f:
    f.write("""
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
        keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(128)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(98, activation='softmax')
    ])
    """)

model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_accuracy', min_delta=0.0001,
  patience=3)

history = model.fit(X, 
                    y, 
                    epochs=EPOCHES, 
                    batch_size=BATCH_SIZE, 
                    verbose=2, 
                    callbacks=[checkpoint_callback, earlystop_callback])