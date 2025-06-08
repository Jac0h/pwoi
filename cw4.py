# === a. Import modułów ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# === b. Wczytywanie danych ===
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# === c. Wstępne przetwarzanie danych ===
# i. Kodowanie całkowitoliczbowe
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

# ii. Kodowanie 1 z n
y_onehot = to_categorical(y_int)

# iii. Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# === d. Tworzenie modelu sieci neuronowej ===
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=72))
model.add(Dense(3, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# === e. Uczenie sieci ===
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# === f. Testowanie sieci ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Macierz pomyłek
cm = confusion_matrix(y_test_labels, y_pred)
print("Macierz pomyłek:")
print(cm)
