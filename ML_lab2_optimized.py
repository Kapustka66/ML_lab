# ML_lab2_optimized.py
import os
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# ---------------------------
# 1. Распаковка zip
# ---------------------------
zip_path = "notMNIST_large.zip"
extract_folder = "notMNIST_large"

if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

# ---------------------------
# 2. Автоопределение папки с буквами
# ---------------------------
letters_set = set("ABCDEFGHIJ")
for root, dirs, files in os.walk(extract_folder):
    if set(dirs) >= letters_set:
        data_folder = root
        break
print("Используемая папка с буквами:", data_folder)

# ---------------------------
# 3. Загрузка изображений
# ---------------------------
def load_images(folder):
    images = []
    labels = []
    label_dict = {letter: idx for idx, letter in enumerate("ABCDEFGHIJ")}
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                try:
                    img_path = os.path.join(path, file)
                    img = Image.open(img_path).convert("L")
                    img = np.array(img) / 255.0
                    images.append(img)
                    labels.append(label_dict[label])
                except:
                    pass
    return np.array(images), np.array(labels)

X, y = load_images(data_folder)
X = X.reshape(-1, 28*28)
y = to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Данные загружены:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# ---------------------------
# 4. Логистическая регрессия для сравнения
# ---------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, np.argmax(y_train, axis=1))
y_pred = clf.predict(X_test)
logreg_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print("Logistic Regression accuracy:", logreg_acc)

# ---------------------------
# 5. Настройки нейросети
# ---------------------------
hidden_layers = [512, 256, 128]  # можно менять от 1 до 5 слоев
activation = 'relu'
dropout_rate = 0.5
l2_reg = 0.001
epochs = 30
batch_size = 128

# ---------------------------
# 6. Learning rate scheduler
# ---------------------------
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch:
        return lr * 0.5
    return lr

lr_callback = LearningRateScheduler(scheduler)

# ---------------------------
# 7. Создание модели
# ---------------------------
model = Sequential()
model.add(Input(shape=(28*28,)))
for neurons in hidden_layers:
    model.add(Dense(neurons, activation=activation, kernel_regularizer=l2(l2_reg)))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='softmax'))

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 8. Обучение модели
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[lr_callback]
)

# ---------------------------
# 9. Оценка модели
# ---------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Deep NN test accuracy:", test_acc)
print(f"Улучшение по сравнению с логистической регрессией: {test_acc - logreg_acc:.4f}")

# ---------------------------
# 10. График точности
# ---------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train/Validation Accuracy')
plt.show()
