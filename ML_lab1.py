import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Распаковка архива
# -------------------------------
zip_path = 'notMNIST_small.zip'
extract_dir = 'notMNIST_small'

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Архив распакован.")
else:
    print("Папка с распакованными данными уже существует.")

# Двойная вложенность
classes_root = os.path.join(extract_dir, 'notMNIST_small')
classes = sorted([d for d in os.listdir(classes_root) if os.path.isdir(os.path.join(classes_root, d))])
print("Классы:", classes)

# -------------------------------
# 2. Отображение нескольких изображений
# -------------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, cls in enumerate(classes):
    cls_path = os.path.join(classes_root, cls)
    images_in_class = [f for f in os.listdir(cls_path) 
                       if os.path.isfile(os.path.join(cls_path, f)) and f.lower().endswith('.png')]
    if not images_in_class:
        continue
    img_name = images_in_class[0]
    img_path = os.path.join(cls_path, img_name)
    try:
        img = Image.open(img_path)
        axes[i//5, i%5].imshow(np.array(img), cmap='gray')
        axes[i//5, i%5].set_title(cls)
        axes[i//5, i%5].axis('off')
    except UnidentifiedImageError:
        continue
plt.show()

# -------------------------------
# 3. Баланс классов
# -------------------------------
print("\nБаланс классов:")
for cls in classes:
    cls_path = os.path.join(classes_root, cls)
    images_in_class = [f for f in os.listdir(cls_path) 
                       if os.path.isfile(os.path.join(cls_path, f)) and f.lower().endswith('.png')]
    print(f"{cls}: {len(images_in_class)} изображений")

# -------------------------------
# 4. Формируем списки всех изображений и меток
# -------------------------------
all_images = []
all_labels = []

for cls in classes:
    cls_path = os.path.join(classes_root, cls)
    images_in_class = [f for f in os.listdir(cls_path) 
                       if os.path.isfile(os.path.join(cls_path, f)) and f.lower().endswith('.png')]
    for img_name in images_in_class:
        all_images.append(os.path.join(cls_path, img_name))
        all_labels.append(cls)

print(f"Всего изображений: {len(all_images)}")

# -------------------------------
# 5. Разделение на train/val/test
# -------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    all_images, all_labels, test_size=6000, stratify=all_labels, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, train_size=10000, test_size=2000, stratify=y_temp, random_state=42)

print(f"Размеры выборок: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# -------------------------------
# 6. Функция для загрузки изображения с фильтрацией битых файлов
# -------------------------------
def load_image_array(path):
    try:
        return np.array(Image.open(path)).flatten()
    except (OSError, UnidentifiedImageError):
        return None

# -------------------------------
# 7. Удаление дубликатов в обучающей выборке
# -------------------------------
val_set = set(filter(None, map(lambda p: load_image_array(p).tobytes() if load_image_array(p) is not None else None, X_val)))
test_set = set(filter(None, map(lambda p: load_image_array(p).tobytes() if load_image_array(p) is not None else None, X_test)))

X_train_unique = []
y_train_unique = []

for i, path in enumerate(X_train):
    arr = load_image_array(path)
    if arr is None:
        continue
    arr_bytes = arr.tobytes()
    if arr_bytes not in val_set and arr_bytes not in test_set:
        X_train_unique.append(path)
        y_train_unique.append(y_train[i])

print(f"Обучающая выборка после удаления дубликатов: {len(X_train_unique)}")

# -------------------------------
# 8. Подготовка массивов для классификатора
# -------------------------------
# Преобразуем изображения в массивы и фильтруем битые файлы
def load_images_to_array(paths, labels):
    X, y = [], []
    for i, path in enumerate(paths):
        try:
            arr = np.array(Image.open(path)).flatten().astype(np.float32) / 255.0
            X.append(arr)
            y.append(labels[i])
        except (OSError, UnidentifiedImageError):
            continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

X_train_arrays, y_train_array = load_images_to_array(X_train_unique, y_train_unique)
X_val_arrays, y_val_array = load_images_to_array(X_val, y_val)

print(X_train_arrays.shape, X_train_arrays.dtype)
print(y_train_array.shape, y_train_array.dtype)
print(X_val_arrays.shape, X_val_arrays.dtype)
print(y_val_array.shape, y_val_array.dtype)

# -------------------------------
# 9. Кодируем метки в числа
# -------------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_array)
y_val_encoded = le.transform(y_val_array)

# -------------------------------
# 10. Обучение MLP с разными размерами выборки
# -------------------------------
train_sample_sizes = [50, 100, 1000, 5000, len(X_train_arrays)]
accuracies = []

for size in train_sample_sizes:
    if size < 1000:
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=200,
            random_state=42
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    clf.fit(X_train_arrays[:size], y_train_encoded[:size])
    y_pred = clf.predict(X_val_arrays)
    acc = accuracy_score(y_val_encoded, y_pred)
    accuracies.append(acc)
    print(f"Размер обучающей выборки: {size}, точность: {acc:.4f}")

plt.figure(figsize=(8,5))
plt.plot(train_sample_sizes, accuracies, marker='o')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность на валидации')
plt.title('Зависимость точности MLPClassifier от размера обучающей выборки')
plt.grid(True)
plt.show()
