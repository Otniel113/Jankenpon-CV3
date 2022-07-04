# Jankenpon-CV3
Klasifikasi gambar gunting, batu, kertas menggunakan deep learning Convolutional Neural Network (CNN)

## Dataset
Dataset didapat dari Drgfreeman di Kaggle https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors dengan banyak data sebagai berikut:

| Jenis Gambar | Banyaknya |
| -- | -- |
| Batu | 726 |
| Gunting | 750 |
| Kertas | 712 |

Semua ukuran gambar sama yaitu 300x200 dengan format .png data juga dipisah menjadi data train dan data test dengan perbandingan 80:20.

## Ekstraksi Fitur
Gambar dilakukan scaled dengan membagi dengan 255 agar range nilainya 0-1 (dari yang awalnya menggunakan RGB dengan nilai 0-255 masing-masing). Ekstraksi fitur dilakukan sendiri dengan menggunakan konvolusi pada CNN.

## Model
Dibangun 3 arsitektur CNN dengan spesifikasi sebagai berikut:
```python
# CNN1
model = Sequential([
    Conv2D(4, (3,3), activation='relu', input_shape=X_train[0].shape),
    MaxPooling2D(2, 2),
    Conv2D(8, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='sigmoid')
])

# CNN2
model2 = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=X_train[0].shape),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='sigmoid')
])

# CNN3
model3 = Sequential([
    Conv2D(4, (3,3), padding='same', activation='relu', input_shape=X_train[0].shape),
    MaxPooling2D(2,2),
    Conv2D(8, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(16, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

```

Dan juga parameter yang lain dapat dilihat sebagai berikut:

| Nama parameter | Nilai parameter |
| -- | -- |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Metrik | Akurasi |
| Banyak Epoch | 10 |

## Hasil dan Perbandingan dengan Sebelumnya

| Repository | Model | Akurasi Data Train | Akurasi Data Test |
| -- | -- | -- | -- |
| [Jankenpon-CV](https://github.com/Otniel113/Jankenpon-CV) | MLP tanpa ekstraksi fitur | 0.642 | 0.342 |
| [Jankenpon-CV2](https://github.com/Otniel113/Jankenpon-CV2) | MLP dengan HOG | 0.948 | 0.965 |
| [Jankenpon-CV3](https://github.com/Otniel113/Jankenpon-CV3) | CNN1 | 0.981 | 0.917 |
| [Jankenpon-CV3](https://github.com/Otniel113/Jankenpon-CV3) | CNN2 | 0.991 | 0.956 |
| [Jankenpon-CV3](https://github.com/Otniel113/Jankenpon-CV3) | CNN3 | <b>0.994</b> | <b>0.972</b> |
