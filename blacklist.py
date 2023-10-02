import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist         # библиотека базы выборок Mnist
from keras.layers import Dense, Flatten
from sklearn.metrics import classification_report, confusion_matrix


(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_new_train = copy.deepcopy(y_train)
y_new_test = copy.deepcopy(y_test)

for x in range(len(y_new_train)):
    if (y_new_train[x] != 6) and (y_new_train[x] != 0):
        y_new_train[x] = 1

for x in range(len(y_new_test)):
    if (y_new_test[x] != 6) and (y_new_test[x] != 0):
        y_new_test[x] = 1

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_new_train, 10)
y_test_cat = keras.utils.to_categorical(y_new_test, 10)

y_train_cat_new = np.vstack((y_train_cat[:,0], y_train_cat[:,1], y_train_cat[:,6])).transpose()
y_test_cat_new = np.vstack((y_test_cat[:,0], y_test_cat[:,1], y_test_cat[:,6])).transpose()

# отображение первых 25 изображений из обучающей выборки
f, a = plt.subplots(5, 5, figsize=(10,5))
f.suptitle('Первые 25 изображений из пакета MNIST')
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='sigmoid'),
    Dense(3, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


hist = model.fit(x_train, y_train_cat_new, batch_size=32, epochs=5, validation_split=0.2)

train_loss = hist.history['loss']
xc = range(5)
plt.figure()
plt.plot(xc, train_loss)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка обучения')
plt.title('Ошибка на тренировочном наборе')
plt.show()


model.evaluate(x_test, y_test_cat_new)

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )
print( np.argmax(res) )

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

for i in range(len(pred)):
    if pred[i] == 2:
        pred[i] = 6

print(pred.shape)

print('Ответы\n', y_test[:30])
print('Предсказания\n', pred[:30])

# Выделение неверных вариантов
mask = pred == y_new_test
print(mask[:30])


x_false = x_test[~mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов
f, a = plt.subplots(figsize=(10, 5))
f.suptitle('На каких картинках ошиблись')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

target_names = ['0', 'Остальные цифры', '6']
print(classification_report(y_new_test, pred, target_names=target_names))
print(confusion_matrix(y_new_test, pred))

plt.show()



from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
