Проект представляет собой реализацию различных алгоритмов распознавания цифр по изображениям с помощью машинного обучения.


Результаты алгоритмов представлены в приложении PyQt5 (папка application).
![image](https://github.com/svadim17/MachineLearning/assets/100597068/2d356946-b74f-47b3-8fbf-ec0cdea29382)

Папка **Literature** содержит условия выполненных заданий.

**Используемый датасет:** notMNIST (модели обучались на датасете notMNISTsmall: 18724 изображения)

**Реализованные алгоритмы:**
  - Простейший классификатор с помощью Логистической регрессии
  - Полносвязная нейронная сеть
  - Полносвязная нейронная сеть с регуляризацией и методом сброса нейронов (dropout)
  - Полносвязная нейронная сеть с регуляризацией и методом сброса нейронов (dropout) с использованием динамически изменяемой скоростью обучения (learning rate)
  - Сверточная нейронная сеть
  - Нейронная сеть со слоями, реализующие операцию пулинга (Pooling) с функцией среднего
  - Нейронная сеть со слоями, реализующие операцию пулинга (Pooling) с функцией максимума
  - Сверточная нейронная сеть с архитектурой LeNet-5
Сравнения результатов модели представлены в приложении во вкладке Conclusion.

Все реализованные модели сохранены в папке **saved_models**

**TO RUN:** Запустить файл проекта main.py, после этого загрузится приложение.
