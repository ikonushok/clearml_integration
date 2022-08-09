**Ссылки**
- Гайд по установке clearml: https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps
- Гайд по установке сервера clearml на Windows 10: https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_win/
- Гайд по созданию воркера, обрабатывающего очередь: https://clear.ml/docs/latest/docs/clearml_agent
- Гайд по загрузке данных на сервер: https://clear.ml/docs/latest/docs/clearml_data/clearml_data
- Гайд по деплою: https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving/
- Плейлист с туториалами: https://youtube.com/playlist?list=PLMdIlCuMqSTnoC45ME5_JnsJX0zWqDdlO
- Примеры работы с clearml: https://github.com/allegroai/clearml-serving/tree/main/examples
- Примеры работы с деплойем: https://github.com/allegroai/clearml-serving/tree/main/examples

**Функциональность**
- Запуск эксперимента из UI в браузере

![image](./Images/5.png)
![image](./Images/6.png)
![image](./Images/7.png)
- Изменение параметров эксперимента через UI в браузере
![image](./Images/8.png)
- Версионирование данных (можно так же архитектуры через файлы передавать, поэтому есть полный контроль над версиями данных и архитектурами нейронок), гайд по работе с данными тут: https://www.youtube.com/watch?v=mQtCTnmvf3A&list=PLMdIlCuMqSTnoC45ME5_JnsJX0zWqDdlO&index=9
- Сохранение кода, использованного при запуске
![image](./Images/9.png)
![image](./Images/10.png)
![image](./Images/11.png)
- Разбиение кода на несколько частей и запуск в виде пайплайна
- Кэширование выхода отдельных частей пайплайна (позволяет запускать пайплайн частями при внесении изменений), для этого необходимо выставить параметр cache_executed_step=True при добавлении ступени пайплайна
- Отрисовка графиков в UI в браузере (scalar для графиков через логгер и plots для графиков через mpl)
![image](./Images/3.png)
![image](./Images/4.png)
- Просмотр отдельных ступеней пайплайна
![image](./Images/1.png)
![image](./Images/2.png)
- Просмотр вывода в консоль
![image](./Images/9.png)
![image](./Images/10.png)
![image](./Images/12.png)
Либо
![image](./Images/13.png)
- Выход отдельной ступени
![image](./Images/14.png)
- Сравнение результатов
![image](./Images/15.png)
![image](./Images/16.png)
![image](./Images/17.png)
- Hyperparameter sweep
![image](./Images/18.png)
