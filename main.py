from clearml import PipelineController

#первая ступень- подготовка данных
def prepare_data(calc_key_points_file_id, window_size, dataset_id):
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm
    from datetime import datetime, timedelta
    import pytz
    from clearml import Dataset
    import importlib.util
    import sys
    window_size = int(window_size)
    utc = pytz.utc
    et_tz = pytz.timezone('US/Eastern')
    dataset_folder = Dataset.get(dataset_id=dataset_id).get_local_copy()

    # чтение данных
    minutes_df = pd.read_csv(f'{dataset_folder}/minutes_df.csv', sep=';')
    levels_df = pd.read_csv(f'{dataset_folder}/levels_df.csv', sep=';')
    minutes_df['Date'] = pd.to_datetime(minutes_df['Date'])
    levels_df['Date'] = pd.to_datetime(levels_df['Date'])
    minutes_df = minutes_df.set_index('Date')
    levels_df = levels_df.set_index('Date')
    #округление триггеров до 1e-1
    levels_df['high_trigger'] = levels_df['high_trigger'].round(1)
    levels_df['low_trigger'] = levels_df['low_trigger'].round(1)
    minutes_df['high_trigger'] = minutes_df['high_trigger'].round(1)
    minutes_df['low_trigger'] = minutes_df['low_trigger'].round(1)
    #поиск первых точек триггера и тех из них, которые являются точками пробоя
    trigger_points = []
    breakout_points = []
    for low, high in tqdm(list(zip(levels_df.index[levels_df['peaks_low'] == -1][:-1],
                                   levels_df.index[levels_df['peaks_low'] == -1][1:]))):
        tmp = minutes_df.loc[levels_df.loc[low:].index[levels_df.offsets_low.loc[low]]:high].iloc[:-1] #промежуток между началами двух уровней
        triggers = tmp.index[tmp.trigger == 2] #все триггеры на данном промежутке
        if len(triggers) == 0:
            continue
        if minutes_df.loc[triggers[0]].rebound != 2:
            breakout_points.append(triggers[0]) #если первый триггер является точкой пробоя, сохраняем его как пробой
        trigger_points.append([low, triggers[0]]) #сохраняем первый триггер
    trigger_points = np.array(trigger_points)
    #чтение и обработка новостей
    news_df = pd.read_csv(f'{dataset_folder}/stock-market-news.csv')
    news_df = news_df.append(pd.read_csv(f'{dataset_folder}/economy.csv'))
    news_df = news_df.append(pd.read_csv(f'{dataset_folder}/commodities-news.csv'))

    news_df['time'] = news_df.time.str.strip('(')
    news_df['time'] = news_df.time.str.strip(')')
    #приводим время к utc
    news_df['time'] = \
        pd.to_datetime(news_df['time']).dt.tz_localize(et_tz,
                                                       ambiguous=[True] * len(news_df['time']))
    news_df['time'] = news_df['time'].dt.tz_convert(utc).dt.tz_localize(None, ambiguous=[True] * len(news_df['time']))

    news_df = news_df.drop_duplicates(subset=['time', 'header'])

    sep = '<SEP>'
    glued_df = news_df.resample('1s', on='time', origin=datetime(2000, 1, 1, 0)).agg({'header': lambda x: sep.join(x)})
    glued_df = glued_df.sort_values(by='time')

    calc_key_points_filepath = Dataset.get(dataset_id=calc_key_points_file_id).get_local_copy() #получаем расположение функции для получения x для каждой из точек
    #импорт функции для получения x
    spec = importlib.util.spec_from_file_location("calc_key_points_module", f'{calc_key_points_filepath}/calc_key_points.py')
    calc_key_points_module = importlib.util.module_from_spec(spec)
    sys.modules["calc_key_points_module"] = calc_key_points_module
    spec.loader.exec_module(calc_key_points_module)
    #получение x
    key_points = calc_key_points_module.calc_key_points(trigger_points, breakout_points, glued_df, sep, window_size)
    #предобработка x
    import swifter
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    stemmer = nltk.PorterStemmer()

    def sent_to_words(sentence):
        return gensim.utils.simple_preprocess(str(sentence), deacc=True)

    def remove_stopwords(doc):
        return ' '.join([stemmer.stem(word) for word in sent_to_words(doc) if word not in stop_words])

    key_points.headers = key_points.headers.swifter.allow_dask_on_strings(enable=True).apply(remove_stopwords)
    return key_points, minutes_df, dataset_folder

# вторая ступен- обучение модели
def train_model(data, model_filepath_id, lr, n_epochs, batch_size):
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm.notebook import tqdm
    import time
    from collections import defaultdict
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import importlib.util
    import sys
    from clearml import Dataset
    from clearml import Logger
    logger = Logger.current_logger()
    n_epochs = int(n_epochs)
    batch_size = int(batch_size)
    lr = float(lr)
    model_filepath = Dataset.get(dataset_id=model_filepath_id).get_local_copy() #получаем расположение модуля с моделью
    #импорт модуля с моделью
    spec = importlib.util.spec_from_file_location("model_module", f'{model_filepath}/model.py')
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = model_module
    spec.loader.exec_module(model_module)

    key_points, minutes_df, dataset_folder = data
    #токенизация x
    num_words = 20000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(key_points.iloc[:int(len(key_points) * 0.7)].headers)
    y = key_points['Class'].values

    seq_len = 512
    x_seq = pad_sequences(tokenizer.texts_to_sequences(key_points.headers), seq_len, padding='post', truncating='post')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y[:int(len(key_points) * 0.7)]),
                                         y=y[:int(len(key_points) * 0.7)])
    #функия обучения модели
    def train(
            model_to_train,
            train_criterion,
            train_optimizer,
            train_batch_gen,
            val_batch_gen,
            num_epochs=50,
    ):
        history = defaultdict(lambda: defaultdict(list))

        for epoch in range(num_epochs):
            train_loss = 0
            train_acc = 0

            val_loss = 0
            val_acc = 0

            start_time = time.time()

            model_to_train.train(True)

            for x_batch, y_batch in tqdm(train_batch_gen):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model_to_train(x_batch)

                loss = train_criterion(logits, y_batch.to(device), weight=(torch.FloatTensor(class_weights)).to(device))

                loss.backward()
                train_optimizer.step()
                train_optimizer.zero_grad()

                train_loss += loss.detach().cpu().numpy()
                y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                train_acc += (y_pred == y_batch.cpu().numpy()).mean()

            train_loss /= len(train_batch_gen)
            train_acc /= len(train_batch_gen)
            history['loss']['train'].append(train_loss)
            history['acc']['train'].append(train_acc)

            model_to_train.train(False)

            for x_batch, y_batch in tqdm(val_batch_gen):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model_to_train(x_batch)

                loss = train_criterion(logits, y_batch.to(device))

                val_loss += loss.detach().cpu().numpy()
                y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                val_acc += (y_pred == y_batch.cpu().numpy()).mean()

            val_loss /= len(val_batch_gen)
            val_acc /= len(val_batch_gen)
            history['loss']['val'].append(val_loss)
            history['acc']['val'].append(val_acc)

            clear_output()

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
            print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
            print("  training accuracy: \t\t\t{:.2f}".format(train_acc))
            print("  validation accuracy: \t\t\t{:.2f}".format(val_acc))

            #логгирование лосса и точности в clearml
            logger.report_scalar('Loss', 'Train loss', iteration=epoch+1, value=train_loss)
            logger.report_scalar('Loss', 'Test loss', iteration=epoch+1, value=val_loss)
            logger.report_scalar('Accuracy', 'Train accuracy', iteration=epoch+1, value=train_acc)
            logger.report_scalar('Accuracy', 'Test accuracy', iteration=epoch+1, value=val_acc)

    #обучение модели
    train_dataset = torch.utils.data.TensorDataset(torch.LongTensor(x_seq[:int(len(key_points) * 0.7)]),
                                                   torch.LongTensor(y[:int(len(key_points) * 0.7)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(torch.LongTensor(x_seq[int(len(key_points) * 0.7):]),
                                                 torch.LongTensor(y[int(len(key_points) * 0.7):]))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = model_module.Model(20000, seq_len, 512, 2, 0)
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-8, lr=lr)
    criterion = F.cross_entropy
    model.to(device)
    train(model, criterion, optimizer, train_dataloader, val_dataloader, n_epochs)
    true_vals = y[int(len(key_points) * 0.7):]
    predictions = model(torch.LongTensor(x_seq[int(len(key_points) * 0.7):]).to(device))
    predictions = predictions.cpu().detach().numpy().argmax(axis=1)
    torch.jit.script(model).save('serving_model.pt') #сохранение модели
    return key_points, minutes_df, true_vals, predictions, dataset_folder

#третья ступень- подсчет качества модели
def get_metrics(data, tp, sl, n_exit):
    import matplotlib.pyplot as plt
    import sklearn.metrics
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm
    from clearml import Dataset, Logger, OutputModel

    logger = Logger.current_logger()
    tp = float(tp)
    sl = float(sl)
    n_exit = int(n_exit)
    key_points, minutes_df, true_vals, predictions, dataset_folder = data
    #подсчет метрик
    accuracy = (predictions == true_vals).mean()
    precision = sklearn.metrics.precision_score(true_vals, predictions)
    recall = sklearn.metrics.recall_score(true_vals, predictions)
    f1 = sklearn.metrics.f1_score(true_vals, predictions)
    roc_auc = sklearn.metrics.roc_auc_score(true_vals, predictions)

    #подсчет доходности, точек входа и выхода, условия выхода за одну сделку
    def get_date_income(minutes_dataframe, date, tgt_delta, stop_loss, n):
        for i in range(1, n):
            cur_index = date + timedelta(seconds=i)
            if cur_index in minutes_dataframe.index:
                if minutes_dataframe.loc[date].low_trigger - minutes_dataframe.loc[cur_index].Low >= tgt_delta: #преодолели target delta
                    return tgt_delta - 0.0466, date, cur_index, 1
                if minutes_dataframe.loc[cur_index].High - minutes_dataframe.loc[date].low_trigger >= stop_loss: #сработал stop-loss
                    return -stop_loss - 0.0466, date, cur_index, 2

        #выход по истечении периода
        cur_index = date + timedelta(seconds=n)
        return -(minutes_dataframe.loc[cur_index].Close - minutes_dataframe.loc[date].low_trigger) - 0.0466, \
            date, cur_index, 0
    #подсчет доходности, точек входа и выхода, условий выхода за все время тестирования
    def backtest(minutes_dataframe, dates, tgt_delta, stop_loss, n):
        income = []
        enters = []
        for date in tqdm(dates):
            date_income, date_in, date_out, condition_met = get_date_income(minutes_dataframe, date,
                                                                            tgt_delta, stop_loss, n)
            income.append(date_income)
            enters.append((date_in, date_out, condition_met))
        return income, enters

    tick_size, tick_value = 0.1, 10
    seconds_df = pd.read_csv(f'{dataset_folder}/seconds_df.csv', sep=';')
    seconds_df['Date'] = pd.to_datetime(seconds_df['Date'])
    seconds_df = seconds_df.set_index('Date')
    income_in_ticks, deals_enters = backtest(seconds_df,
                                             seconds_df.index.intersection(key_points.index[int(len(key_points) * 0.7):][predictions == 1]),
                                             tp, sl, n_exit) #бэктест, получает на вход данные о ценах, точки сделок и условия выхода
    income_in_ticks = np.array(income_in_ticks)
    income_in_ticks[np.isnan(income_in_ticks)] = 0
    #отрисовка кривой доходности
    plt.figure(figsize=(20, 10)).patch.set_facecolor('white')
    plt.grid(visible=True)
    plt.plot(seconds_df.index.intersection(key_points.index[int(len(key_points) * 0.7):][predictions == 1]),
             income_in_ticks.cumsum() / tick_size * tick_value, color='royalblue')
    plt.show()
    #отрисовка кривой доходности в clearml
    for i, day_income in enumerate(income_in_ticks.cumsum() / tick_size * tick_value):
        logger.report_scalar(title='Income', series='Test income', iteration=int(deals_enters[i][0].timestamp()), value=day_income)
    #сохранение модели в clearml
    OutputModel().update_weights('serving_model.pt')
    return accuracy, precision, recall, f1, roc_auc, income_in_ticks.sum() / tick_size * tick_value


if __name__ == '__main__':

    #создание пайплайна
    pipe = PipelineController(
        project='Testing',
        name='Pipeline testing',
        version='1.1',
        add_pipeline_tags=False,
    )
    #выбор очереди исполнения
    pipe.set_default_execution_queue('task_queue')
    #создание параметров пайплайна
    pipe.add_parameter('calc_key_points_file_id', 'c2719f91d95d4c0a8f0e9982304c46d7') #id датасета с файлом, содержащим функцию формирования x
    pipe.add_parameter('window_size', 5) #размер окна для формирования x, например n для стратегии n новостей до триггера
    pipe.add_parameter('dataset_id', '8b83e2dff6d647ba81cd9de0029d3c15') #id датасета с данными новостей, инструментов, etc.
    pipe.add_parameter('model_filepath_id', '7938df1a249e46dca8fa438217c7ffa3') #id датасета с файлом модели
    pipe.add_parameter('lr', 1e-6)
    pipe.add_parameter('n_epochs', 2)
    pipe.add_parameter('batch_size', 8)
    pipe.add_parameter('tp', 1.0)
    pipe.add_parameter('sl', 10.0)
    pipe.add_parameter('n_exit', 600)
    #создание ступени пайплайна на основе функция
    pipe.add_function_step(
        name='data_preparation', #название ступени
        function=prepare_data, #функция
        function_kwargs=dict(calc_key_points_file_id="${pipeline.calc_key_points_file_id}", #аргументы, которые будут переданы в функцию
                             window_size="${pipeline.window_size}",
                             dataset_id="${pipeline.dataset_id}"),
        function_return=['processed_data'], #название для возвращаемых данных
        cache_executed_step=True, #кэширование результатов выполнения ступени, если параметры данной ступени не изменятся, то будет возвращен уже подсчитанный результат, что позволяет сильно экономить ресурсы
    )
    pipe.add_function_step(
        name='model_training',
        function=train_model,
        function_kwargs=dict(data='${data_preparation.processed_data}',
                             model_filepath_id="${pipeline.model_filepath_id}",
                             lr="${pipeline.lr}",
                             n_epochs="${pipeline.n_epochs}",
                             batch_size="${pipeline.batch_size}"),
        parents=['data_preparation'],
        monitor_metrics=[('Loss', 'Train loss'),
                         ('Loss', 'Test loss'),
                         ('Accuracy', 'Train accuracy'),
                         ('Accuracy', 'Test accuracy')], #логгирование метрик не только в отдельную задачу, соответствующую данной ступени, но и в задачу, соответствующую всем пайплайну
        function_return=['predictions'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='model_estimation',
        function=get_metrics,
        function_kwargs=dict(data='${model_training.predictions}',
                             tp="${pipeline.tp}",
                             sl="${pipeline.sl}",
                             n_exit="${pipeline.n_exit}",
                             ),
        parents=['model_training'],
        monitor_metrics=[('Income', 'Test income')],
        function_return=['metrics'],
        cache_executed_step=True,
    )

    # Для локального запуска
    pipe.start_locally(run_pipeline_steps_locally=True)

    # Для запуска через clearml
    # pipe.start()

    print('pipeline completed')
