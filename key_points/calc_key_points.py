import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm.notebook import tqdm

#алгоритм формирования x

def calc_key_points(trigger_points, breakout_points, glued_df, sep, window_size):
    key_points = pd.DataFrame([])
    key_points['Date'] = trigger_points[range(len(trigger_points)), 1]
    key_points['headers'] = np.nan
    tmp_glued_df = glued_df[glued_df['header'] != '']
    for i in tqdm(np.arange(len(key_points))):
        tmp = tmp_glued_df.loc[:trigger_points[i][1] - timedelta(seconds=1)]
        key_points['headers'].iloc[i] = \
            sep.join(filter(None,
                            tmp.values[range(max(-window_size, -len(tmp)), 0), 0])) #для каждой точки берем window_size предшествующих новостей

    key_points = key_points.set_index('Date')
    #размечаем классы
    key_points['Class'] = 0
    key_points['Class'].loc[breakout_points] = 1
    #убираем точки, для которых не нашлось новостей
    key_points = key_points[key_points['headers'] != '']
    return key_points
