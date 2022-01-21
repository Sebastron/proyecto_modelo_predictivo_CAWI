import pandas as pd
import datetime as dt
import numpy as np
import csv

dtypes = {'E_mail': pd.np.str,
          'Sexo':pd.np.str,
          'Edad':pd.np.int8,
          'Segmento':pd.np.str,
          'SubSegmento':pd.np.str,
          'Segto_Agrup':pd.np.str,
          'Carterizado':pd.np.int8,
          'Cantidad_abierto':pd.np.int8,
          'Hubo_Primera_Apertura':pd.np.int8,
          'Mes_envio':pd.np.int8,
          'Dia_envio':pd.np.int8,
          'Hora_envio':pd.np.int8,
          'Fecha':pd.np.str,
          'Dia_semana':pd.np.str}

data = pd.read_csv('Datos_estructurados/DatasetV4.csv', delimiter = ';', dtype=dtypes)
#data['Fecha'] = '2020' + '-' + (data['Fecha_envio'].str.split(' ').str.get(0)).str.split("-").str.get(1) + '-' + (data['Fecha_envio'].str.split(' ').str.get(0)).str.split("-").str.get(0)
data = data.sort_values(['E_mail','Fecha']).reset_index(drop = True)

data['Fecha_Termino'] = ''

# Se agrega fecha de termino en cada envío
for i in range(len(data['Fecha'])):
    if i < len(data['Fecha'])-1:
        data['Fecha_Termino'][i] = data['Fecha'][i+1]
        if data['E_mail'][i] != data['E_mail'][i+1]:
            data['Fecha_Termino'][i] = '2021-01-01'
    print(i, " / ", len(data['Fecha']))
data['Fecha_Termino'] = data['Fecha_Termino'].replace('','2021-01-01')

data['Fecha'] =  pd.to_datetime(data['Fecha'])
data['Fecha_Termino'] =  pd.to_datetime(data['Fecha_Termino'])
data['Duracion_i_f'] = data['Fecha_Termino'] - data['Fecha']
data['Duracion_i_f'] = data['Duracion_i_f'].map(lambda x: np.nan if pd.isnull(x) else x.days)

data.to_csv("Datos_estructurados/DatasetV4.csv", sep= ';', index = False)