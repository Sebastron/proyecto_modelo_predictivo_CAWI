import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense,Dropout,Flatten
from keras import backend as K
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils

def red_neuronal_dos_salidas(df, optimizador, f_perdida, capas, epocas, tamano):
    #Particion de datos de entrenamiento y prueba
    Train = df[(df['Mes_envio']<0.5)]
    X_train = Train.drop('RESPONDIDA', axis=1).to_numpy()
    y_train = Train["RESPONDIDA"].to_numpy()
    Test = df[(df['Mes_envio']>=0.5)]
    X_test = Test.drop('RESPONDIDA', axis=1).to_numpy()
    y_test = Test["RESPONDIDA"].to_numpy()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    #Arquitectura del modelo
    model = Sequential()
    model.add(Dense(10, input_dim = 10, activation='relu'))
    for neuronas, funcion_activacion in capas:
      model.add(Dense(neuronas, activation = funcion_activacion))
    model.add(Dense(2, activation = 'softmax', kernel_initializer='normal'))
    model.compile(loss=f_perdida, optimizer=optimizador, metrics=['binary_accuracy'])
    # Ajuste del modelo
    callEar = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)
    model.fit(X_train, y_train, epochs = epocas, batch_size = tamano, callbacks=[callEar], verbose = 1)
    scores = model.evaluate(X_test, y_test)
    precision = scores[1]
    perdida = scores[0]
    return precision, perdida, model

def red_neuronal_una_salida(df, optimizador, f_perdida, capas, epocas, tamano):
    #Particion de datos de entrenamiento y prueba
    Train = df[(df['Mes_envio']<0.5)]
    X_train = Train.drop('RESPONDIDA', axis=1).to_numpy()
    y_train = Train["RESPONDIDA"].to_numpy()
    Test = df[(df['Mes_envio']>=0.5)]
    X_test = Test.drop('RESPONDIDA', axis=1).to_numpy()
    y_test = Test["RESPONDIDA"].to_numpy()
    #Arquitectura del modelo
    model = Sequential()
    model.add(Dense(10, input_dim = 10, activation='relu'))
    for neuronas, funcion_activacion in capas:
      model.add(Dense(neuronas, activation = funcion_activacion))
    model.add(Dense(1, activation = 'relu', kernel_initializer='normal'))
    model.compile(loss=f_perdida, optimizer=optimizador, metrics=['binary_accuracy'])
    # Ajuste del modelo
    callEar = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)
    model.fit(X_train, y_train, epochs = epocas, batch_size = tamano, callbacks=[callEar], verbose = 1)
    scores = model.evaluate(X_test, y_test)
    precision = scores[1]
    perdida = scores[0]
    return precision, perdida, model

# Datasets
df = pd.read_csv('Dataset_de_prueba_oficial_1.csv', sep=';')
df2 = pd.read_csv('Dataset_de_prueba_oficial_2.csv', sep=';')

# Definiciones Arquitecturas
#activation= 'sigmoid', 'relu', 'softmax', 'tanh'
#loss= 'categorical_crossentropy', 
#Optimizador = 'adam', 'sgd', 'Adamax', 'Adagrad'

#Se debe modificar el arreglo manualmente, donde:
# 1: Nombre de la red
# 2: Optimizador de red
# 3: Funcion de pérdida de la red
# 4: Las capas ocultas de la red, donde uno es N° de neuronas (1) y otro función de optimización (2).
ejecuciones = [['Red1','adam','categorical_crossentropy',[[25,'relu'],[40,'relu'],[100,'sigmoid'],[50,'sigmoid'],[25,'sigmoid'],[10,'relu']]],
               ['Red2','adam','categorical_crossentropy',[[15,'tanh'],[30,'relu']]],
               ['Red3','adam','categorical_crossentropy',[[20,'sigmoid'],[10,'relu']]],
               ['Red4','Adagrad','categorical_crossentropy',[[25,'tanh'],[40, 'elu'],[10,'relu']]],
               ['Red5','sgd','categorical_crossentropy',[[40,'tanh'],[80,'sigmoid'],[35,'sigmoid'],[10,'relu']]],
               ['Red6','adam','categorical_crossentropy',[[30,'elu'],[30,'tanh'], [30,'elu']]],
               ['Red7','sgd','categorical_crossentropy',[[20,'relu'],[50,'sigmoid'],[40, 'relu'], [10,'sigmoid']]],
               ['Red8','adam','categorical_crossentropy',[[30,'relu'],[50,'tanh'],[40, 'sigmoid'], [5,'relu']]],
               ['Red9','sgd','categorical_crossentropy',[[30,'relu'],[65,'tanh'],[100, 'sigmoid'], [65,'relu'], [30,'tanh'], [10, 'sigmoid']]],
               ['Red10','adam','categorical_crossentropy',[[25,'relu'],[60,'tanh'],[90, 'sigmoid'], [60,'relu'], [30,'tanh'], [15, 'sigmoid']]],
               ['Red11','Adagrad','categorical_crossentropy',[[30,'sigmoid'],[55,'relu'],[90, 'tanh'], [65,'relu'], [35,'relu'], [10, 'sigmoid']]],
               ['Red12','sgd','categorical_crossentropy',[[25,'sigmoid'],[50,'relu'],[90, 'tanh'], [120,'relu'], [80,'relu'], [50, 'sigmoid'], [20, 'sigmoid']]],
               ['Red13','Adamax','categorical_crossentropy',[[20,'relu'],[50, 'sigmoid'],[25,'relu']]],
               ['Red14','Adamax','categorical_crossentropy',[[25,'elu'],[40, 'sigmoid'],[60,'tanh'], [35,'sigmoid'], [15,'relu']]],
               ['Red15','Adagrad','categorical_crossentropy',[[25,'sigmoid'],[50, 'elu'],[70,'relu'],[100,'tanh'], [65,'relu'], [25, 'sigmoid'], [10, 'elu']]],
               ['Red16','adam','categorical_crossentropy',[[30,'sigmoid'],[55, 'elu'],[75,'relu'],[100,'tanh'], [70,'relu'], [35, 'sigmoid'], [15, 'elu']]],
               ['Red17','sgd','categorical_crossentropy',[[25,'sigmoid'],[50, 'relu'],[75,'elu'],[100,'sigmoid'], [70,'tanh'], [35, 'relu'], [15, 'elu']]],
               ['Red18','adam','categorical_crossentropy',[[20,'elu'],[50, 'relu'],[25,'elu'],[10,'sigmoid']]],
               ['Red19','adam','categorical_crossentropy',[[25,'relu'],[45, 'softmax'],[20,'sigmoid'],[5,'sigmoid']]],
               ['Red20','sgd','categorical_crossentropy',[[20,'softmax'],[50, 'sigmoid'],[25,'tanh'],[10,'relu']]]
               ]
arr1 = []
arr2 = []
for nombre, optimizador, f_perdida, arquitectura in ejecuciones:
  redes = pd.read_excel('Diseño_arquitecturas.xlsx')
  array_capas=[]
  contador = 1
  for capa in arquitectura:
    array_capas.append([nombre, contador, capa[0], capa[1], optimizador])
    contador+=1
  nueva_Arquitectura = pd.DataFrame(array_capas,columns = ['Red', 'Capa_oculta', 'Neuronas', 'Funcion_activacion', 'Optimizador'])
  redes = pd.concat([redes, nueva_Arquitectura])
  redes.to_excel('Diseño_arquitecturas.xlsx',index = False)
  for salida in range (0, 2):
    for dataset in range (0, 2):
      for i in range (1, 6):
        # Se entrena la red
        if(dataset == 0):
          if(salida==0):
            precision, perdida, Modelo =  red_neuronal_una_salida(df, optimizador, f_perdida, arquitectura, 1000, 2200)
            arr1.append([nombre, "Dataset1", i, optimizador, f_perdida, precision, perdida])
          else:
            precision, perdida, Modelo =  red_neuronal_dos_salidas(df, optimizador, f_perdida, arquitectura, 1000, 2200)
            arr2.append([nombre, "Dataset1", i, optimizador, f_perdida, precision, perdida])
        else:
          if(salida==0):
            precision, perdida, Modelo = red_neuronal_una_salida(df2, optimizador, f_perdida, arquitectura, 1000, 1550)
            arr1.append([nombre, "Dataset2", i, optimizador, f_perdida, precision, perdida])
          else:
            precision, perdida, Modelo =  red_neuronal_dos_salidas(df, optimizador, f_perdida, arquitectura, 1000, 1550)
            arr2.append([nombre, "Dataset2", i, optimizador, f_perdida, precision, perdida])
      # Se guardan los modelos completos (ya entrenados)
      if(dataset == 0):
        if(salida == 0):
          Modelo.save('Modelos/'+ nombre + '_D1S1.h5')
        else:
          Modelo.save('Modelos/'+ nombre + '_D1S2.h5')
      else:
        if(salida == 0):
          Modelo.save('Modelos/'+ nombre + '_D2S1.h5')
        else:
          Modelo.save('Modelos/'+ nombre + '_D2S2.h5')

# Se guarda los resultados de evaluaciones de cada red
df_glob1 = pd.read_excel('Datos_Red_1_salidas.xlsx')
df_salida1 = pd.DataFrame(arr1, columns = ['Red', 'Dataset', 'Iteracion', 'Optimizador', 'F_Perdida', 'Precision', 'Perdida (loss)'])
df_glob1 = pd.concat([df_glob1, df_salida1])
df_glob1.to_excel('Datos_Red_1_salidas.xlsx',index = False)

df_glob2 = pd.read_excel('Datos_Red_2_salidas.xlsx')
df_salida2 = pd.DataFrame(arr2, columns = ['Red', 'Dataset', 'Iteracion', 'Optimizador', 'F_Perdida', 'Precision', 'Perdida (loss)'])
df_glob2 = pd.concat([df_glob2, df_salida2])
df_glob2.to_excel('Datos_Red_2_salidas.xlsx', index = False)