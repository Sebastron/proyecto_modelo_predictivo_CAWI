import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.uic.uiparser import QtWidgets
import numpy as np
import csv
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense,Dropout,Flatten
from keras import backend as K
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils
import tkinter as Tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix

def Prediccion_horario(cantidad_respuestas):
  horario = pd.read_csv('Probabilidad_horario.csv', sep=';')
  horario["Cantidad_estimada"] = (horario["Probabilidad_respuesta"]*cantidad_respuestas).round(0).astype(int)
  horario = horario.drop('Probabilidad_respuesta', axis=1)
  horario = horario.pivot(index ='Hora', columns ='Dia_semana')
  horario = horario.reindex(columns=[('Cantidad_estimada', 'Lunes'), ('Cantidad_estimada', 'Martes'), 
            ('Cantidad_estimada', 'Miercoles'), ('Cantidad_estimada', 'Jueves'), ('Cantidad_estimada', 'Viernes'), 
            ('Cantidad_estimada', 'Sabado'), ('Cantidad_estimada',   'Domingo')])
  aux = horario.to_numpy()
  horario = pd.DataFrame(aux, columns=['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
                  index=range(8, 22))
  return horario

def leer_archivo(correspondencia):
    root = Tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if (file_path[-2:] == 'h5' and correspondencia==0):
        df = keras.models.load_model(file_path)
        if((df.layers[-1].output_shape[1])!= 2):
           print("El modelo que usted ingresó no cuenta con dos dimensiones de salida.")
           aux = str(df.layers[-1].output_shape[1])
           mensaje = QMessageBox()
           mensaje.setWindowTitle('Error de ingreso del Modelo RNA')
           mensaje.setText('Este modelo contiene solo ' + aux +' nodo(s) de salida, y este prototipo solo soporta modelos de dos salidas (output).' )
           mensaje.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
           mensaje.setIcon(QMessageBox.Warning)
           mensaje.exec()
           return False
        else:
            print("Modelo correcto.")
    elif (file_path[-3:] == 'csv' and correspondencia==1):
        df = pd.read_csv(file_path, sep=';')
    else:
        print('Formato incorrecto, vuelve a ingresar.')
    return df, str(file_path)

def prediction(df, model): 
    X = df.drop('RESPONDIDA', axis=1).to_numpy()
    Y = df["RESPONDIDA"].to_numpy()
    Y = np_utils.to_categorical(Y)
    response_real = df[(df['RESPONDIDA'] ==1)].shape[0]
    new_predictions = model.predict(X)
    confusion = confusion_matrix(Y.argmax(axis=1), new_predictions.argmax(axis=1))
    response_predict = np.sum(confusion[:,1], axis=0)
    horario_real = Prediccion_horario(response_real)
    horario_predict = Prediccion_horario(response_predict)
    #horario_real.to_excel('Horario_real.xlsx', sheet_name='Horario')
    #horario_predict.to_excel('Horario_predecido.xlsx', sheet_name='Horario_pred')
    return horario_real, horario_predict, str(response_real), str(response_predict)
