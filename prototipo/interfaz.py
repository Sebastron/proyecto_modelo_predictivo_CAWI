from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QDialog, QTextEdit
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtCore import Qt, QTimer, QAbstractTableModel
from PyQt5 import uic
import sys
import pandas as pd
import prototipe as pt

class interfaz(QMainWindow):
    
    def __init__(self):
        super().__init__()
        uic.loadUi("Interfaz/interfaz_grafica.ui", self) #lectura de plantilla de interfaz principal
        width = 525
        height = 338
        self.setFixedSize(width,height)
        self.label_6.setPixmap(QPixmap("./Interfaz/Imagenes/sendinblue-logo-vector.png"))
        self.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
        self.browseModel.setCursor(Qt.PointingHandCursor)
        self.browseDataset.setCursor(Qt.PointingHandCursor)
        self.Comienzo.setCursor(Qt.PointingHandCursor)
        self.Comienzo.setEnabled(False)
        self.browseModel.clicked.connect(self.buscarModelo)
        self.browseDataset.clicked.connect(self.buscarDataset)
        self.Comienzo.clicked.connect(self.predecir)    

    def buscarModelo(self):
        try:
            self.modelo, path = pt.leer_archivo(0)
            self.texto1.setPlainText(path)
            if self.texto2.toPlainText():
                self.Comienzo.setEnabled(True)
            
        except:
            mensaje = QMessageBox()
            mensaje.setWindowTitle('Error de ingreso del Modelo RNA')
            mensaje.setText('No has ingresado el modelo correctamente. Vuelve a intentarlo.')
            mensaje.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
            mensaje.setIcon(QMessageBox.Warning)
            mensaje.exec()

    def buscarDataset(self):
        try:
            self.dataset, path = pt.leer_archivo(1)
            self.texto2.setPlainText(path)
            if self.texto1.toPlainText():
                self.Comienzo.setEnabled(True)
        except:
            mensaje = QMessageBox()
            mensaje.setWindowTitle('Error de ingreso del dataset en .csv')
            mensaje.setText('No has ingresado el archivo CSV correctamente. Vuelve a intentarlo. Nota: El separador que soporta este archivo es ;')
            mensaje.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
            mensaje.setIcon(QMessageBox.Warning)
            mensaje.exec()

    def predecir(self):
        try:
            self.horario_real, self.horario_predict, cantidad_real, cantidad_pred = pt.prediction(self.dataset, self.modelo)
            ventana = Ventana()
            width = 564
            height = 620
            ventana.setFixedSize(width,height)
            ventana.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
            ventana.btn_entendido.setCursor(Qt.PointingHandCursor)
            ventana.btn_entendido.clicked.connect(lambda: ventana.close())
            self.model1 = PdTable(self.horario_real)
            self.model2 = PdTable(self.horario_predict)
            ventana.label1.setText("Cantidad de respuesta estimada: " + cantidad_real)
            ventana.label2.setText("Cantidad de respuesta estimada: " + cantidad_pred)
            ventana.tabla1.setModel(self.model1)
            ventana.tabla2.setModel(self.model2)
            ventana.exec()
        except:
            mensaje = QMessageBox()
            mensaje.setWindowTitle('Error relacionado con el archivo csv')
            mensaje.setText("El dataset '.csv' que usted ingresó, no cuenta con la variable objetivo 'RESPONDIDA' y/o tienen cantidad de campos independientes diferente al 10. Sugiero revisar el archivo que acaba de ingresar.")
            mensaje.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
            mensaje.setIcon(QMessageBox.Warning)
            mensaje.exec()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Cerrar", "¿Está seguro que desea salir?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

class PdTable(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data
 
    def rowCount(self, parent=None):
        return self._data.shape[0]
 
    def columnCount(self, parent=None):
        return self._data.shape[1]
 
         # Mostrar datos
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None
 
         # Mostrar encabezado de fila y columna
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.axes[0][col]
        return None

class Ventana(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi('Interfaz/resultados.ui', self)
        self.setWindowIcon(QIcon("./Interfaz/Imagenes/logo-SendinBlue-1.png"))
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Cerrar", "¿Está seguro que desea salir?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = interfaz()
    GUI.show()
    sys.exit(app.exec_())