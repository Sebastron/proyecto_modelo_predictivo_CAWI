Para utilizar y ejecutar el prototipo,se debe primero instalar las 
dependencias, ingrensando el comando que se muestra a continuación.

Nota: La versión de Python que se usó es 3.8.2, se recomienda 
      mantener esta versión debido a que tensorflow no soporta 
      otras que son más recientes.

pip3 install -r requerimientos.txt

Una vez hecho, en caso de no haber ningún error, por terminal se debe
ingresar el siguiente comando, para así comenzar con la ejecución del
prototipo. 

py interfaz.py

Si todo sale bien, debería generar la ventana principal, donde se
debe ingresar los archivos .h5 y .csv para su lectura. Para archivos
de modelos de RNA, se recomienda utilizar todos que contenga dos nodos
de salidas, los cuales estos se encuentra en dos carpetas.

1) En la misma carpeta del prototipo, "/Mejores modelos"
2) En la carpeta del proyecto,
   "/Codigos de fuente/Generación modelos RNA/Modelos/Modelos de dos salidas"
Como de preferencia, eliga el archivo "Dataset_de_prueba_oficial_1.csv" y 
"Dataset_de_prueba_oficial_2.csv" para probar el prototipo.
