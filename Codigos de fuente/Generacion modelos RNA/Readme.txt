Para utilizar y ejecutar el generador de modelos,se debe primero 
instalar las dependencias, ingrensando el comando que se muestra 
a continuación.

Nota: La versión de Python que se usó es 3.8.2, se recomienda 
      mantener esta versión debido a que tensorflow no soporta 
      otras que son más recientes.

pip3 install -r requerimientos.txt

Una vez hecho, en caso de no haber ningún error, por terminal se debe
ingresar el siguiente comando, para así comenzar con la ejecución del
programa. 

py generacion_modelos.py

Si todo sale bien, debería aparecer las ejecuciones de epocas en el 
términal. Al terminar de ejecutarse, se actualizan los resultados de
evaluaciones en los archivos "Datos_Red_1_salidas.xlsx", 
"Datos_Red_1_salidas" y "Diseño_arquitecturas". Además, se guardan
los modelos entrenados como archivo ".h5", en la carpeta "/Modelos".

IMPORTANTE: Se debe modificar la variable lista "ejecuciones" para
            almacenar resultados y modelos distintos.