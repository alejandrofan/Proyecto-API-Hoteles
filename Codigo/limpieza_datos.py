#LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os

ruta=("C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles/")
df=pd.read_csv(ruta+"Data/hotel_bookings.csv")

#INFORMACIÓN DEL DATASET
#Número de filas y columnas
print(df.shape) #(119390, 32)

#Información de columnas y tipo de datos
print(df.info())

#Estadístcos descriptivos de variables numéricas.
print(df.describe())

#10 primeras filas
print(df.head(10))

#LIMPIEZA DE DATOS
#Revisión de variables faltantes
#Ver que variables faltantes.
hotel_bookings_nan=df[df.isna().any(axis=1)]
hotel_bookings_nan #[119173 rows x 32 columns]
#119390-119173= 217 filas que no tienen variables faltantes

#Que columnas hay variables faltantes
df.isna().sum() #children= 4, country= 488 , agent= 16340, company= 112593

#Eliminamos la columna "company" por exceso de NAs 
df=df.drop(columns= ['company'])
#Cuantas filas y columnas tenemos una vez eliminadas.
df.shape #(87370, 31)

#Rellenamos las variables faltantes por 0
df['agent']=df['agent'].fillna(0)

df['children']=df['children'].fillna(0)

#Rellenamos con la moda.
df['country']= df['country'].fillna(df['country'].mode()[0])

#Revisamos si hay filas duplicados.
duplicados= df.duplicated().sum()
print(duplicados) #hay 32020 duplicados
#los eliminamos para obtener un mejor resultado en el modelo.
df=df.drop_duplicates()


import matplotlib.pyplot as plt
import seaborn as sns

save_grafico=(ruta+"Gráficos")

#IDENTIFICACIÓN DE OUTLIERS
#Se puede ver la diferencia abismal entre el 50%,75% y el max. La diferencia es extremádamente grande.
print(df['adr'].describe())

#Histograma
plt.figure(figsize=(10,6))
sns.histplot(df['adr'], bins=50, kde=True)
plt.title("Distribución de ADR")
plt.xlabel("ADR (Tarifa media diaria)")
plt.ylabel("Frecuencia")
plt.savefig(save_grafico+'/Distribución_ADR.png')
plt.show()
plt.close() #Los valores ADR son muy altos pero con poca frecuencia (Evidencia de outliers)

#Box plot
plt.figure(figsize=(10,6))
sns.boxplot(x=df['adr'])
plt.title('Boxplot de ADR para Detección de Outliers')
plt.xlabel('ADR (Tarifa Media Diaria)')
plt.savefig(save_grafico+'/Boxplot_ADR.png')
plt.show() 
"""Se puede ver claramente que hay una gran cantidad de puntos a la derecha de la caja y su bigote
Por lo que visualmente y estadísticamente confirmamos la existencia de muchos outliers en "ADR."""

"""
En este caso no los eliminamos, porque pueden ser hoteles de lujo, reservas premium, etc.
Debido a esta razón, no eliminaremos los outliers sino que los usaremos para entrenar el modelo.
y mejorar la calidad de acierto."""

#GUARDAR DATOS EN SQLITE
import sqlite3
import os

base_datos=(ruta+"Data/")
conexion=sqlite3.connect(base_datos+"hotel_bookings_clean.db")
df.to_sql("hotel_bookings_clean",conexion, if_exists="replace",index=False)
conexion.close()





