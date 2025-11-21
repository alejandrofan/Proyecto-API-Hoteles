import pandas as pd

ruta=("C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles")
df=pd.read_csv(ruta+"Data/hotel_bookings.csv")

#EXPLORACIÓN DE DATOS

#Número de filas y columnas
print(df.shape) #(119390, 32)

#Información de columnas y tipo de datos
print(df.info())

#Estadístcos descriptivos de variables numéricas.
print(df.describe())

#10 primeras filas
print(df.head(10))

#Revisión de variables faltantes
#Ver que variables faltantes.
hotel_bookings_nan=df[df.isna().any(axis=1)]
hotel_bookings_nan #[119173 rows x 32 columns]
#119390-119173= 217 filas que no tienen variables faltantes

#Que columnas
df.isna().sum()
    