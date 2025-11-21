#LIBRERIAS:
#----------

from fastAPI import FastAPI
import pandas as pd

ruta="C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles/Data/hotel_bookings.csv"

app=FastAPI()

@app.get("/hoteles")
async def datos_hoteles()