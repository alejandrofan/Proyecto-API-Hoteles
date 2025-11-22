#LIBRERIAS:
#----------

from fastAPI import FastAPI
import pandas as pd

ruta="C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles/Data/hotel_bookings.csv"

app=FastAPI()


df = pd.read_csv(ruta)

MESES_VALIDOS = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]


@app.get("/pais/")
async def datos_hoteles():
    if "country" not in df.columns:
        return {"error": "La columna 'country' no existe en el CSV"}
    
    # Extraer países únicos y ordenarlos
    countries = sorted(df["country"].dropna().unique().tolist())
    return {"countries": countries}


@app.get("/tipos_hotel/")
async def tipos_hoteles():
    if "hotel" not in df.columns:
        return {"error": "La columna 'hotel' no existe en el CSV"}
    
    hotel_types = sorted(df["hotel"].dropna().unique().tolist())
    return {"hotel_types": hotel_types}


@app.get("/reservas_status/")
async def contar_reservas():
    if "reservation_status" not in df.columns:
        return {"error": "La columna 'reservation_status' no existe en el CSV"}
    
    # Contar cuántas reservas hay de cada tipo
    conteo = df["reservation_status"].value_counts().to_dict()
    
    # Filtrar solo los estados que nos interesan
    estados_interes = ["Canceled", "No-Show", "Check-Out"]
    resultado = {estado: conteo.get(estado, 0) for estado in estados_interes}
    
    return {"reservation_status_count": resultado}



@app.get("/estadisticas/")
async def estadisticas_generales():
    columnas_necesarias = ["adults", "children", "stays_in_week_nights", "stays_in_weekend_nights"]
    
    # Verificar que existan las columnas
    columnas_existentes = [col for col in columnas_necesarias if col in df.columns]
    
    if not columnas_existentes:
        return {"error": "No se encontraron las columnas necesarias para estadísticas"}
    
    # Usar describe y convertir a diccionario
    stats = df[columnas_existentes].describe().round(2).to_dict()
    
    # Añadir total de reservas
    stats["total_reservas"] = len(df)
    
    return stats



@app.get("/reservas/")
async def reservas_por_fecha(year: int, mes: str):
    # Validar año
    if year not in [2015, 2016, 2017]:
        return {"error": "El año debe ser 2015, 2016 o 2017"}
    
    # Validar mes
    if mes not in MESES_VALIDOS:
        return {"error": f"Mes inválido. Debe ser uno de {MESES_VALIDOS}"}
    
    # Filtrar el DataFrame
    reservas_filtradas = df[(df["arrival_date_year"] == year) & (df["arrival_date_month"] == mes)]
    
    # Convertir a lista de diccionarios para devolver como JSON
    resultado = reservas_filtradas.to_dict(orient="records")
    
    return {"year": year, "mes": mes, "total_reservas": len(resultado), "reservas": resultado}













