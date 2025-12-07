import sqlite3
import pandas as pd
import os
import joblib
import uvicorn
import nest_asyncio
import xgboost 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime 

# Configuración para evitar errores en Jupyter/Spyder
nest_asyncio.apply()

app = FastAPI(
    title="Hotel Intelligence API",
    description="API para análisis de datos hoteleros y consulta de KPIs",
    version="2.7" 
)

# --- CONFIGURACIÓN CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = "C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles/" 
raiz_data = BASE_DIR + "Data/"
raiz_ml = BASE_DIR + "ML/"

DB_NAME = raiz_data + "hoteles_oficial.db"
CSV_FILE = raiz_data + "hotel_bookings.csv"
MODEL_FILE = raiz_ml + "modelo_pipeline_completo.joblib" 

# --- INICIALIZACIÓN BBDD ---
def init_db():
    if not os.path.exists(raiz_data):
        os.makedirs(raiz_data, exist_ok=True)

    # Verificamos que existan las tablas necesarias
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Tabla Historial Predicciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_consulta TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hotel TEXT,
            mes_llegada TEXT,
            antelacion_dias INTEGER,
            adults INTEGER,
            ninos INTEGER,
            habitacion TEXT,
            segmento TEXT,
            precio_predicho REAL
        )
    """)
    conn.commit()
    
    # 2. Tabla Reservas 
    if not os.path.exists(DB_NAME):
        conn.execute("CREATE TABLE IF NOT EXISTS reservas (id INTEGER PRIMARY KEY)") 
    
    conn.close()

init_db()

def ejecutar_query(sql: str, params: tuple = ()):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        return cursor

# --- CARGA DEL MODELO PIPELINE ---
pipeline = None
try:
    if os.path.exists(MODEL_FILE):
        print(f"🔄 Cargando pipeline desde: {MODEL_FILE}")
        pipeline = joblib.load(MODEL_FILE)
        print("✅ Pipeline cargado correctamente.")
    else:
        print(f"⚠️ El archivo {MODEL_FILE} no existe. La predicción fallará.")
except Exception as e:
    print(f"❌ Error cargando el joblib: {e}")

# --- MODELOS PYDANTIC ---

class DatosReserva(BaseModel):
    hotel: str
    arrival_date_year: int 
    arrival_date_month: str
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    reserved_room_type: str
    assigned_room_type: str 
    deposit_type: str
    customer_type: str
    lead_time: int
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    booking_changes: int
    agent: float
    company: float 
    days_in_waiting_list: int
    required_car_parking_spaces: int
    total_of_special_requests: int
    total_people: int
    total_nights: int

class Reserva(BaseModel):
    hotel: str
    arrival_date_year: int
    arrival_date_month: str
    adr: float
    is_canceled: int = 0
    country: str
    adults: int
    children: float

# --- ENDPOINTS ---

@app.post("/prediccion_adr")
async def obtener_prediccion(datos: DatosReserva): 
    if pipeline is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")

    try:
        # 1. Predecir
        df_input = pd.DataFrame([datos.model_dump()])
        prediccion = pipeline.predict(df_input)
        resultado_adr = round(float(prediccion[0]), 2)
        
        # 2. Guardar en Base de Datos (Historial)
        query_save = """
            INSERT INTO historial_predicciones 
            (fecha_consulta, hotel, mes_llegada, antelacion_dias, adultos, ninos, habitacion, segmento, precio_predicho)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        ahora = datetime.now()
        
        params_save = (
            ahora,
            datos.hotel,
            datos.arrival_date_month,
            datos.lead_time,
            datos.adults,
            datos.children,
            datos.reserved_room_type,
            datos.market_segment,
            resultado_adr
        )
        ejecutar_query(query_save, params_save)
        
        return {"predicción_adr": resultado_adr}
    
    except Exception as e:
        print(f"Error detallado: {e}") 
        raise HTTPException(status_code=400, detail=f"Error del Modelo ML: {str(e)}")

@app.get("/historial_predicciones/", tags=["Business Intelligence"])
async def consultar_historial():
    try:
        query = "SELECT * FROM historial_predicciones ORDER BY id DESC LIMIT 50"
        cursor = ejecutar_query(query)
        resultados = [dict(row) for row in cursor.fetchall()]
        return {"historial": resultados}
    except Exception as e:
        return {"historial": [], "error": str(e)}

@app.get("/analisis-reservas/", tags=["Business Intelligence"])
def consultar_datos(year: int, mes: str):
    query = """
        SELECT 
            COUNT(*) as total,
            SUM(is_canceled) as canceladas,
            AVG(adr) as precio_promedio
        FROM reservas 
        WHERE arrival_date_year = ? AND arrival_date_month = ?
    """
    row = ejecutar_query(query, (year, mes)).fetchone()
    
    if not row or row["total"] == 0:
        return {"mensaje": "No hay datos para esa fecha"}

    return {
        "fecha": f"{mes} {year}",
        "total_reservas": row["total"],
        "canceladas": row["canceladas"],
        "precio_promedio": round(row["precio_promedio"], 2) if row["precio_promedio"] else 0
    }

@app.get("/pais/", tags=["Filtros y Listados"])
async def listar_paises_origen():
    query = "SELECT DISTINCT country FROM reservas WHERE country IS NOT NULL ORDER BY country"
    cursor = ejecutar_query(query)
    paises = [row["country"] for row in cursor.fetchall()]
    return {"total_paises": len(paises), "countries": paises}

@app.get("/tipos_hotel/", tags=["Filtros y Listados"])
async def listar_tipos_hoteles():
    query = "SELECT DISTINCT hotel FROM reservas ORDER BY hotel"
    cursor = ejecutar_query(query)
    tipos = [row["hotel"] for row in cursor.fetchall()]
    return {"hotel_types": tipos}

@app.get("/reservas_status/", tags=["Estadísticas Rápidas"])
async def contar_estado_reservas():
    try:
        query = """
            SELECT is_canceled, COUNT(*) as cantidad 
            FROM reservas 
            GROUP BY is_canceled
        """
        cursor = ejecutar_query(query)
        resultado = {"Check-Out": 0, "Canceled": 0, "No-Show": 0}
        
        for row in cursor.fetchall():
            if row["is_canceled"] == 1:
                resultado["Canceled"] += row["cantidad"]
            else:
                resultado["Check-Out"] += row["cantidad"]
                
        return {"reservation_status_count": resultado}
    except Exception as e:
        return {"error": str(e)}

@app.get("/estadisticas/", tags=["Estadísticas Rápidas"])
async def estadisticas_descriptivas():
    cols = ["adults", "children", "stays_in_week_nights", "stays_in_weekend_nights", "adr"]
    stats = {}
    for col in cols:
        try:
            query = f"SELECT AVG({col}) as media, MIN({col}) as minimo, MAX({col}) as maximo FROM reservas"
            row = ejecutar_query(query).fetchone()
            stats[col] = {
                "mean": round(row["media"], 2) if row["media"] else 0,
                "min": row["minimo"],
                "max": row["maximo"]
            }
        except:
            stats[col] = "Error"
    
    total = ejecutar_query("SELECT COUNT(*) as total FROM reservas").fetchone()["total"]
    stats["total_reservas"] = total
    return stats



@app.post("/reservas/", tags=["Gestión Reservas"])
def crear_reserva(reserva: Reserva):
    try:
        row = ejecutar_query("SELECT MAX(id) as max_id FROM reservas").fetchone()
        new_id = (row["max_id"] if row and row["max_id"] is not None else 0) + 1
    except:
        new_id = 1

    query = """
        INSERT INTO reservas (id, hotel, arrival_date_year, arrival_date_month, adr, is_canceled, country, adults, children)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (new_id, reserva.hotel, reserva.arrival_date_year, reserva.arrival_date_month, 
              reserva.adr, reserva.is_canceled, reserva.country, reserva.adults, reserva.children)
    
    cursor = ejecutar_query(query, params)
    return {"mensaje": "Reserva creada", "id": new_id}


@app.put("/reservas/{id}", tags=["Gestión Reservas"])
def actualizar_reserva(id: int, reserva_actualizada: Reserva):

    query_check = "SELECT id FROM reservas WHERE id = ?"
    check = ejecutar_query(query_check, (id,)).fetchone()
    
    if not check:
        raise HTTPException(status_code=404, detail=f"Reserva con ID {id} no encontrada")

    query_update = """
        UPDATE reservas
        SET hotel = ?, 
            arrival_date_year = ?, 
            arrival_date_month = ?, 
            adr = ?, 
            is_canceled = ?, 
            country = ?, 
            adults = ?, 
            children = ?
        WHERE id = ?
    """
    
    params = (
        reserva_actualizada.hotel,
        reserva_actualizada.arrival_date_year,
        reserva_actualizada.arrival_date_month,
        reserva_actualizada.adr,
        reserva_actualizada.is_canceled,
        reserva_actualizada.country,
        reserva_actualizada.adults,
        reserva_actualizada.children,
        id
    )
    
    try:
        ejecutar_query(query_update, params)
        return {"mensaje": f"Reserva {id} actualizada correctamente", "datos": reserva_actualizada}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar la base de datos: {str(e)}")

@app.delete("/reservas/{id}", tags=["Gestión Reservas"])
def borrar_reserva(id: int):
    check = ejecutar_query("SELECT id FROM reservas WHERE id = ?", (id,)).fetchone()
    if not check:
        raise HTTPException(status_code=404, detail="ID no encontrado")
    ejecutar_query("DELETE FROM reservas WHERE id = ?", (id,))
    return {"mensaje": f"Reserva {id} eliminada"}


if __name__ == "__main__":
    print("🚀 Iniciando servidor API (v2.7) ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)



