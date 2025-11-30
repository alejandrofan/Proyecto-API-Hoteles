import sqlite3
import os
import pandas as pd

# Configuración de la base de datos
raiz = "C:\Proyecto-API-Hoteles/Data/"
DB_NAME = raiz + "hoteles_oficial.db"
CSV_FILE = raiz + "hotel_bookings.csv"

def crear_base_datos_gestion():
    print("🚀 Iniciando creación de base de datos para gestión...")
    
    if not os.path.exists(CSV_FILE):
        print("❌ Error: No encuentro el archivo CSV.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Limpieza suave (mantener todas las columnas)
    print("🧹 Realizando limpieza básica de nulos...")
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['children'] = df['children'].fillna(0)
    
    # Guardar tabla de reservas
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("reservas", conn, if_exists="replace", index_label="id")
    conn.close()
    
    print(f"✅ Tabla 'reservas' creada correctamente.")

def crear_tabla_predicciones():
    print("🔮 Creando tabla para el historial de predicciones...")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Creamos la tabla si no existe
    # Incluimos fecha automática (TIMESTAMP) y los campos clave del formulario
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_consulta TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hotel TEXT,
            mes_llegada TEXT,
            antelacion_dias INTEGER,
            adultos INTEGER,
            ninos INTEGER,
            habitacion TEXT,
            segmento TEXT,
            precio_predicho REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Tabla 'historial_predicciones' verificada/creada.")

if __name__ == "__main__":
    # 1. Crear la tabla principal con los datos del CSV
    crear_base_datos_gestion()
    
    # 2. Crear la tabla vacía para guardar futuras predicciones
    crear_tabla_predicciones()
    
    print(f"\n🎉 BASE DE DATOS COMPLETADA EN: {DB_NAME}")