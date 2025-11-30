# Proyecto API Hoteles: Predicción de ADR - ML + Interfaz Web
Este proyecto desarrolla un sistema capaz de predecir el Average Daily Rate (ADR) de reservas hoteleras utilizando técnicas de Machine Learning, en concreto el modelo XGBoost. 

Además, se implementa una API desarrollada con FastAPI que permite no solo obtener predicciones dinámicas a partir de inputs proporcionados por el usuario, sino también acceder a información relevante del dataset hotel_bookings.csv.

Finalmente, se integra una interfaz web en HTML que facilita la interacción del usuario con el sistema, permitiendo consultar datos, explorar estadísticas y realizar predicciones del ADR de manera sencilla y accesible desde cualquier navegador. 
## Problema de negocio
El sector hotelero depende de una correcta estrategia de **revenue management** para maximizar la rentabilidad. Una de las métricas clave es el **ADR (Average Daily Rate)**, que refleja el precio promedio pagado por habitación en un periodo concreto.

Fijar correctamente el ADR es un reto debido a fluctuaciones en la demanda, temporada, tipo de cliente, duración de la estancia, políticas de cancelación y otras variables operativas.

- Un ADR demasiado alto puede reducir la ocupación.

- Un ADR demasiado bajo puede suponer pérdidas al desaprovechar ingresos potenciales.

**Solución:** Este proyecto ofrece una herramienta que predice el ADR óptimo basado en las características reales de la reserva, permitiendo tomar mejores decisiones comerciales y mejorar el rendimiento financiero.
## Objetivos

1.  **Análisis de Datos:** Explorar y limpiar el dataset `hotel_bookings.csv` para entender las variables que influyen en el precio.
2.  **Modelo Predictivo:** Entrenar y optimizar un modelo de Machine Learning (**XGBoost**) capaz de estimar el precio con alta precisión.
3.  **Desarrollo de API:** Construir un backend con **FastAPI** que exponga el modelo y permita la gestión de datos (CRUD) y consultas estadísticas.
4.  **Interfaz de Usuario:** Crear un Dashboard web intuitivo que permita a los usuarios interactuar con el sistema sin necesidad de saber programación.
## Herramientas y Proceso del proyecto
### Data Science/Backend:
* **Python:** Lenguaje principal.
* **Pandas & NumPy:** Manipulación y limpieza de datos.
* **Scikit-Learn & XGBoost:** Entrenamiento del modelo predictivo y pipelines de procesamiento.
* **Joblib:** Persistencia del modelo entrenado.
* **FastAPI & Uvicorn:** Creación del servidor API REST y despliegue asíncrono.
* **SQLite:** Base de datos relacional ligera para almacenar reservas e historial de predicciones.

### Interfaz Web/Frontend:
* **HTML & CSS:** Estructura y diseño.
* **Bootstrap 5:** Diseño responsivo y componentes visuales modernos.
* **JavaScript (Vanilla):** Lógica del usuario y conexión con la API (Fetch API).
* **Chart.js:** Visualización de gráficos estadísticos.

## Origen de los datos
Los datos utilizados en este estudio provienen del dataset **“hotel_bookings.csv”**, recopilado de Kaggle. Este dataset se emplea exclusivamente con fines académicos y no para usos reales.

## Instalación y Ejecución
1. **Preparar entorno**: Recomendable usar un entorno virtual (conda o venv). pip install -r requirements.txt
2. **Inicializar la Base de Datos**: Crea las tablas necesarias ("reservas" y "historial_predicciones"). python Database_Reservas_Hotel.py
3. **Arrancar la API**: Verás un mensaje indicando que el servidor corre en: http://127.0.0.1:8000. python main.py
4. **Usar la Aplicación**: La intefaz web se conectará automáticamente a tu API local. index.html

## Trabajo Universitario
Este proyecto ha sido desarrollado como trabajo académico para fines de aprendizaje y evaluación universitaria. Su objetivo es demostrar el flujo completo de un proyecto de Data Science, Machine Learning, desarrollo de API e interfaz web interactivo. 
**Nota:** No está pensado para un entorno de producción real sin ajustes adicionales de seguridad y escalabilidad.
