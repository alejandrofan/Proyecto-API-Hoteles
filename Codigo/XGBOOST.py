#LIBRERIAS
import pandas as pd
import sqlite3
import os
import xgboost
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

conexion=sqlite3.connect("C:/Users/Administrador/OneDrive - ESIC/Proyecto Big Data I/TO Hoteles/Data/hotel_bookings_clean.db")

df=pd.read_sql_query("select * from hotel_bookings_clean", conexion)

conexion.close()

df.info()


#TRANSFORMACIÓN DE CATEGÓRICAS
categoricas= df.select_dtypes(include='object').columns.tolist()
print(categoricas)

df[categoricas].head(5)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop=None)
encoded= ohe.fit_transform(df[categoricas])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categoricas))
print(encoded_df)

numericas_cols = df.select_dtypes(exclude='object').columns.drop('adr')
df_final = pd.concat([df[numericas_cols], encoded_df, df['adr']], axis=1)
print(df_final)


#ENTRENAMIENTO DEL MODELO
x = df_final.drop('adr', axis=1)
y = df_final['adr']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split (x, y, test_size=0.5, random_state=1)

print(x_train.shape, x_test.shape)

x_valid, x_test, y_valid, y_test =train_test_split (x_test, y_test, test_size=0.5, random_state=1)

print(x_train.shape, x_test.shape, x_valid.shape)


#XGBOOST
xgb =xgboost.XGBRegressor()
parametros = {
    'n_estimators': [100, 200],      # ¿Cuántos árboles usar?
    'learning_rate': [0.05, 0.1],    # ¿Qué tan rápido aprende?
    'max_depth': [6],                 # ¿Qué tan profundos son los árboles?
    'random_state': [42],             # Semilla para reproducibilidad
    'n_jobs': [1],                    # ¿Cuántos procesadores usar?
    'objective': ['reg:squarederror'] # Tipo de problema (regresión)
}


from sklearn.model_selection import GridSearchCV
clf = GridSearchCV (xgb, parametros, cv=3, scoring='neg_mean_squared_error',verbose=1)

clf.fit_params = (x_train,y_train,
    'early_stopping_rounds': 10,      # Parar si no mejora en 10 intentos
    'eval_metric': 'rmse',            # Medir el error con RMSE (Error cuadrático medio)
    'eval_set': [(x_valid, y_valid)]    # Datos para evaluar
)













