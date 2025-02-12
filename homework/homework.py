# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import json

# Paso 1: Cargar y limpiar los datos
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], inplace=True)
    df = df.dropna()
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: x if x <= 4 else 4)
    return df

train_file = "files/input/train_data.csv"
test_file = "files/input/test_data.csv"
df_train = load_and_clean_data(train_file)
df_test = load_and_clean_data(test_file)

# Paso 2: Separar variables predictoras y objetivo
x_train, y_train = df_train.drop(columns=['default']), df_train['default']
x_test, y_test = df_test.drop(columns=['default']), df_test['default']

# Paso 3: Construcción del pipeline
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features = list(set(x_train.columns) - set(categorical_features))

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaler', MinMaxScaler(), numerical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Paso 4: Optimización de hiperparámetros
cv_scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='balanced_accuracy')
print(f'Balanced Accuracy (CV mean): {np.mean(cv_scores):.4f}')

# Entrenar el modelo final
pipeline.fit(x_train, y_train)

# Paso 5: Guardar el modelo
model_path = "files/models/model.pkl.gz"
with gzip.open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

# Paso 6: Calcular métricas
def compute_metrics(model, x, y, dataset_type):
    y_pred = model.predict(x)
    metrics = {
    'type': 'metrics',
    'dataset': dataset_type,
    'precision': float(precision_score(y, y_pred)),
    'balanced_accuracy': float(balanced_accuracy_score(y, y_pred)),
    'recall': float(recall_score(y, y_pred)),
    'f1_score': float(f1_score(y, y_pred))
    }
    return metrics

metrics_train = compute_metrics(pipeline, x_train, y_train, 'train')
metrics_test = compute_metrics(pipeline, x_test, y_test, 'test')

# Paso 7: Calcular matrices de confusión
def compute_confusion_matrix(model, x, y, dataset_type):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
    'type': 'cm_matrix',
    'dataset': dataset_type,
    'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
    'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }
    return cm_dict

cm_train = compute_confusion_matrix(pipeline, x_train, y_train, 'train')
cm_test = compute_confusion_matrix(pipeline, x_test, y_test, 'test')

# Guardar métricas y matrices de confusión
metrics_path = "files/output/metrics.json"
with open(metrics_path, 'w') as f:
    for item in [metrics_train, metrics_test, cm_train, cm_test]:
        f.write(json.dumps(item) + "\n")

print("Modelo entrenado y métricas guardadas.")

