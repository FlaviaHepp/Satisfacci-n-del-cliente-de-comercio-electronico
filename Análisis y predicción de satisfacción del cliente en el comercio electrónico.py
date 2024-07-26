"""Descripción general
Bienvenido al cuaderno de análisis de satisfacción del cliente de comercio electrónico de Shopzilla. Este conjunto de datos completo 
captura puntuaciones de satisfacción del cliente durante un período de un mes en la plataforma de comercio electrónico Shopzilla, una 
entidad seudónima. Con 85,907 filas y 20 columnas, este conjunto de datos proporciona una rica fuente para realizar tareas de análisis de 
datos exploratorios (EDA), visualización y clasificación de aprendizaje automático. El conjunto de datos se ha fabricado cuidadosamente 
utilizando la biblioteca Faker para garantizar la ocultación de detalles genuinos y al mismo tiempo mantener su utilidad para obtener 
información valiosa.

Objetivo
Este conjunto de datos sirve como un recurso sólido para evaluar el desempeño del servicio al cliente, pronosticar niveles de satisfacción 
y realizar análisis del comportamiento del cliente dentro del sector del comercio electrónico. La información contenida en el conjunto de 
datos incluye características cruciales como el nombre del canal, detalles del pedido, comentarios de los clientes, información de los 
agentes y, lo más importante, puntuaciones de satisfacción del cliente (CSAT).

Descripción de datos
El conjunto de datos abarca varios aspectos de las interacciones con los clientes, incluido el identificador único para cada registro, el 
nombre del canal, la información del pedido, las marcas de tiempo para informar y responder a los problemas, las respuestas a las encuestas 
de los clientes, los detalles del producto, las métricas de desempeño de los agentes y las puntuaciones CSAT. Se proporciona una 
descripción detallada de los datos, lo que garantiza claridad sobre el significado de cada columna y ayuda a una comprensión integral del 
conjunto de datos.
Este cuaderno está diseñado para facilitar las tareas de clasificación de aprendizaje automático, visualización y análisis de datos 
exploratorios. Ya sea que esté interesado en comprender la dinámica del servicio al cliente, predecir la satisfacción del cliente o 
analizar el desempeño de los agentes, este cuaderno proporciona una base sólida para sus conocimientos basados ​​en datos.
"""
#Importar bibliotecas
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder ,OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
#ExploracióndeDatos
df= pd.read_csv('Customer_support_data.csv')
df

df.info()

df.dtypes

df.describe()

df.duplicated().sum()

df.isna().sum()

#Limpieza de datos
#Varias columnas contienen valores faltantes:

"Customer Remarks"# tiene 57,165 entradas faltantes.
"Order_id,"
"order_date_time,"
"Customer_City,"
"Product_category,"
"Item_price," and "connected_handling_time" #tienen distintos grados de datos faltantes.
# Eliminar columnas innecesarias
df_cleaned = df.drop(["Customer Remarks", "Order_id", "order_date_time"], axis=1)

# Imputar valores faltantes para características numéricas
df_cleaned["Item_price"].fillna(df_cleaned["Item_price"].median(), inplace=True)
df_cleaned["connected_handling_time"].fillna(df_cleaned["connected_handling_time"].median(), inplace=True)

# Imputar valores faltantes para características categóricas
df_cleaned["Customer_City"].fillna("Unknown", inplace=True)
df_cleaned["Product_category"].fillna("Unknown", inplace=True)


# Convertir columnas de marca de tiempo al formato de fecha y hora
timestamp_columns = ["Issue_reported at", "issue_responded", "Survey_response_Date"]
for column in timestamp_columns:
    df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
    
# Imputar valores faltantes para funciones de marca de tiempo
for column in ["Issue_reported at", "issue_responded"]:
    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
    
df_cleaned.isna().sum()

df_cleaned.info()

#Visualización de datos
# Establecer tema oscuro para Plotly
template = "plotly_dark"

# Visualice la distribución de las puntuaciones CSAT por turno de agente de forma interactiva
csat_distribution_by_shift_fig = px.box(df_cleaned, x='Agent Shift', y='CSAT Score', color='Agent Shift',
                                         title='Distribución de puntuaciones CSAT por turno de agente\n',
                                         labels={'Agent Shift': 'Turno de agente', 'CSAT Score': 'Puntuación CSAT'},
                                         template=template)
csat_distribution_by_shift_fig.update_layout(
    xaxis=dict(title='Turno de agente'),
    yaxis=dict(title='Puntuación CSAT'),
    font=dict(family='Arial', size=12, color='white')
)

# Mostrar gráfico interactivo
csat_distribution_by_shift_fig.show()

# Distribución de puntuaciones CSAT
csat_distribution_fig = px.histogram(df_cleaned, x='CSAT Score', title='Distribución de puntuaciones CSAT\n',
                                     labels={'CSAT Score': 'Puntuación CSAT'},
                                     color='CSAT Score', template=template)
csat_distribution_fig.update_layout(
    xaxis=dict(title='Puntuación CSAT', tickmode='linear'),
    yaxis=dict(title='Conteo'),
    font=dict(family='Arial', size=12, color='white')
)

# Mostrar gráfico interactivo
csat_distribution_fig.show()

# Gráfico circular de distribución de canales.
pie_chart_fig = px.pie(df_cleaned, names='channel_name', title='Distribución de canales\n',
                       labels={'channel_name': 'Canal'},
                       template=template)
pie_chart_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
pie_chart_fig.show()


# Gráfico de líneas de series temporales de puntuaciones CSAT a lo largo del tiempo
time_series_fig = px.line(df_cleaned, x='Survey_response_Date', y='CSAT Score', title='Puntuaciones CSAT a lo largo del tiempo\n',
                           labels={'Survey_response_Date': 'Fecha', 'CSAT Score': 'Puntuación CSAT'},
                           template=template)
time_series_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
#time_series_fig.show()

# Gráfico Sunburst de distribución de categorías y subcategorías
sunburst_chart_fig = px.sunburst(df_cleaned, path=['category', 'Sub-category'], title='Distribución de categorías y subcategorías\n',
                                 labels={'category': 'Categoria', 'Sub-category': 'Subcategoria'},
                                 template=template)
sunburst_chart_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
sunburst_chart_fig.show()

# Extraer información del mes de Survey_response_Date
df_cleaned['Month'] = df_cleaned['Survey_response_Date'].dt.to_period('M')

# Distribución mensual de CSAT
monthly_csat_dist_fig = px.histogram(df_cleaned, x='CSAT Score', color='CSAT Score', facet_col='Month',
                                      title='Distribución CSAT mensual\n',
                                      labels={'CSAT Score': 'Puntuación CSAT', 'Month': 'Mes'},
                                      template=template)
monthly_csat_dist_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
monthly_csat_dist_fig.show()

# Agentes principales según el gráfico de barras de puntuación CSAT
top_agents_csat_bar_fig = px.bar(df_cleaned.groupby('Agent_name')['CSAT Score'].mean().reset_index().nlargest(10, 'CSAT Score'),
                                  x='Agent_name', y='CSAT Score', title='Principales agentes por puntuación CSAT\n',
                                  labels={'Agent_name': 'Nombre del agente', 'CSAT Score': 'Puntuación CSAT promedio'},
                                  color='CSAT Score', template=template)
top_agents_csat_bar_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
top_agents_csat_bar_fig.show()

# Distribución del segmento de tenencia del agente
tenure_bucket_dist_fig = px.histogram(df_cleaned, x='Tenure Bucket', title='Distribución del período de tenencia del agente\n',
                                       labels={'Tenure Bucket': 'Cubo de tenencia'},
                                       color='Tenure Bucket', template=template)
tenure_bucket_dist_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Mostrar gráfico interactivo
tenure_bucket_dist_fig.show()

# Puntajes CSAT por canal
csat_by_channel_fig = px.box(df_cleaned, x='channel_name', y='CSAT Score',
                             title='Puntuaciones CSAT por canal\n',
                             labels={'channel_name': 'Canal', 'CSAT Score': 'Puntuación CSAT'},
                             color='channel_name', template=template)
csat_by_channel_fig.update_layout(
    xaxis=dict(title='canal\n'),
    yaxis=dict(title='Puntuación CSAT\n'),
    font=dict(family='Arial', size=12, color='white')
)

# Mostrar gráfico interactivo
csat_by_channel_fig.show()

#Importancia de la característica
# Seleccione características relevantes y variable objetivo
features = df_cleaned[['Item_price', 'connected_handling_time']]
target = df_cleaned['CSAT Score']

# Convertir características categóricas a numéricas usando codificación one-hot
features_encoded = pd.get_dummies(features)

# Dividir los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

# Crear un clasificador de bosque aleatorio
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajustar el modelo a los datos de entrenamiento.
rf_model.fit(X_train, y_train)

# Importancia de la característica de la trama
plt.style.use('dark_background')
plt.figure(figsize=(15, 10))
sns.barplot(x=rf_model.feature_importances_, y=features_encoded.columns)
plt.title('Importancia de la característica: bosque aleatorio\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Puntuación de importancia\n')
plt.ylabel('Características\n')
plt.show()

#Aprendizaje automático
# Predecir puntuaciones CSAT en el conjunto de pruebas
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Exactitud: {accuracy:.2f}')
print('Informe de clasificación:\n', classification_rep)

# Puntuaciones de validación cruzada
cross_val_scores = cross_val_score(rf_model, features_encoded, target, cv=5)
print(f'Puntuaciones de validación cruzada: {cross_val_scores}')

"""El modelo funciona bien en la predicción de la clase mayoritaria (puntuación CSAT 5), pero existen desafíos a la hora de predecir las clases minoritarias.
Considere la posibilidad de que la clase aborde el desequilibrio mediante técnicas como el sobremuestreo o el submuestreo.
La ingeniería de funciones o la selección de funciones adicionales pueden mejorar aún más el rendimiento del modelo.
Podría resultar beneficioso ajustar los hiperparámetros del modelo o explorar otros algoritmos."""
# Abordar el desequilibrio de clases utilizando SMOTE (Técnica de sobremuestreo de minorías sintéticas)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_encoded, target)

# Dividir los datos remuestreados en conjuntos de entrenamiento y prueba.
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Crear un clasificador de bosque aleatorio para predicción
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajustar el modelo a los datos de entrenamiento.
rf_model.fit(X_train_rf, y_train_rf)

# Predecir puntuaciones CSAT en el conjunto de pruebas
y_pred_rf = rf_model.predict(X_test_rf)

# Evaluar el modelo
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
classification_rep_rf = classification_report(y_test_rf, y_pred_rf)

print(f'Precisión del modelo de bosque aleatorio: {accuracy_rf:.2f}')
print('Informe de clasificación:\n', classification_rep_rf)

# Puntuaciones de validación cruzada
cross_val_scores_rf = cross_val_score(rf_model, X_resampled, y_resampled, cv=5)
print(f'Puntuaciones de validación cruzada: {cross_val_scores_rf}')

"""Se ha aplicado la técnica SMOTE para abordar el desequilibrio de clases, pero la precisión del modelo ha disminuido. A continuación se 
presentan algunas recomendaciones y observaciones:

Ingeniería de funciones: explore funciones o transformaciones adicionales que podrían capturar mejor patrones en los datos. Considere las 
interacciones entre funciones o la creación de nuevas funciones.

Ajuste de hiperparámetros: ajuste los hiperparámetros del modelo de bosque aleatorio. Utilice técnicas como la búsqueda en cuadrícula o la 
búsqueda aleatoria para encontrar la combinación óptima.

Importancia de las funciones: revise el análisis de importancia de las funciones para asegurarse de que se estén utilizando las funciones 
más relevantes. Puede experimentar con diferentes conjuntos de funciones o probar diferentes métodos de selección de funciones.

Métodos de conjunto: explore otros métodos o algoritmos de conjunto que puedan manejar mejor conjuntos de datos desequilibrados. Gradient 
Boosting, AdaBoost o XGBoost son alternativas potenciales.

Métricas de evaluación del modelo: considere utilizar métricas de evaluación alternativas, como puntuación F1, precisión y recuperación, 
especialmente si existe un desequilibrio de clases significativo. Estas métricas proporcionan una visión más equilibrada del rendimiento 
del modelo.

Ajuste de umbral: ajuste el umbral de probabilidad de clasificación para equilibrar la precisión y la recuperación. Esto es crucial en 
conjuntos de datos desequilibrados.

Técnicas avanzadas: explore técnicas avanzadas, como métodos de conjunto que combinan diferentes modelos, apilan o utilizan arquitecturas 
avanzadas de aprendizaje profundo.

Agrupación"""
df_cleaned.dtypes

# Seleccione características relevantes para la agrupación
features_cluster = df_cleaned[['Item_price', 'connected_handling_time', 'CSAT Score']]

# Estandarizar las características
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features_cluster)

# Aplicar agrupamiento de K-medias
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(features_standardized)

# Visualizar clústeres
fig = plt.figure(figsize=(18, 8))

# Gráfico de dispersión 3D para 'Item_price', 'connected_handling_time' y 'CSAT Score'
plt.style.use('dark_background')
ax = fig.add_subplot(121, projection='3d')
ax.scatter(features_cluster['Item_price'], features_cluster['connected_handling_time'], features_cluster['CSAT Score'], c=df_cleaned['Cluster'], cmap='viridis')
ax.set_xlabel('\nPrecio del articulo\n')
ax.set_ylabel('\nTiempo de manipulación conectado\n')
ax.set_zlabel('\nPuntuación CSAT\n')
ax.set_title('Agrupación 3D de funciones\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Diagrama de dispersión 2D para 'Item_price' y 'connected_handling_time'
plt.style.use('dark_background')
plt.subplot(122)
sns.scatterplot(x='Item_price', y='connected_handling_time', hue='Cluster', data=df_cleaned, palette='viridis')
plt.title('Agrupación del precio del artículo y el tiempo de manipulación conectado\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Precio del articulo\n')
plt.ylabel('Tiempo de manipulación conectado\n')
plt.show()

# Establecer tema oscuro para seaborn
sns.set_theme(style="darkgrid")

# Establecer una paleta de colores para una mejor visualización
palette = sns.color_palette('pastel')

# Visualizar la relación entre Cluster y Agent Shift
plt.style.use('dark_background')
plt.figure(figsize=(14, 8))
sns.countplot(x='Cluster', hue='Agent Shift', data=df_cleaned, palette=palette)
plt.title('Distribución de clústeres por turno de agente\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Grupo\n', fontsize=12)
plt.ylabel('Conteo\n', fontsize=12)
plt.legend(title='Turno de agente\n', loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#import lightgbm as lgb
from lightgbm.sklearn import LGBMRanker
import matplotlib.pyplot as plt

df = pd.read_csv('Customer_support_data.csv')
print(df.columns.tolist())
print(df[0:3])
print(df.info())

df=df.drop('connected_handling_time',axis=1)
df=df.dropna()
print(df.info())

n=len(df)
N=list(range(n))
random.seed(2023)
random.shuffle(N)
df=df.iloc[N].reset_index(drop=True)
from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

df=labelencoder(df)
feature_cols=['Unique id', 'channel_name', 'category', 'Sub-category', 'Customer Remarks', 'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date', 'Customer_City', 'Product_category', 'Item_price','Agent_name', 'Supervisor', 'Manager', 'Tenure Bucket', 'Agent Shift', ]
target= 'CSAT Score'
group_col='Customer Remarks'
dataX=df[feature_cols]
dataY=df['CSAT Score']
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2, random_state=42)
LGBMRanker
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=2,
    importance_type='gain',
)

ranker

#get_session_lengths for group
def get_session_lengths(df, group_col):
    return df.groupby(group_col).size().reset_index(name='session_length')

session_lengths_train = get_session_lengths(trainX,group_col)
session_lengths_train[0:5]

ranker.fit(
    trainX,
    trainY,
    group=np.array(session_lengths_train['session_length'])
)

scores = ranker.predict(testX)
scores

plt.style.use('dark_background')
fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Puntuación prevista frente a puntuación CSAT\n', fontsize = '16', fontweight = 'bold')
ax.set_ylabel('Puntuación prevista\n')
ax.set_xlabel('Puntuación CSAT\n')
ax.scatter(testY,scores,alpha=0.2)
plt.show()

print(len(testX))
testX=testX.reset_index(drop=True)
testY=testY.reset_index(drop=True)

result=pd.concat([testX,testY,pd.DataFrame(data=scores,columns=['score'])],axis=1)
print(result)