import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tempfile
import shutil


class MLSystem:

    def __init__(self):
        self.model = None

    def train_kaggle(data_path):
        
        # Cargar datos desde la ruta
        #df = pd.read_csv(data_path)

        try:
            df = pd.read_csv(data_path, on_bad_lines='skip')  # Ignorar líneas mal formateadas
        except pd.errors.ParserError as e:
            print(f"Error al analizar el archivo CSV: {e}")
            return None, None
        
        # Dividimos los datos entre características y etiquetas (esto dependerá de la estructura del dataset de Kaggle)
        X = df.drop(columns=["loan_status"],axis=1)  # Aquí 'target' es el nombre de la columna de las etiquetas (ajústalo según el dataset)
        y = df['loan_status']

        #Para cambiar las columnas categóricas a numéricas
        label_encoder = LabelEncoder()
        cat_col=X.select_dtypes(exclude=np.number).columns
        for col in cat_col:
            X[col]=label_encoder.fit_transform(X[col])

        # División en datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar un modelo de ejemplo (RandomForest)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Hacer predicciones y calcular precisión
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, accuracy