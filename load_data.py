import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tempfile
import shutil

def GetDataKaggle():

    #Variables de entorno de credenciales de Kaggle
    os.environ['KAGGLE_USERNAME'] = 'fabian426'
    os.environ['KAGGLE_KEY'] = 'e7eb2c70bc652c11cb51980cce0b6d8c'
    competition = 'playground-series-s4e10'
    
    # Inicia la API de Kaggle
    api = KaggleApi()
    api.authenticate()

    # Define la ruta donde se guardarán los archivos, relativa a la raíz del proyecto
    data_path = os.path.join("data")

    # Crea la ruta si no existe
    os.makedirs(data_path, exist_ok=True)

    # Descargar los archivos train.csv y test.csv a la ruta especificada
    competition_name = "playground-series-s4e10"
    api.competition_download_file(competition_name, "train.csv", path=data_path)
    api.competition_download_file(competition_name, "test.csv", path=data_path)

    # Ruta completa al archivo train.csv
    train_file_path = os.path.join(data_path, "train.csv")

    return train_file_path