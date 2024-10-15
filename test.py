import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tempfile
import shutil

import unittest
from unittest.mock import patch, MagicMock

from load_data import GetDataKaggle
from ml import MLSystem

# Clase de prueba
class TestKagglePipeline(unittest.TestCase):

    # Prueba unitaria de GetDataKaggle, para verificar si se autenticó correctamente y sobre todo si se descargó bien los archivos
    @patch('__main__.KaggleApi')
    def test_GetDataKaggle(self, mock_kaggle_api):
        """Probar que GetDataKaggle descarga archivos correctamente utilizando mock"""
        # Crear una instancia simulada de la API de Kaggle
        mock_api_instance = mock_kaggle_api.return_value
        mock_api_instance.competition_download_file = MagicMock()

        # Llamar a la función
        GetDataKaggle()

        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.competition_download_file.assert_any_call('playground-series-s4e10', 'train.csv', path="data")
        mock_api_instance.competition_download_file.assert_any_call('playground-series-s4e10', 'test.csv', path="data")

    # Prueba unitaria para verificar de que entrene bien el modelo
    def test_train_kaggle(self):
        # Crear datos simulados para las pruebas
        data = {
            'feature1': [1, 2, 3, 4, 5,6,7,8,9,10],
            'feature2': ['A', 'B', 'A', 'B','A','C','D','A', 'B','E'],
            'loan_status': [0, 1, 0, 1, 0, 0, 0,1,1,1]
        }
        df = pd.DataFrame(data)
        data_path = os.path.join('data', 'train.csv')
        df.to_csv(data_path, index=False)

        ml_system = MLSystem()

        # Testea el entrenamiento
        model, accuracy = ml_system.train_kaggle(data_path)

        # Asegurarse de que se entrena un modelo y la precisión es válida
        self.assertIsNotNone(model) # asegura que no devuelva none
        self.assertGreaterEqual(accuracy, 0) #al menos que el accuracy sea mayor a 0

    #Prueba para la interacción entre la función de obtener datos y el entrenamiento del modelo
    def test_integration_load_train(self):

        # Ejecutar GetDataKaggle para descargar los datos
        data_path = GetDataKaggle()

        # Verificar que los archivos se descargaron correctamente
        self.assertTrue(os.path.exists(data_path), "El archivo train.csv no se descargó correctamente!")

        ml_system = MLSystem()

        # Entrenar el modelo con los datos descargados
        model, accuracy = ml_system.train_kaggle(data_path)

        # Verificar que se entrene el modelo y se calcule una precisión válida
        self.assertIsNotNone(model, "El modelo no se entrenó correctamente, modelo no válido")
        self.assertGreaterEqual(accuracy, 0, "La precisión del modelo no es válida, menor o igual que 0")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)