import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data():
    """
    Charge et nettoie les données des fichiers bottle.csv et cast.csv
    """
    # Chemins des fichiers
    data_dir = Path("../Data")
    bottle_path = data_dir / "bottle.csv"
    cast_path = data_dir / "cast.csv"
    
    # Chargement des données avec spécification des types
    print("Chargement de bottle.csv...")
    bottle_data = pd.read_csv(bottle_path, dtype={
        'Depthm': 'float64',
        'T_degC': 'float64',
        'Salnty': 'float64',
        'O2ml_L': 'float64',
        'STheta': 'float64'
    })
    
    print("Chargement de cast.csv...")
    cast_data = pd.read_csv(cast_path, dtype={
        'Sta_ID': 'str',
        'Lat_Dec': 'float64',
        'Lon_Dec': 'float64'
    })
    
    # Nettoyage des données bottle
    print("Nettoyage des données bottle...")
    bottle_data = clean_bottle_data(bottle_data)
    
    # Nettoyage des données cast
    print("Nettoyage des données cast...")
    cast_data = clean_cast_data(cast_data)
    
    return bottle_data, cast_data

def clean_bottle_data(df):
    """
    Nettoie les données bottle
    """
    # Sélection des colonnes pertinentes
    cols_to_keep = ['Depthm', 'T_degC', 'Salnty', 'O2ml_L', 'STheta']
    df = df[cols_to_keep].copy()
    
    # Conversion des types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Renommage des colonnes
    df.columns = ['Profondeur', 'Temperature', 'Salinite', 'Oxygene', 'Densite']
    
    # Suppression des valeurs manquantes
    df = df.dropna()
    
    # Suppression des valeurs aberrantes (méthode IQR)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

def clean_cast_data(df):
    """
    Nettoie les données cast
    """
    # Sélection des colonnes pertinentes
    cols_to_keep = ['Sta_ID', 'Date', 'Lat_Dec', 'Lon_Dec']
    df = df[cols_to_keep].copy()
    
    # Renommage des colonnes
    df.columns = ['Station', 'Date', 'Latitude', 'Longitude']
    
    # Conversion des types
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Suppression des valeurs manquantes
    df = df.dropna()
    
    return df 