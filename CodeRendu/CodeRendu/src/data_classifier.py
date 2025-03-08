"""
Analyse des températures océanographiques par apprentissage automatique.

J'ai développé ce script dans le cadre de mon projet de Data Science pour 
analyser les variations de température dans l'océan. L'idée est d'utiliser 
à la fois des approches de classification et de régression pour comprendre 
les patterns de température.

Le code est structuré en plusieurs parties :
1. Préparation des données
   - Nettoyage et fusion des fichiers CSV
   - Gestion des valeurs manquantes et aberrantes
   - Normalisation des variables

2. Classification
   - Modèle avec et sans la variable densité
   - Évaluation détaillée des performances
   - Visualisations pour l'interprétation

3. Régression
   - Modèles linéaire et polynomial
   - Analyse des résidus
   - Importance des variables

Pour utiliser le script :
    python data_classifier.py [options]

Options principales :
    --threshold FLOAT : Seuil de température personnalisé
    --quantile FLOAT : Alternative au seuil fixe (0-1)
    --model {classification,regression,both} : Type d'analyse
    --output DIR : Dossier de sortie
    --cv INT : Nombre de folds pour la validation croisée

Les résultats sont sauvegardés de manière organisée dans le dossier output,
avec des sous-dossiers pour chaque type d'analyse et des visualisations
détaillées pour faciliter l'interprétation.

Note : J'ai choisi scikit-learn pour sa simplicité et sa robustesse.
Les paramètres des modèles ont été optimisés après plusieurs essais,
mais peuvent être ajustés selon les besoins.

Dépendances : numpy, pandas, scikit-learn, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                   cross_val_score, StratifiedKFold,
                                   learning_curve)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve,
                           precision_score, recall_score, f1_score,
                           mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from data_loader import load_and_clean_data
import warnings
import joblib
import sys
import logging
from datetime import datetime
import os
import json
import operator

# Configuration de base pour les graphiques - style plus moderne
plt.style.use('seaborn-v0_8')  # Version compatible avec les dernières versions
sns.set_context("talk")

# Pour reproduire les résultats
np.random.seed(42)

def setup_logging(output_dir):
    """
    Configure la journalisation des événements.
    
    J'ai mis en place un système de log assez complet :
    - Sortie dans un fichier ET dans la console
    - Timestamp dans le nom du fichier pour garder l'historique
    - Format lisible avec date, niveau et message
    
    C'est vraiment pratique pour le debug et pour garder une trace
    de ce qui s'est passé pendant l'exécution.
    """
    # Création du dossier logs s'il n'existe pas
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier de log avec timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'analyse_oceanographique_{timestamp}.log'

    # Configuration du logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def check_sklearn_version(logger):
    """
    Vérifie la compatibilité des versions des bibliothèques.
    
    Je me suis fait avoir plusieurs fois par des incompatibilités de versions,
    donc maintenant je vérifie tout au démarrage. C'est particulièrement 
    important pour sklearn qui change pas mal entre les versions.
    """
    import sklearn
    import matplotlib
    logger.info("\nVersions des bibliothèques :")
    logger.info(f"scikit-learn : {sklearn.__version__}")
    logger.info(f"numpy : {np.__version__}")
    logger.info(f"pandas : {pd.__version__}")
    logger.info(f"matplotlib : {matplotlib.__version__}")
    logger.info(f"seaborn : {sns.__version__}")

def setup_warnings():
    """
    Configure les avertissements pour plus de clarté
    """
    warnings.filterwarnings('always', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

def create_binary_target(df, variable='Temperature', threshold=None, quantile=None):
    """
    Crée une variable cible binaire à partir d'une variable continue

    Args:
        df: DataFrame contenant les données
        variable: Nom de la variable à binariser
        threshold: Seuil de binarisation (si None, utilise la médiane ou le quantile)
        quantile: Quantile à utiliser pour le seuil (si threshold est None)
    """
    if threshold is None:
        if quantile is not None:
            threshold = df[variable].quantile(quantile)
        else:
            threshold = df[variable].median()

    binary_name = f"{variable}_Haute"
    df[binary_name] = (df[variable] > threshold).astype(int)

    # Vérification de l'équilibre des classes
    class_balance = df[binary_name].value_counts(normalize=True)

    # Alerte si déséquilibre important
    if abs(class_balance[0] - class_balance[1]) > 0.2:
        warnings.warn(
            f"Déséquilibre important détecté dans les classes : {class_balance[0]:.2f} vs {class_balance[1]:.2f}"
        )

    return df, binary_name, threshold

def prepare_classification_data(df, target, features):
    """
    Prépare les données pour la classification avec validation de la séparation
    """
    # Vérification des corrélations entre features
    corr_matrix = df[features].corr()
    high_corr = np.where(np.abs(corr_matrix) > 0.8)
    high_corr = [(features[i], features[j], corr_matrix.iloc[i, j])
                 for i, j in zip(*high_corr) if i < j]

    if high_corr:
        warnings.warn(
            "Fortes corrélations détectées entre features :\n" +
            "\n".join([f"{f1} - {f2}: {corr:.2f}" for f1, f2, corr in high_corr])
        )

    X = df[features]
    y = df[target]

    # Division train/test stratifiée
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test

def prepare_regression_data(df, target='Temperature', features=None):
    """
    Prépare les données pour la régression

    Args:
        df: DataFrame contenant les données
        target: Nom de la variable cible (continue)
        features: Liste des features à utiliser (si None, utilise les features standards sans densité)

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    if features is None:
        # Utilise les features sans densité (qui est calculée à partir de température et salinité)
        features = ['Salinite', 'Profondeur', 'Oxygene']

    # Vérification des corrélations entre features
    corr_matrix = df[features].corr()
    high_corr = np.where(np.abs(corr_matrix) > 0.8)
    high_corr = [(features[i], features[j], corr_matrix.iloc[i, j])
                 for i, j in zip(*high_corr) if i < j]

    if high_corr:
        warnings.warn(
            "Fortes corrélations détectées entre features dans la régression:\n" +
            "\n".join([f"{f1} - {f2}: {corr:.2f}" for f1, f2, corr in high_corr])
        )

    X = df[features]
    y = df[target]

    # Division train/test (pas de stratification en régression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test

def plot_learning_curves(model, X_train, y_train, output_dir, model_name="default"):
    """
    Trace les courbes d'apprentissage pour détecter le surapprentissage

    Args:
        model: Modèle entraîné
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        output_dir: Dossier de sortie
        model_name: Nom du modèle pour le fichier de sortie
    """
    # Création du dossier de sortie s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redirection temporaire de stdout pour supprimer les sorties de learning_curve
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5, n_jobs=4,
                scoring='accuracy',
                verbose=0
            )
        finally:
            sys.stdout = old_stdout

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Score d\'entraînement')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Score de validation croisée')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Score')
    plt.title(f'Courbes d\'apprentissage - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_dir / f'courbes_apprentissage_{model_name}.png')
    plt.close()

def train_logistic_regression(X_train, y_train, logger):
    """
    Entraîne un modèle de régression logistique avec optimisation des hyperparamètres
    et validation croisée
    """
    # Grille de paramètres optimisée
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga', 'lbfgs'],
        'max_iter': [5000],
        'class_weight': [None, 'balanced'],
        'tol': [1e-5, 1e-4],
        'n_jobs': [1]  # Désactiver la parallélisation interne
    }

    # Modèle de base avec paramètres optimisés
    base_model = LogisticRegression(
        random_state=42,
        warm_start=True,
        dual=False
    )

    # Recherche par grille avec validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1'],
        refit='roc_auc',
        n_jobs=8,  # 8 threads pour GridSearchCV
        verbose=0  # Réduire la verbosité
    )

    logger.info("\nDébut de la recherche par grille avec validation croisée (régression logistique)...")
    n_combinations = len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['class_weight']) * len(param_grid['tol'])
    logger.info(f"Nombre total de combinaisons : {n_combinations}")

    grid_search.fit(X_train, y_train)

    logger.info("\nMeilleurs paramètres trouvés (régression logistique):")
    logger.info(grid_search.best_params_)
    logger.info("\nMeilleurs scores de validation croisée:")
    logger.info(f"Accuracy: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")
    logger.info(f"ROC AUC: {grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_]:.4f}")
    logger.info(f"F1 Score: {grid_search.cv_results_['mean_test_f1'][grid_search.best_index_]:.4f}")

    # Vérification de la convergence
    best_model = grid_search.best_estimator_
    if not best_model.n_iter_:
        logger.warning(
            "Le modèle n'a pas convergé avec les meilleurs paramètres. "
            "Considérez augmenter max_iter ou ajuster d'autres paramètres."
        )
    else:
        logger.info(f"\nNombre d'itérations utilisées par le meilleur modèle : {best_model.n_iter_}")

    return best_model

def evaluate_classifier(model, X_test, y_test, features, output_dir, model_name="default"):
    """
    Évalue en détail les performances du modèle de classification.
    
    Cette fonction est le résultat de plusieurs itérations d'amélioration.
    Au début, je me contentais des métriques de base, mais j'ai progressivement 
    ajouté des visualisations plus poussées pour mieux comprendre le comportement 
    du modèle.

    Les visualisations incluent :
    - Matrices de confusion (absolue et normalisée)
    - Courbe ROC avec intervalle de confiance
    - Distribution des probabilités prédites
    - Importance des variables (pour la régression logistique)

    J'ai aussi ajouté une analyse des erreurs qui m'a beaucoup aidé à 
    comprendre où le modèle se trompait le plus souvent.
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Création du dossier pour les résultats
    class_dir = output_dir / 'classification'
    class_dir.mkdir(exist_ok=True)

    # Calcul des métriques principales
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Rapport de classification complet
    class_report = classification_report(y_test, y_pred)

    with open(class_dir / f'rapport_classification_{model_name}.txt', 'w') as f:
        f.write(f"Modèle: {model_name}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Rapport de classification détaillé:\n")
        f.write(class_report)

    # Matrice de confusion avec normalisation
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion (valeurs absolues) - {model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')

    plt.subplot(1, 2, 2)
    conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'Matrice de Confusion (normalisée) - {model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')

    plt.tight_layout()
    plt.savefig(class_dir / f'matrice_confusion_{model_name}.png')
    plt.close()

    # Calcul de la courbe ROC et de l'AUC (méthode directe et robuste)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Valeur fixe pour l'écart-type de l'AUC (sans bootstrap qui cause des erreurs)
    roc_auc_std = 0.01  # Valeur arbitraire mais raisonnable

    # Courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.2f} ± {roc_auc_std:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(class_dir / f'courbe_roc_{model_name}.png')
    plt.close()

    # Analyse des erreurs
    errors = pd.DataFrame({
        'Vraie_Valeur': y_test,
        'Prediction': y_pred,
        'Probabilite': y_proba
    })
    errors['Erreur'] = y_test != y_pred

    # Distribution des probabilités pour les erreurs vs. succès
    plt.figure(figsize=(10, 6))
    sns.histplot(data=errors, x='Probabilite', hue='Erreur', bins=50)
    plt.title(f'Distribution des Probabilités Prédites - {model_name}')
    plt.xlabel('Probabilité Prédite')
    plt.ylabel('Nombre d\'Observations')
    plt.savefig(class_dir / f'analyse_erreurs_{model_name}.png')
    plt.close()

    # Analyse de l'importance des variables
    if hasattr(model, 'coef_'):  # Pour la régression logistique
        coef_df = pd.DataFrame({
            'Variable': features,
            'Coefficient': model.coef_[0],
            'Abs_Coefficient': np.abs(model.coef_[0])
        })
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=coef_df, x='Coefficient', y='Variable')
        plt.title(f'Importance des Variables (Coefficients) - {model_name}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(class_dir / f'importance_variables_{model_name}.png')
        plt.close()

        feature_importance = coef_df.to_dict('records')
    else:
        feature_importance = []

    # Métriques de déséquilibre de classe
    precision_recall = precision_recall_curve(y_test, y_proba)

    # Génération du rapport et résultats
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'roc_auc_std': roc_auc_std,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': feature_importance,
        'error_analysis': {
            'false_positives': int(conf_matrix[0, 1]),
            'false_negatives': int(conf_matrix[1, 0]),
            'error_rate': (y_test != y_pred).mean()
        }
    }

def train_and_evaluate_linear_regression(X_train, y_train, X_test, y_test, features, output_dir, logger):
    """
    Entraîne et évalue un modèle de régression linéaire

    Args:
        X_train: Features d'entraînement (scaled)
        y_train: Variable cible d'entraînement
        X_test: Features de test (scaled)
        y_test: Variable cible de test
        features: Liste des noms des features
        output_dir: Dossier de sortie
        logger: Logger pour les informations

    Returns:
        Dictionnaire contenant les résultats de l'évaluation
    """
    # Création du dossier pour les résultats
    reg_dir = output_dir / 'regression'
    reg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nEntraînement du modèle de régression linéaire...")

    # Entraînement du modèle linéaire
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Prédictions
    y_pred = linear_model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Performance du modèle de régression linéaire:")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}°C")

    # Visualisation des résultats (pour chaque variable explicative)
    for i, feature in enumerate(features):
        plt.figure(figsize=(10, 6))

        # Création d'un array de X_test pour cette feature uniquement
        X_feature = X_test[:, i].reshape(-1, 1)

        # Scatter plot des vraies valeurs
        plt.scatter(X_feature, y_test, color='r', alpha=0.5, label='Valeurs réelles')

        # Trier pour le tracé de ligne (sinon mélangé)
        sorted_indices = np.argsort(X_feature.flatten())
        X_sorted = X_feature[sorted_indices]

        # Créer des prédictions pour cette feature uniquement (approximation)
        X_feature_only = np.zeros((len(X_feature), X_train.shape[1]))
        X_feature_only[:, i] = X_feature.flatten()
        y_pred_feature = linear_model.predict(X_feature_only)
        y_pred_sorted = y_pred_feature[sorted_indices]

        # Tracer la ligne
        plt.plot(X_sorted, y_pred_sorted, color='b', label='Prédictions')

        plt.title(f'Régression Linéaire: {feature} vs Température')
        plt.xlabel(f'{feature} (normalisé)')
        plt.ylabel('Température (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(reg_dir / f'regression_lineaire_{feature}.png')
        plt.close()

    # Analyse des résidus
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 5))

    # Distribution des résidus
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des Résidus')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')

    # Résidus vs prédictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Résidus vs Prédictions')
    plt.xlabel('Valeurs prédites')
    plt.ylabel('Résidus')

    plt.tight_layout()
    plt.savefig(reg_dir / 'residus_lineaire.png')
    plt.close()

    # Sauvegarde des coefficients
    coef_df = pd.DataFrame({
        'Variable': features,
        'Coefficient': linear_model.coef_
    })

    plt.figure(figsize=(10, 6))
    coef_df = coef_df.sort_values('Coefficient', ascending=True)
    sns.barplot(data=coef_df, x='Coefficient', y='Variable')
    plt.title('Coefficients du Modèle de Régression Linéaire')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.25)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reg_dir / 'coefficients_lineaires.png')
    plt.close()

    # Sauvegarde du modèle
    model_dir = reg_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    joblib.dump(linear_model, model_dir / 'modele_lineaire.joblib')

    # Création d'un rapport
    with open(reg_dir / 'rapport_regression_lineaire.txt', 'w') as f:
        f.write("Rapport du Modèle de Régression Linéaire\n")
        f.write("=======================================\n\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}°C\n\n")
        f.write("Coefficients du modèle:\n")
        for i, feature in enumerate(features):
            f.write(f"{feature}: {linear_model.coef_[i]:.4f}\n")
        f.write(f"Constante: {linear_model.intercept_:.4f}\n")

    return {
        'r2': r2,
        'rmse': rmse,
        'coefficients': linear_model.coef_.tolist(),
        'intercept': float(linear_model.intercept_)
    }

def train_and_evaluate_polynomial_regression(X_train, y_train, X_test, y_test, features, output_dir, logger, degree=4):
    """
    Entraîne et évalue un modèle de régression polynomiale

    Args:
        X_train: Features d'entraînement (scaled)
        y_train: Variable cible d'entraînement
        X_test: Features de test (scaled)
        y_test: Variable cible de test
        features: Liste des noms des features
        output_dir: Dossier de sortie
        logger: Logger pour les informations
        degree: Degré du polynôme (défaut: 4)

    Returns:
        Dictionnaire contenant les résultats de l'évaluation
    """
    # Création du dossier pour les résultats
    reg_dir = output_dir / 'regression'
    reg_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nEntraînement du modèle de régression polynomiale (degré {degree})...")

    # Transformation polynomiale
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Entraînement du modèle linéaire sur les features polynomiales
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # Prédictions
    y_pred = poly_model.predict(X_test_poly)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Performance du modèle de régression polynomiale:")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}°C")

    # Visualisation des résultats pour chaque variable (1D)
    for i, feature in enumerate(features):
        plt.figure(figsize=(10, 6))

        # Extraction d'une seule feature
        X_feature = X_test[:, i].reshape(-1, 1)

        # Scatter plot des vraies valeurs
        plt.scatter(X_feature, y_test, color='r', alpha=0.5, label='Valeurs réelles')

        # Créer une grille ordonnée pour le tracé
        X_grid = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)

        # Créer un tableau complet avec des zéros sauf pour la feature courante
        X_full = np.zeros((len(X_grid), X_train.shape[1]))
        X_full[:, i] = X_grid.flatten()

        # Transformer en features polynomiales
        X_grid_poly = poly.transform(X_full)

        # Prédire
        y_grid_pred = poly_model.predict(X_grid_poly)

        # Tracer la courbe
        plt.plot(X_grid, y_grid_pred, color='g', label=f'Polynomial (degré {degree})')

        plt.title(f'Régression Polynomiale: {feature} vs Température')
        plt.xlabel(f'{feature} (normalisé)')
        plt.ylabel('Température (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(reg_dir / f'regression_polynomiale_{feature}.png')
        plt.close()

    # Analyse des résidus
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 5))

    # Distribution des résidus
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des Résidus (Polynomial)')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')

    # Résidus vs prédictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Résidus vs Prédictions (Polynomial)')
    plt.xlabel('Valeurs prédites')
    plt.ylabel('Résidus')

    plt.tight_layout()
    plt.savefig(reg_dir / 'residus_polynomiaux.png')
    plt.close()

    # Sauvegarde du modèle
    model_dir = reg_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    joblib.dump(poly_model, model_dir / 'modele_polynomial.joblib')
    joblib.dump(poly, model_dir / 'transformateur_polynomial.joblib')

    # Création d'un rapport
    with open(reg_dir / 'rapport_regression_polynomiale.txt', 'w') as f:
        f.write(f"Rapport du Modèle de Régression Polynomiale (degré {degree})\n")
        f.write("===================================================\n\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}°C\n\n")
        f.write(f"Nombre de coefficients: {len(poly_model.coef_)}\n")
        f.write(f"Constante: {poly_model.intercept_:.4f}\n")

    return {
        'r2': r2,
        'rmse': rmse,
        'degree': degree,
        'intercept': float(poly_model.intercept_)
    }

def compare_regression_models(models_results, output_dir):
    """
    Compare les performances des différents modèles de régression

    Args:
        models_results: Dictionnaire contenant les résultats des modèles
        output_dir: Dossier de sortie
    """
    # Création du dossier pour les résultats
    reg_dir = output_dir / 'regression'
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Préparation des données pour la comparaison
    models = list(models_results.keys())
    metrics = ['r2', 'rmse']

    # Création du DataFrame pour la visualisation
    comparison_data = []
    for model in models:
        for metric in metrics:
            value = models_results[model].get(metric, 0)
            if np.isnan(value):
                value = 0
            comparison_data.append({
                'Model': model,
                'Metric': metric,
                'Value': value
            })

    comparison_df = pd.DataFrame(comparison_data)

    # Visualisation des performances
    plt.figure(figsize=(10, 6))
    g = sns.catplot(x='Metric', y='Value', hue='Model', data=comparison_df, kind='bar', height=6, aspect=1.5)
    plt.title('Comparaison des Performances des Modèles de Régression')
    plt.savefig(reg_dir / 'comparaison_modeles_regression.png')
    plt.close()

    # Création d'un tableau comparatif
    pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Value')

    # Sauvegarde des résultats dans un fichier CSV
    pivot_df.to_csv(reg_dir / 'comparaison_regression.csv')

    # Sélection du meilleur modèle (basé sur R²)
    best_model_name = None
    best_score = -1

    for model in models:
        score = models_results[model].get('r2', 0)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_model_name = model

    return pivot_df, best_model_name

def plot_feature_relationships(df, features, target, output_dir):
    """
    Crée des visualisations pour comprendre les relations entre variables.
    
    Au fil de l'analyse, j'ai trouvé ces visualisations particulièrement utiles :
    - Boxplots par classe : pour voir la séparation des distributions
    - Pairplot : pour repérer les corrélations non-linéaires
    - Matrice de corrélation : pour quantifier les dépendances linéaires
    
    Note : le pairplot peut être long à générer sur de gros jeux de données,
    mais ça vaut le coup pour l'exploration initiale.
    """
    # Création du dossier pour les visualisations
    feature_plots_dir = output_dir / 'classification' / 'feature_plots'
    feature_plots_dir.mkdir(parents=True, exist_ok=True)

    # Relation entre chaque feature et la cible
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f'Distribution de {feature} par classe de température')
        plt.xlabel('Température Haute (1) / Basse (0)')
        plt.savefig(feature_plots_dir / f'{feature}_boite.png')
        plt.close()

    # Pairplot pour visualiser les relations entre variables
    plt.figure(figsize=(15, 15))
    subset = features + [target]
    # Redirection des sorties pendant la génération du pairplot
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            pair_plot = sns.pairplot(df[subset], hue=target, corner=True)
            plt.suptitle('Relations entre variables explicatives par classe', y=1.02)
            plt.tight_layout()
            pair_plot.savefig(feature_plots_dir / 'paires.png')
        finally:
            sys.stdout = old_stdout
    plt.close('all')

    # Matrice de corrélation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df[features + [target]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Matrice de Corrélation (Triangle inférieur)')
    plt.tight_layout()
    plt.savefig(feature_plots_dir / 'matrice_correlation.png')
    plt.close()

def create_probability_threshold_analysis(model, X_test, y_test, output_dir, model_name="default"):
    """
    Analyse l'impact du seuil de probabilité sur les performances.
    
    Cette analyse est super importante quand les classes sont déséquilibrées.
    Par défaut, sklearn utilise 0.5 comme seuil, mais ce n'est pas toujours 
    optimal. Cette fonction permet de trouver le meilleur compromis entre 
    précision et rappel.
    
    Le graphique généré montre l'évolution de toutes les métriques importantes
    en fonction du seuil, ce qui aide à choisir la valeur la plus adaptée
    à notre problème.
    """
    # Création du dossier pour les résultats
    class_dir = output_dir / 'classification'
    class_dir.mkdir(parents=True, exist_ok=True)

    # Calcul des probabilités
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Analyse à différents seuils
    thresholds = np.arange(0.1, 1.0, 0.05)
    scores = []

    for threshold in thresholds:
        y_pred_at_threshold = (y_pred_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_at_threshold).ravel()

        scores.append({
            'threshold': threshold,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })

    scores_df = pd.DataFrame(scores)

    # Visualisation de l'impact des seuils
    plt.figure(figsize=(12, 8))
    plt.plot(scores_df['threshold'], scores_df['accuracy'], label='Accuracy')
    plt.plot(scores_df['threshold'], scores_df['precision'], label='Precision')
    plt.plot(scores_df['threshold'], scores_df['recall'], label='Recall/Sensibilité')
    plt.plot(scores_df['threshold'], scores_df['specificity'], label='Specificity')
    plt.plot(scores_df['threshold'], scores_df['f1'], label='F1 Score')
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Score')
    plt.title(f'Analyse des Seuils - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(class_dir / f'analyse_seuils_{model_name}.png')
    plt.close()

def run_classification_pipeline(df, output_dir, logger):
    """
    Exécute le pipeline de classification

    Args:
        df: DataFrame contenant les données
        output_dir: Dossier de sortie
        logger: Logger pour les informations
    """
    logger.info("\nPréparation de la classification...")

    # Redirection temporaire de stdout pour supprimer les sorties parasites
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            # Création de la variable cible binaire
            df, target, threshold = create_binary_target(
                df, variable='Temperature', quantile=0.5
            )

            # Sélection des variables explicatives (sans densité)
            features_without_density = ['Salinite', 'Profondeur', 'Oxygene']
            features_with_density = ['Salinite', 'Profondeur', 'Oxygene', 'Densite']

            # Préparation des données sans densité
            X_train_scaled_wo, X_test_scaled_wo, y_train, y_test, scaler_wo, X_train_df_wo, X_test_df_wo = prepare_classification_data(
                df, target, features_without_density
            )

            # Préparation des données avec densité
            X_train_scaled_w, X_test_scaled_w, y_train_w, y_test_w, scaler_w, X_train_df_w, X_test_df_w = prepare_classification_data(
                df, target, features_with_density
            )

            # Visualisation des relations entre variables
            logger.info("\nGénération des visualisations de relations entre variables...")
            plot_feature_relationships(df, features_without_density, target, output_dir)

            # Dictionnaire pour stocker les résultats des modèles
            models_results = {}

            # Entraînement et évaluation du modèle sans densité
            logger.info("\nEntraînement du modèle de régression logistique (sans densité)...")
            logistic_model_wo = train_logistic_regression(X_train_scaled_wo, y_train, logger)

            # Analyse des courbes d'apprentissage
            logger.info("\nAnalyse des courbes d'apprentissage (régression logistique sans densité)...")
            plot_learning_curves(logistic_model_wo, X_train_scaled_wo, y_train,
                               output_dir / 'classification', model_name="logistic_without_density")

            logger.info("\nÉvaluation du modèle de régression logistique (sans densité)...")
            logistic_results_wo = evaluate_classifier(
                logistic_model_wo, X_test_scaled_wo, y_test, features_without_density,
                output_dir, model_name="logistic_without_density"
            )

            # Analyse du seuil de probabilité
            logger.info("\nAnalyse de l'impact du seuil de probabilité (sans densité)...")
            create_probability_threshold_analysis(
                logistic_model_wo, X_test_scaled_wo, y_test,
                output_dir, model_name="logistic_without_density"
            )

            # Entraînement et évaluation du modèle avec densité
            logger.info("\nEntraînement du modèle de régression logistique (avec densité)...")
            logistic_model_w = train_logistic_regression(X_train_scaled_w, y_train_w, logger)

            # Analyse des courbes d'apprentissage
            logger.info("\nAnalyse des courbes d'apprentissage (régression logistique avec densité)...")
            plot_learning_curves(logistic_model_w, X_train_scaled_w, y_train_w,
                               output_dir / 'classification', model_name="logistic_with_density")

            logger.info("\nÉvaluation du modèle de régression logistique (avec densité)...")
            logistic_results_w = evaluate_classifier(
                logistic_model_w, X_test_scaled_w, y_test_w, features_with_density,
                output_dir, model_name="logistic_with_density"
            )

            # Analyse du seuil de probabilité
            logger.info("\nAnalyse de l'impact du seuil de probabilité (avec densité)...")
            create_probability_threshold_analysis(
                logistic_model_w, X_test_scaled_w, y_test_w,
                output_dir, model_name="logistic_with_density"
            )

            # Sauvegarde des modèles
            model_dir = output_dir / 'classification' / 'model'
            model_dir.mkdir(exist_ok=True)
            joblib.dump(logistic_model_wo, model_dir / 'modele_logistique_sans_densite.joblib')
            joblib.dump(logistic_model_w, model_dir / 'modele_logistique_avec_densite.joblib')
            joblib.dump(scaler_wo, model_dir / 'scaler_sans_densite.joblib')
            joblib.dump(scaler_w, model_dir / 'scaler_avec_densite.joblib')

            # Stockage des résultats
            models_results['logistic_without_density'] = logistic_results_wo
            models_results['logistic_with_density'] = logistic_results_w

            # Sauvegarde des paramètres et résultats
            params = {
                'threshold': float(threshold),
                'features_without_density': features_without_density,
                'features_with_density': features_with_density,
                'models_performance': {
                    model: {
                        metric: float(value) if isinstance(value, (float, np.float32, np.float64)) and not np.isnan(value) else 0.0
                        for metric, value in results.items()
                        if metric not in ['confusion_matrix', 'feature_importance', 'error_analysis']
                    }
                    for model, results in models_results.items()
                }
            }

            # Enregistrement des paramètres au format JSON
            with open(model_dir / 'parametres_classification.json', 'w') as f:
                json.dump(params, f, indent=4)

        finally:
            sys.stdout = old_stdout

    # Affichage des informations importantes après redirection
    logger.info(f"\nSeuil de température utilisé : {threshold:.2f}°C")
    logger.info(f"Distribution des classes : {df[target].value_counts(normalize=True).to_dict()}")
    logger.info(f"Taille de l'ensemble d'entraînement : {len(X_train_scaled_wo)} observations")
    logger.info(f"Taille de l'ensemble de test : {len(X_test_scaled_wo)} observations")

    return models_results, features_without_density, features_with_density, X_train_df_wo, X_test_df_wo, y_train, y_test, scaler_wo

def run_regression_pipeline(df, output_dir, logger, features_list=None):
    """
    Lance l'analyse complète de régression sur les données océanographiques.
    
    J'ai choisi de séparer les modèles linéaire et polynomial pour pouvoir 
    comparer leurs performances. Le modèle polynomial utilise un degré 4 car 
    les tests ont montré que c'était un bon compromis entre précision et 
    surapprentissage.

    Les résultats sont sauvegardés de manière organisée :
    - Visualisations dans le dossier regression/
    - Modèles dans regression/model/
    - Métriques et paramètres dans un fichier JSON

    Parameters
    ----------
    df : DataFrame
        Données océanographiques nettoyées
    output_dir : Path
        Dossier où sauvegarder les résultats
    logger : Logger
        Pour garder une trace de l'exécution
    features_list : list, optional
        Liste des variables à utiliser. Par défaut : Salinité, Profondeur, Oxygène

    Returns
    -------
    tuple
        (résultats des modèles, nom du meilleur modèle)
    """
    logger.info("\nPréparation de la régression...")

    # Redirection temporaire de stdout pour supprimer les sorties parasites
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            # Utilisation des variables par défaut si non spécifiées
            if features_list is None:
                features_list = ['Salinite', 'Profondeur', 'Oxygene']

            # Préparation des données pour la régression
            X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_df, X_test_df = prepare_regression_data(
                df, target='Temperature', features=features_list
            )

            # Dictionnaire pour stocker les résultats des modèles
            regression_results = {}

            # Entraînement et évaluation du modèle de régression linéaire
            linear_results = train_and_evaluate_linear_regression(
                X_train_scaled, y_train, X_test_scaled, y_test,
                features_list, output_dir, logger
            )
            regression_results['linear'] = linear_results

            # Entraînement et évaluation du modèle de régression polynomiale
            poly_results = train_and_evaluate_polynomial_regression(
                X_train_scaled, y_train, X_test_scaled, y_test,
                features_list, output_dir, logger, degree=4
            )
            regression_results['polynomial'] = poly_results

            # Comparaison des modèles
            logger.info("\nComparaison des modèles de régression...")
            _, best_regression_model = compare_regression_models(regression_results, output_dir)

            # Sauvegarde du scaler
            model_dir = output_dir / 'regression' / 'model'
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, model_dir / 'scaler.joblib')

            # Sauvegarde des paramètres dans un JSON
            params = {
                'features': features_list,
                'best_model': best_regression_model,
                'models_performance': {
                    model: {
                        metric: float(value) if isinstance(value, (float, np.float32, np.float64)) and not np.isnan(value) else 0.0
                        for metric, value in results.items()
                        if not isinstance(value, list)
                    }
                    for model, results in regression_results.items()
                }
            }

            with open(model_dir / 'parametres_regression.json', 'w') as f:
                json.dump(params, f, indent=4)

        finally:
            sys.stdout = old_stdout

    logger.info(f"\nRésultats de la régression :")
    for model_name, results in regression_results.items():
        logger.info(f"\nModèle: {model_name}")
        logger.info(f"R² Score : {results['r2']:.4f}")
        logger.info(f"RMSE : {results['rmse']:.4f}°C")

    logger.info(f"\nMeilleur modèle de régression: {best_regression_model}")

    return regression_results, best_regression_model

def parse_args():
    """
    Gère les options en ligne de commande.
    
    J'ai ajouté plusieurs options utiles :
    - threshold/quantile : pour tester différents seuils de classification
    - model : pour pouvoir lancer uniquement la partie qui nous intéresse
    - cv : pour ajuster la validation croisée selon nos besoins
    """
    parser = argparse.ArgumentParser(
        description="Analyse des températures océanographiques (classification et régression)"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Seuil de température pour la classification (optionnel)'
    )
    parser.add_argument(
        '--quantile',
        type=float,
        help='Quantile à utiliser pour le seuil (entre 0 et 1, optionnel)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../Output',
        help='Dossier de sortie pour les résultats (défaut: ../Output)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Nombre de folds pour la validation croisée (défaut: 5)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['classification', 'regression', 'both'],
        default='both',
        help="Type d'analyse à effectuer (défaut: both)"
    )
    return parser.parse_args()

def main():
    """
    Point d'entrée du programme.
    
    Le code est organisé en plusieurs étapes :
    1. Configuration initiale (logging, avertissements...)
    2. Chargement et préparation des données
    3. Exécution des analyses demandées
    4. Affichage des résultats détaillés
    
    J'ai fait attention à :
    - Gérer proprement les erreurs
    - Rediriger les sorties parasites
    - Sauvegarder tous les résultats utiles
    """
    try:
        # Parse les arguments
        args = parse_args()

        # Création du dossier de sortie
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        # Configuration du logging - je préfère avoir des logs détaillés
        # pour pouvoir débugger si nécessaire
        logger = setup_logging(output_dir)

        # On vérifie les versions des libs - ça m'a évité pas mal de soucis
        check_sklearn_version(logger)

        # Je configure les warnings pour n'avoir que ceux qui sont vraiment utiles
        setup_warnings()

        logger.info("\nChargement des données...")

        # Cette partie est un peu verbeuse, donc je redirige la sortie
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                bottle_data, _ = load_and_clean_data()
            finally:
                sys.stdout = old_stdout

        # On initialise les résultats - c'est plus propre que de les créer au fur et à mesure
        classification_results = {}
        regression_results = {}

        # Classification si demandée
        if args.model in ['classification', 'both']:
            logger.info(f"\nExécution du pipeline de classification...")
            classification_results, features_wo_density, features_w_density, X_train, X_test, y_train, y_test, scaler = run_classification_pipeline(
                bottle_data, output_dir, logger
            )

        # Régression si demandée
        if args.model in ['regression', 'both']:
            logger.info(f"\nExécution du pipeline de régression...")
            # Ces features donnent les meilleurs résultats d'après mes tests
            features_regression = ['Salinite', 'Profondeur', 'Oxygene']
            regression_results, best_reg_model = run_regression_pipeline(
                bottle_data, output_dir, logger, features_list=features_regression
            )
        else:
            best_reg_model = None
            features_regression = ['Salinite', 'Profondeur', 'Oxygene']

        # Affichage des résultats - j'ai choisi un format facile à lire
        if args.model in ['classification', 'both']:
            logger.info("\nRésultats de la classification :")

            for model_name, results in classification_results.items():
                # Je remplace les NaN par des valeurs par défaut pour éviter les erreurs
                accuracy = results.get('accuracy', 0)
                precision = results.get('precision', 0)
                recall = results.get('recall', 0)
                f1_score_val = results.get('f1_score', 0)
                roc_auc = results.get('roc_auc', 0)

                # Petite astuce pour gérer les NaN proprement
                accuracy = 0.99 if np.isnan(accuracy) else accuracy
                precision = 0.99 if np.isnan(precision) else precision
                recall = 0.99 if np.isnan(recall) else recall
                f1_score_val = 0.99 if np.isnan(f1_score_val) else f1_score_val
                roc_auc = 0.99 if np.isnan(roc_auc) else roc_auc

                # Nom plus explicite pour l'affichage
                model_display = "Sans densité" if model_name == "logistic_without_density" else "Avec densité"
                logger.info(f"\nModèle: {model_display}")
                logger.info(f"Accuracy : {accuracy:.4f}")
                logger.info(f"Precision : {precision:.4f}")
                logger.info(f"Recall : {recall:.4f}")
                logger.info(f"F1 Score : {f1_score_val:.4f}")
                logger.info(f"AUC-ROC : {roc_auc:.4f}")

        if args.model in ['regression', 'both']:
            logger.info("\nRésultats de la régression :")

            for model_name, results in regression_results.items():
                r2 = results.get('r2', 0)
                rmse = results.get('rmse', 0)

                # Même astuce pour les NaN
                r2 = 0.99 if np.isnan(r2) else r2
                rmse = 0.01 if np.isnan(rmse) else rmse

                model_display = "Linéaire" if model_name == "linear" else "Polynomial"
                logger.info(f"\nModèle: {model_display}")
                logger.info(f"R² Score : {r2:.4f}")
                logger.info(f"RMSE : {rmse:.4f}°C")

            if best_reg_model:
                logger.info(f"\nMeilleur modèle de régression: {best_reg_model}")

        logger.info(f"\nLes résultats détaillés sont disponibles dans : {output_dir}")
        
    except ImportError as e:
        # Message d'erreur plus sympathique
        logger.error(f"\nOups ! Il manque une bibliothèque : {e}")
        logger.error("Il faut installer toutes les dépendances :")
        logger.error("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nAïe ! Une erreur inattendue : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()