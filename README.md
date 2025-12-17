# Titanic Survival Prediction - Logistic Regression from Scratch

Implémentation manuelle de la régression logistique multivariée pour prédire la survie des passagers du Titanic.

## 📋 Description

Ce projet implémente une régression logistique multivariée **from scratch** (sans utiliser de bibliothèques de machine learning comme scikit-learn) pour prédire si un passager du Titanic a survécu ou non. Le modèle est entraîné sur le dataset Titanic en utilisant uniquement NumPy et Pandas.

## 🎯 Fonctionnalités

- **Préprocessing des données** :
  - Gestion des valeurs manquantes (Age, Embarked)
  - Transformation des variables catégorielles (Sex, Ticket, Embarked) en variables binaires (one-hot encoding)
  - Normalisation des caractéristiques numériques

- **Implémentation de la régression logistique** :
  - Fonction hypothèse (sigmoid)
  - Fonction de coût (log-likelihood)
  - Calcul du gradient
  - Descente de gradient pour l'optimisation

- **Évaluation** :
  - Calcul de la précision (accuracy)
  - Suivi du coût pendant l'entraînement

## 📦 Dépendances

```bash
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.3.0
jupyter >= 1.0.0
```

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/titanic-logistic-regression.git
cd titanic-logistic-regression
```

2. Installez les dépendances :
```bash
pip install numpy pandas matplotlib jupyter
```

3. Téléchargez les données :
   - Placez les fichiers `train.csv` et `test.csv` du dataset Titanic dans le répertoire du projet
   - Les données sont disponibles sur [Kaggle](https://www.kaggle.com/c/titanic/data)

## 📊 Utilisation

1. Ouvrez le notebook Jupyter :
```bash
jupyter notebook multivariate_logistic_regression.ipynb
```

2. Exécutez toutes les cellules pour :
   - Charger et explorer les données
   - Préparer les données (nettoyage, transformation)
   - Entraîner le modèle de régression logistique
   - Évaluer les performances

## 📈 Résultats

Le modèle atteint une précision d'environ **79%** sur l'ensemble d'entraînement après 100,000 itérations avec un taux d'apprentissage de 0.001.

## 🔧 Structure du Code

- **Préprocessing** : Sélection des caractéristiques, gestion des valeurs manquantes, encodage one-hot
- **Normalisation** : Normalisation min-max des caractéristiques numériques
- **Fonctions du modèle** :
  - `hypothesis(X, theta)` : Calcule la probabilité de survie
  - `cost(X, Y, theta)` : Calcule la fonction de coût (log-likelihood)
  - `gradient(X, Y, theta)` : Calcule le gradient
  - `gradient_descent(X, Y, lr, steps)` : Entraîne le modèle
  - `accuracy(X, Y, theta)` : Calcule la précision

## 📝 Caractéristiques Utilisées

- **Pclass** : Classe du passager (1, 2, 3)
- **Sex** : Sexe (male, female)
- **Age** : Âge
- **SibSp** : Nombre de frères/sœurs/époux à bord
- **Parch** : Nombre de parents/enfants à bord
- **Ticket** : Catégorie du ticket (transformée)
- **Fare** : Prix du billet
- **Embarked** : Port d'embarquement (C, Q, S)

## 🎓 Apprentissage

Ce projet est conçu à des fins éducatives pour comprendre :
- Le fonctionnement interne de la régression logistique
- L'implémentation de l'algorithme de descente de gradient
- Le preprocessing des données pour le machine learning
- L'évaluation des modèles de classification

## 📄 Licence

Ce projet est libre d'utilisation à des fins éducatives.

## 👤 Auteur

Projet réalisé dans le cadre du cours d'Automatisation Supervisée (Supervised Learning)

---

**Note** : Les fichiers `train.csv` et `test.csv` ne sont pas inclus dans le repository. Veuillez les télécharger depuis [Kaggle](https://www.kaggle.com/c/titanic/data).

