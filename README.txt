
## Analyse sentimentale de tweets

Ce TP entraîne des classifieurs de tweets (texte) pour les catégoriser comme ayant un sentiment 'positif' ou 'négatif'. 

## Objectifs

Les objectifs principaux sont :
1. Extraction des caractéristiques (feature extraction) du jeu de données de tweets d'entraînement et de test
2. Construire, comparer et évaluer cinq classifieurs différents : arbre de décision, forêt aléatoire, AdaBoost, GBoost et Bagging.
3. Choisir le modèle le plus performant et faire la prédiction de sentiments sur le jeu de données tests

## Contexte

Apprentissage automatique
Classification
Analyse sentimentale

## Fichiers inclus

Pour le prétraitement et l'extraction des caractéristiques:

| Fichier  | Utilité |
| -------- | ------- |
| test.csv | données brutes : tweet et sentiment (étiquettes) |
| train.csv | données brutes : tweet, sans sentiment |
| test_data.csv | données prétraitées avec caractéristiques |
| emojis_emoticons.txt | lexique personnalisé d'émojis et d'émoticônes positifs et négatifs |
| stopwords.txt | lexique personnalisé de stopwords |
| data_preparation.py | script pour générer test_data.csv et training_data.csv |

Pour l'entraînement et l'évaluation des modèles:
- training_data.csv
- test_predictions.csv
- training.py

| Fichier  | Utilité |
| -------- | ------- |
| training_data.csv | données prétraitées avec caractéristiques, sans sentiment |
| test_predictions.csv | données brutes : tweet, sans sentiment |
| training.py | script pour construire et évaluer les classifieurs, générer test_predictions.csv |


## Mise en place

Si ce n'est pas déjà fait, télécharger la version adaptée de Python pour votre OS [ici](https://www.python.org/downloads/). Python3.10 est minimalement requis.

1. Télécharger le dossier. Changer le working directory à ce dossier. 

2. Télécharger le module requis:
```
pip install afinn
```

3. Créer et activer un environnement virtuel:
```
python3 -m venv venv
```
```
source venv/bin/activate
```

4. Télécharger les lexiques requis:

```
import nltk
nltk.download('vader_lexicon')
nltk.download('opinion_lexicon')
nltk.download('punkt')
```

## Usage

Une fois la mise en place complétée (section précédente). S'assurer que tous les fichiers se trouvent dans le même dossier (voir section fichiers inclus). 

1. Faire le prétraitement des données. Dans le dossier:
```
python data_preparation.py
```
Pour faire le prétraitement des données tests, il suffit de changer le nom du input_csv (ligne 88 de data_preparation.py) à test.csv. Similairement, changer le nom du output_csv (ligne 89).

2. Construire et évaluer les classifieurs:
```
python training.py
```

## Technologies utilisées

- Python 3.12
- afinn 0.2.dev0
- pandas 2.2.3
- seaborn 0.13.2
- matplotlib 3.9.2
- nltk 3.9.1
- scikit-learn 1.6.1
- numpy 2.0.2 


## Auteure
- Émilie Roy
