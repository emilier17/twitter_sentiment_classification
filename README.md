
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

Pour le prétraitement et l'extraction des caractéristiques (data_preparation.py):

| Fichier  | Type    | Utilité |
| -------- | ------- | ------- |
| test.csv | Input | données brutes : tweet et sentiment (étiquettes) |
| train.csv | Input | données brutes : tweet, sans sentiment |
| emojis_emoticons.txt | Input | lexique personnalisé d'émojis et d'émoticônes positifs et négatifs |
| stopwords.txt | Input | lexique personnalisé de stopwords |
| test_data.csv | Output | données prétraitées avec caractéristiques |
| training_data.csv | Output | données prétraitées avec caractéristiques, sans sentiment |

Pour l'entraînement et l'évaluation des modèles classifieurs (training.py):

| Fichier  | Type    | Utilité |
| -------- | ------- | ------- |
| test_data.csv | Input | données prétraitées avec caractéristiques |
| training_data.csv | Input | données prétraitées avec caractéristiques, sans sentiment |
| test_predictions.csv | Output | test_data.csv avec prédictions de sentiment |


## Mise en place

Si ce n'est pas déjà fait, télécharger la version adaptée de Python pour votre OS [ici](https://www.python.org/downloads/). Python3.10 est minimalement requis.

1. Télécharger le dossier. Changer le working directory à ce dossier. 

2. Créer et activer un environnement virtuel:
Unix:
```
python3 -m venv venv
source venv/bin/activate
```
Windows (command prompt):
```
python -m venv venv
venv\Scripts\activate.bat
```

3. Télécharger les modules requis:
```
pip install -r requirements.txt
```

4. Ouvrir une session interactive python:
```
python
```

4. Télécharger les lexiques requis dans la session interactive python:

```
import nltk
nltk.download('vader_lexicon')
nltk.download('opinion_lexicon')
nltk.download('punkt')
```

5. Quitter la session interactive python:
```
exit()
```

## Usage

Une fois la mise en place complétée (section précédente). S'assurer que tous les fichiers se trouvent dans le même dossier (voir section fichiers inclus). 

1. Faire le prétraitement des données. Dans l'environnement virtuel:
```
python data_preparation.py
```
Pour faire le prétraitement des données tests, il suffit de changer le nom du input_csv (ligne 88 de data_preparation.py) à 'test.csv'. Similairement, changer le nom du output_csv (ligne 89).

2. Construire et évaluer les classifieurs:
```
python training.py
```

3. Quitter l'environnement virtuel:
```
deactivate
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
