"""
INF7370 - Automne 2025
Émilie Roy
Remise 28 octobre 2025
TP1 Classification sentimentale de tweets

script 2/2
Entraînement, comparaison et analyse des modèles classifieurs
"""

import pandas as pd
import os
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Changing directory to script path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
test_df = pd.read_csv("test_data.csv", encoding="latin1",nrows=500)
test_df.head(10)


##### Helper functions #####

def perform_GridSearchCV(model, param_grid, x_train, y_train, cv=5):
    """
    Perform GridSearchCV with user parameters and return best model and its parameters according to cross-validation results
    """
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv)
    grid_search.fit(x_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_params, best_model

def evaluate_model(model, x_test, y_test):
    """
    Compute accuracy, precision, recall and F1 score for a trained model. Return metrics as dict
    """
    y_pred = model.predict(x_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    report = classification_report(y_test, y_pred)
    print(f"\nPerformance of {model.__class__.__name__}: {report}")
    return metrics

def train_and_evaluate(model, param_grid, x_train, y_train, x_test, y_test, model_name):
    """
    Perform GridSearchCV, evaluate the best model, and return best model, its parameters, and its metrics.
    """
    best_params, best_model = perform_GridSearchCV(model, param_grid, x_train, y_train)
    metrics = evaluate_model(best_model, x_test, y_test)
    return model_name, best_model, best_params, metrics

def plot_feature_importances(model, feature_names, model_name):
    """
    Plot feature importances for a trained tree-based model
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort descending

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
    plt.title(f"Feature Importances - {model_name}")
    plt.tight_layout()
    plt.show()


def plot_combined_feature_importances(models, model_names, feature_names):
    """
    Combine and plot feature importances from multiple tree-based models.
    ----------
    models : list
        List of trained models (must have .feature_importances_ attribute)
    model_names : list
        Names corresponding to each model
    feature_names : list
        Names of the features used in training
    """
    # Collect feature importances
    all_importances = pd.DataFrame({
        name: model.feature_importances_ for name, model in zip(model_names, models)
    }, index=feature_names)

    # Normalize to make models comparable
    all_importances = all_importances / all_importances.sum()

    # Sort features by their mean importance across models
    all_importances['mean_importance'] = all_importances.mean(axis=1)
    all_importances = all_importances.sort_values('mean_importance', ascending=False)

    # Melt for seaborn plotting
    melted = all_importances.drop(columns='mean_importance').reset_index().melt(
        id_vars='index', var_name='Model', value_name='Importance')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='index', y='Importance', hue='Model')
    plt.xticks(rotation=45)
    plt.title("Comparaison des modèles : importance des caractéristiques")
    plt.xlabel("Caractéristique")
    plt.ylabel("Importance normalisée")
    plt.tight_layout()
    plt.show()



##### Split training data #####

# Chose a 80/20 split for training data / test data

complete_df = pd.read_csv("training_data.csv", encoding="latin1")

x_train, x_test, y_train, y_test = train_test_split(
    complete_df[["afinn", 'nltk', 'nbPosWords', 'nbNegWords', 'nbPosEmojis',
                 'nbNegEmojis', 'nbPeriods', 'nbExcla', 'nbInterog']],
                 complete_df['Sentiment'], test_size=0.2, random_state=17)

print(f"Train size x: {len(x_train)} | Train size y: {len(y_train)}")
print(f"Test size x: {len(x_test)} | Test size y: {len(y_test)}")



##### Train models #####

# Setup base classifier (Decision tree)
dt_params = {
    'max_depth': [None, 3, 5, 8, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4, 6, 8]
    }

tree_params, best_tree = perform_GridSearchCV(tree.DecisionTreeClassifier(random_state=17), dt_params, x_train, y_train)
print(tree_params)
best_tree_depth = tree_params['max_depth']
print(best_tree_depth)

# Setup parameter grids for GridSearchCV
rf_params = {'n_estimators': [100, 200, 300, 400]}

ada_params = {
    'n_estimators': [50, 100, 250],
    'learning_rate' : [0.01, 0.1, 1.0]
}

gboost_params = {
    'n_estimators': [100, 250, 500],
    'learning_rate' : [0.01, 0.1, 1.0]
}

bag_params = {'n_estimators': [10, 15, 20, 25]}

# Initialize models
models = [
    ("Random Forest", RandomForestClassifier(max_depth = best_tree_depth, random_state = 17), rf_params),
    ("AdaBoost", AdaBoostClassifier(estimator = clone(best_tree), random_state = 17), ada_params),
    ("Gradient Boost", GradientBoostingClassifier(max_depth = best_tree_depth, random_state = 17), gboost_params),
    ("Bagging", BaggingClassifier(estimator = clone(best_tree), random_state = 17), bag_params)
]

# Evaluate and train models
results = {}

for name, model, params in models:
    model_name, best_model, best_params, metrics = train_and_evaluate(model, params, x_train, y_train, x_test, y_test, name)
    results[model_name] = metrics

    # Store best models in dedicated variables
    if name == "Random Forest":
        best_rf = best_model
    elif name == "AdaBoost":
        best_ada = best_model
    elif name == "Gradient Boost":
        best_gboost = best_model
    elif name == "Bagging":
        best_bag = best_model

# Add in evaluation of base classifier (Decision Tree)
results["Decision Tree"] = evaluate_model(best_tree, x_test, y_test)

print(results)



##### Figures #####

#-- Correlation matrix between features
# Compute correlation matrix
corr_matrix = complete_df[["afinn", "nltk", "nbPosWords", "nbNegWords",
                           "nbPosEmojis", "nbNegEmojis",
                           "nbPeriods", "nbExcla", "nbInterog"]].corr()
# Display heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", center=0, fmt=".2f")
plt.title("Matrice de corrélation des caractéristiques")
plt.show()


#-- Feature importance
feature_names = ["afinn", "nltk", "nbPosWords", "nbNegWords",
                 "nbPosEmojis", "nbNegEmojis", "nbPeriods", "nbExcla", "nbInterog"]
models = [best_tree, best_rf, best_ada, best_gboost]
model_names = ["Arbre de décision", "Forêt aléatoire", "AdaBoost", "GBoost"]
plot_combined_feature_importances(models, model_names, feature_names)

#-- Confusion matrices
models = {
    "Arbre de décision": best_tree,
    "Forêt aléatoire": best_rf,
    "AdaBoost": best_ada,
    "GBoost": best_gboost,
    "Bagging": best_bag
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

# Plot all confusion matrices without colorbars
for ax, (name, model) in zip(axes, models.items()):
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=["Négatif (0)", "Positif (1)"],
        cmap="YlGnBu",
        colorbar=False,
        ax=ax
    )
    ax.set_title(name, fontsize=12)
    ax.grid(False)
    ax.set_xlabel("")  # remove duplicate labels
    ax.set_ylabel("")

# Hide extra subplot (the 6th one)
for ax in axes[len(models):]:
    ax.set_visible(False)

# Add shared colorbar
im = axes[0].images[0]
fig.subplots_adjust(right=0.88, wspace=0.4, hspace=0.4)
cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.4])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel("Nombre de tweets", rotation=270, labelpad=15)

# Add shared axis labels (once for all subplots)
fig.text(0.5, 0.04, "Classe prédite", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Classe réelle", va="center", rotation="vertical", fontsize=12)

# Add global title and adjust layout
plt.tight_layout(rect=[0.05, 0.05, 0.88, 0.95])
plt.show()



#### APPLYING THE CHOSEN MODEL TO TEST DATA #####

# Read test data
test_df = pd.read_csv("test_data.csv", encoding="latin1",skiprows=range(1,100001)) #the first 100k lines in test_data.csv are the same as trainging_data.csv
test_df.head(10)
print(test_df.columns)

# Prediction
feature_cols = ["afinn", "nltk", "nbPosWords", "nbNegWords",
                "nbPosEmojis", "nbNegEmojis", "nbPeriods", "nbExcla", "nbInterog"]

test_df["PredictedSentiment"] = best_bag.predict(test_df[feature_cols])

print(test_df[["SentimentText", "PredictedSentiment"]].head(20))

# Save predictions
pred_df = test_df[["ItemID", "SentimentText", "PredictedSentiment"]]
pred_df.to_csv("test_predictions.csv", index=False)



#### SAMPLING TWEETS FOR MANUAL EVALUATION #####

positive_tweets = pred_df[pred_df["PredictedSentiment"] == 1]
negative_tweets = pred_df[pred_df["PredictedSentiment"] == 0]

# Sample 20 from each
sample_pos = positive_tweets.sample(n=20, random_state=45)
sample_neg = negative_tweets.sample(n=20, random_state=45)

# Combine into a single DataFrame
sampled_tweets = pd.concat([sample_pos, sample_neg]).reset_index(drop=True)
print(sampled_tweets[["SentimentText", "PredictedSentiment"]])