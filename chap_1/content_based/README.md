## Implémentation Content-Based Filtering

Cette partie est dédiée à l'implémentation d'un système de recommandation basé sur le *content-based filtering*.

On utilisera deux datasets :

1. **Dataset films** : dataset personnel construit à partir de données agrégées via Wikipedia, Wikidata et nettoyées via l'intelligence artificielle Llama3.2:3B  
2. **Dataset historique utilisateurs** : dataset externe (non construit personnellement). Un objectif de ce travail est de construire notre propre dataset d'historique utilisateur. Pour l'instant, nous utilisons un dataset Kaggle.

Le fichier principal de cette partie est `main.py`.

### Prérequis Kaggle
Pour utiliser `main.py`, inscrivez-vous sur [Kaggle](https://www.kaggle.com) et générez une clé API :  
[https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication)

Placez votre **username** et **clé API** dans le fichier `.env` à la racine du repo (voir `.env.example`).
