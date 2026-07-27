## Implémentation Content-Based Filtering

Cette partie contient l'implémentation d'un système de recommandation basé sur le *content-based filtering* en utilisant le dataset **MovieLens 20M**.

Le fichier principal pour exécuter le programme est `main.py`.

### Prérequis (Kaggle API)

Pour télécharger automatiquement le dataset, le programme a besoin de s'authentifier à Kaggle. 

1. Créez un compte sur [Kaggle](https://www.kaggle.com/) si ce n'est pas déjà fait.
2. Allez dans vos paramètres de compte (*Settings* -> *API TOKEN* -> *Generate New Token*) pour récupérer votre nom d'utilisateur et votre clé API.
3. À la racine du projet, créez un fichier `.env`.
4. Ajoutez-y ces deux variables avec vos informations :

```env
KAGGLE_USERNAME=xxxxx
KAGGLE_API_TOKEN=xxxxx
```
### Lancement

Une fois le fichier `.env` configuré, vous pouvez lancer le programme :

```bash
python main.py
```