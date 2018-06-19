# Classeur Random Forest

Pour entrainer votre classeur et faire des prédictions vous devez IMPERATIVEMENT utiliser python 2. 
Ensuite il vous faudra suivre les trois étapes suivantes

## Extract_features

Dans un dossier vous devez préparer vos fichier midi de la manière suivante :

.\Dossier\GENRE\midi\fichiers.mid

Avec un dossier GENRE par genre de musique à classer.

/!\ Les fichiers midi doivent être prétraité de la même manière que les fichiers midi que vous allez donner à votre réseaux (une option permet au script de prétraitement de donner des fichiers midi en sortie)

Ensuite lancer extract_features.py sur votres dossier (cela prend beaucoup de temps).

## Train_model

Lancer le script avec les paramètres demandé, et vous obtiendrait 3 fichiers .pkl permettant au prochain script de faire de la prédiction

## Classifier

Vous donner à ce script un dossier de fichier midi, un dossier de sortie, et un dossier où sont stockés les .pkl . Si les fichiers midi n'ont pas ecore de feature associées il faut ajouter l'option --feature_extraction dans la commande.

Ensuite les fichiers vont être classé seulement si le classeur à plus de 90% de confiance dans sa prédiction (par défaut, vous pouvez le modifier avec la commande --proba 0.9 pour 90% de confiance). Vous trouverez les fichiers ainsi classé dans le dossié de sortie indiqué, est classé par dossier de prédiction (les classes). Leurs pourcentage de confiance lors de la prédiction est indiqué sauvegardé dans le nom des fichiers.