# Projet de Machine Learning à l'UQAC

## Crawler

Ce fichier vous permet de récupérer des fichiers midi de jeux vidéo NES, ou des fichier midi des jeux Final Fantasy.

## Preprocess


## Performance RNN Magenta

Nous avons dans un premier temps généré des musiques via le model Performance_RNN de Magenta (avec la configuration de base 'performance'). Avant de donner notre dataset de fichiers midi à ce model nous les avons prétraité (avec seulement l'option permettant de réccuperer des fichiers midi en fin de preprocess). Puis nous avons laissé s'entrainer le model pendant 15 heures.


## Model proposé


## Classeur

Nous avons développé un classeur permettant de classer les musiques généré par nos modèles en trois catégories :
* REAL MUSIC : Catégorie entrainé avec des musiques de jeux NES (sauf Final Fantasy)
* Final Fantasy : Catégorie entrainé avec des musiques de jeux Final Fantasy
* RANDOM : Catégorie entrainé avec des musiques généré par performance RNN en début d'apprentissage. 

De cette manière nous pouvons catégoriser (avec un niveau de certitudes) les musiques que notre modèle génère.

## Résultats