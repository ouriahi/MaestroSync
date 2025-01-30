# Démarrage Rapide

## Matériel Nécessaire
- Raspberry Pi 4B
- Caméra compatible (Pi Camera v2)

## Environnement
- Python venv

## Les problèmes rencontrés
Nous avons rencontré de grandes difficultés avec les versions des bibliothèques nécessaires. Il était
également problématique que Picamera2 ne puisse pas accéder à libcamera, bien qu’il en ait besoin.
Nous avons donc dû trouver les bonnes versions de chaque bibliothèque pour que la communication
entre elles soit la plus fonctionnelle possible et rendre libcamera accessible dans un environnement
Python.

## Solution Finale
### L’environnement du projet
**Créer l’environnement**

Pour commencer, nous devons créer un environnement virtuel Python pour isoler les dépendances du
projet. Utilisez la commande suivante :
```bash
python3 -m venv env_projet
```
**Installer les bibliothèques**

Activez l’environnement virtuel et installez les bibliothèques nécessaires :
```bash
source env_projet/bin/activate
pip install opencv-python picamera2 mediapipe
```
**Fix de la bibliothèque libcamera**

Pour que Picamera2 puisse accéder à libcamera dans l’environnement, nous devons définir les variables
d’environnement appropriées :
```bash
export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
```
Ces commandes permettent à Picamera2 de trouver les bibliothèques nécessaires pour fonctionner
correctement avec libcamera.
