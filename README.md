# MaestroSync
Donner le rythme à ceux qui ne peuvent pas le voir
# 🎵 MaestroSync  
*Un projet open-source pour rendre la musique accessible aux malvoyants*  

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-En%20développement-orange)](https://github.com/yourusername/MaestroSync)

---

## 🌟 **Présentation**  
**MaestroSync** est un système innovant qui analyse les gestes d'un chef d'orchestre en temps réel et génère des retours sonores pour permettre aux personnes malvoyantes de ressentir le rythme musical.  

**Idée clé** : Utiliser l'IA pour transformer les mouvements en sons.  

---

## 🎥 **Démonstration**  
![Demo](demo.gif)  
*Détection des mains et feedback sonore en direct*

---

## ✨ **Fonctionnalités**  
- 🖐️ **Suivi des mains** avec MediaPipe  
- 🔊 **Retour audio** personnalisable (bips, voix, rythmes)  
- ⚡ **Traitement en temps réel** sur Raspberry Pi  
- 👁️ **Accessibilité** pour les musiciens malvoyants  

## 🔧 **Fonctionnement interne**

### **Flux de traitement optimisé** :
1. **Thread dédié** pour la capture vidéo (évite les pertes de frames)
2. **Pipeline de traitement** :
   - Conversion RGB → Détection MediaPipe → Calcul de déplacement
3. **Système d'événements** :
   - File d'attente audio (évite les conflits de threads)
   - Mécanisme de throttling (délai paramétrable entre les sons)

### **Optimisations clés** :
- Utilisation de `queue.Queue` pour le buffering inter-threads
- Prétraitement vidéo en basse résolution (320x240) pour fluidité
- Calcul de distance Euclidienne optimisé avec NumPy

---

## 🛠️ **Technologies**  
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FF6F00?logo=mediapipe)](https://mediapipe.dev/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)](https://opencv.org/)  
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?logo=raspberrypi)](https://www.raspberrypi.org/)  
[![Pygame](https://img.shields.io/badge/Pygame-000000?logo=pygame)](https://www.pygame.org/)  
[![Tkinter](https://img.shields.io/badge/Tkinter-%23075BAB?logo=tkinter)](https://docs.python.org/fr/3.13/library/tkinter.html/)

---

## 🚀 **Installation**  
```bash
# Cloner le dépôt
git clone https://github.com/ouriahi/MaestroSync.git

# Installer les dépendances
pip install -r requirements.txt
```