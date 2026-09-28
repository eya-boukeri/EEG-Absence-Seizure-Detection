# 🧠 Détection Automatique des Crises d'Absence à partir de Signaux EEG

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Projet de **Data Mining appliqué aux données biomédicales** visant à concevoir une chaîne complète de traitement du signal et d'apprentissage automatique pour détecter automatiquement les épisodes de **crises d'absence épileptiques** à partir d'enregistrements électroencéphalographiques (EEG) multi-canaux.

---

## 📌 Sommaire

- [Contexte Biomédical](#-contexte-biomédical)
- [Architecture & Pipeline du Projet](#-architecture--pipeline-du-projet)
- [Description des Données](#-description-des-données)
- [Structure du Dépôt](#-structure-du-dépôt)
- [Méthodologie & Étapes](#-méthodologie--étapes)
  - [1. Restructuration & Prétraitement](#1-restructuration--prétraitement)
  - [2. Alignement des Annotations & Variable Cible](#2-alignement-des-annotations--variable-cible)
  - [3. Segmentation en Fenêtres Glissantes](#3-segmentation-en-fenêtres-glissantes)
  - [4. Extraction de Caractéristiques](#4-extraction-de-caractéristiques)
  - [5. Classification & Évaluation](#5-classification--évaluation)
  - [6. Reconstruction Temporelle](#6-reconstruction-temporelle)
- [Installation & Utilisation](#-installation--utilisation)
- [Publication sur GitHub](#-procédure-pour-publier-sur-github)
- [Auteurs](#-auteurs)

---

## 🔬 Contexte Biomédical

L'**électroencéphalogramme (EEG)** est l'examen de référence pour enregistrer l'activité électrique cérébrale via des électrodes disposées sur le cuir chevelu. 

Les **crises d'absence** (forme fréquente d'épilepsie généralisée, notamment chez l'enfant et l'adolescent) se manifestent sur le tracé EEG par des **décharges paroxystiques bilatérales, synchrones et symétriques de pointes-ondes à 3 Hz**.

### Problématique
La relecture manuelle d'enregistrements EEG de plusieurs heures ou jours par les neurologues est :
- Extrêmement chronophage,
- Sujette à la fatigue visuelle et à la variabilité inter-annotateurs.

Ce projet met en place un modèle de **Machine Learning supervisé** capable de segmenter automatiquement le signal, d'extraire des descripteurs pertinents et d'identifier avec précision les intervalles de crise.

---

## ⚙️ Architecture & Pipeline du Projet

Le workflow suit la méthodologie standard du Data Mining pour les séries temporelles biomédicales :

```
                        ┌────────────────────────────────┐
                        │        Signaux EEG Bruts       │
                        │    (23 canaux @ 256 Hz, CSV)   │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │ Exploration & Restructuration  │
                        │    (Format Large ➔ Long)       │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │ Exploitation des Annotations   │
                        │   (Bornes a/b, Onset, Excel)   │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │   Construction Cible `Class`   │
                        │     (0 = Normal, 1 = Crise)    │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │   Segmentation Temporelle      │
                        │   (Fenêtres 2s, Recouvr. 50%)  │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │   Extraction Caractéristiques  │
                        │   - Temporelles (RMS, Energie) │
                        │   - Spectrales FFT (δ, θ, α, β)│
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │    Modélisation ML & Test      │
                        │   (DT, Random Forest, KNN, SVM)│
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │ Évaluation & Reconstruction    │
                        │ (F1-score, Recall, Début/Fin)  │
                        └────────────────────────────────┘
```

---

## 📂 Description des Données

Chaque patient dispose d'enregistrements exportés depuis MATLAB vers CSV :
- **Fichier Signal (`<id>_0000d.csv`)** : Format large (1 ligne = 1 seconde à 256 Hz, 23 canaux : 19 canaux EEG standard selon le système 10-20 + signaux polygraphiques EMG1, EMG2, SLI, ECG).
- **Fichier Annotations (`<id>_0000h.csv`)** : Événements horodatés (`Onset`, `Annotations`, `Duration`). Les crises y sont délimitées soit par les marqueurs cliniques `a` (début) $\rightarrow$ `crise` $\rightarrow$ `b` (fin), soit par des marqueurs ponctuels.
- **Référentiel global (`Crises_absences.xlsx`)** : Répertoire clinique listant le nombre de crises (`NombreCE`) et leurs durées exactes pour valider les détections.

> ⚠️ **Note de stockage :** Les fichiers CSV de signaux bruts représentent plusieurs centaines de mégaoctets chacun (dépassant la limite GitHub de 100 Mo). Ils sont donc exclus du suivi Git via le fichier `.gitignore`.

---

## 📁 Structure du Dépôt

```plaintext
data_mining/
├── data/
│   ├── .gitkeep
│   ├── Crises_absences.xlsx       # Base de référence des crises
│   ├── Annotations.xlsx           # Annotations cliniques
│   └── (Fichiers CSV volumineux exclus du repo via .gitignore)
├── output/                        # Résultats, visualisations et modèles exportés
│   └── .gitkeep
├── mini-projet-data-mining.pdf    # Cahier des charges et sujet détaillé
├── mini_projet_eeg.ipynb          # Notebook Jupyter principal (Code, pipeline, visualisations)
├── requirements.txt               # Dépendances Python nécessaires
├── .gitignore                     # Exclusion de l'environnement virtuel et gros volumes
└── README.md                      # Documentation du projet
```

---

## 🧪 Méthodologie & Étapes

### 1. Restructuration & Prétraitement
Conversion des signaux EEG bruts du format *large* (1 ligne = 1 seconde avec 256 colonnes par canal) au format *long* classique (`Time, EEG canal 1, ..., EEG canal N`) avec horodatage continu en secondes.

### 2. Alignement des Annotations & Variable Cible
- Nettoyage des encodages et appariement glouton des balises de début (`a`) et fin (`b`).
- Complétion des crises isolées grâce aux durées exactes recensées dans le fichier Excel.
- Génération d'une étiquette binaire par échantillon : `0` (hors crise) / `1` (pendant la crise).

### 3. Segmentation en Fenêtres Glissantes
- Fenêtres temporelles de **$2.0\text{ s}$** ($512\text{ échantillons}$ à $256\text{ Hz}$).
- Recouvrement de **$50\%$** ($1.0\text{ s}$ de décalage).
- Attribution de la classe par fenêtre : classe 1 si au moins **$50\%$** des échantillons internes sont en crise.

### 4. Extraction de Caractéristiques
Pour chaque fenêtre et chaque canal d'intérêt :
- **Domaine temporel :** Moyenne, Écart-type, Variance, Minimum, Maximum, Amplitude crête-à-crête, Valeur efficace (RMS), Énergie du signal.
- **Domaine fréquentiel (Transformée de Fourier Rapide - FFT) :** Puissance spectrale dans les sous-bandes physiologiques :
  - **Delta ($\delta$) :** 0.5 – 4 Hz *(ralentissements pathologiques)*
  - **Theta ($\theta$) :** 4 – 8 Hz *(activité épileptogène clé)*
  - **Alpha ($\alpha$) :** 8 – 13 Hz *(rythme de veille/repos)*
  - **Beta ($\beta$) :** 13 – 30 Hz *(activité rapide)*

### 5. Classification & Évaluation
- Découpage stratifié des données : **$80\%$ entraînement** / **$20\%$ test**.
- Entraînement et comparaison de plusieurs algorithmes :
  - **Decision Tree (Arbre de Décision)**
  - **Random Forest (Forêt Aléatoire)**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machine (SVM)**
- **Analyse critique du déséquilibre de classe :** Les crises ne représentent généralement que 1 à 2 % de la durée totale de l'enregistrement. L'*Accuracy* seule est trompeuse. L'évaluation privilégie donc :
  - **Recall (Sensibilité)** : Priorité médicale absolue (minimiser les faux négatifs = ne rater aucune crise).
  - **Precision & F1-Score** : Mesure de compromis pour éviter les fausses alertes.
  - **Matrice de confusion**.

### 6. Reconstruction Temporelle
Fusion des fenêtres consécutives prédites positives pour reconstruire les intervalles temporels de crise `[Début (s), Fin (s)]` et comparaison directe avec les annotations médicales de référence.

---

## 💻 Installation & Utilisation

### Prérequis
- Python 3.10 ou supérieur
- Git

### 1. Cloner le projet
```bash
git clone https://github.com/eya-boukeri/<nom-du-repo>.git
cd <nom-du-repo>
```

### 2. Créer et activer un environnement virtuel
```bash
# Sous Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Sous Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancer le Notebook Jupyter
```bash
jupyter notebook mini_projet_eeg.ipynb
```

---

## 🚀 Procédure pour Publier sur GitHub

Si vous n'avez pas encore créé le repository sur GitHub, voici la démarche pas-à-pas :

### Étape 1 : Créer le repository sur GitHub
1. Rendez-vous sur votre compte GitHub : **[https://github.com/new](https://github.com/new)**
2. Nommez le repository (par exemple : `eeg-absence-seizure-detection` ou `data-mining-eeg`).
3. Choisissez la visibilité (**Public** ou **Private**).
4. **Important :** Ne cochez *PAS* "Add a README file", ni ".gitignore", ni "license" (car nous les avons déjà créés localement).
5. Cliquez sur **Create repository**.

### Étape 2 : Initialiser Git et pusher depuis votre terminal
Ouvrez un terminal (PowerShell ou Bash) dans le dossier du projet `d:\data_mining\` et exécutez :

```bash
# 1. Initialiser le dépôt Git local
git init

# 2. Configurer la branche principale par défaut
git branch -M main

# 3. Ajouter tous les fichiers (le .gitignore protège les gros fichiers et .venv)
git add .

# 4. Créer le premier commit
git commit -m "feat: initial commit with EEG processing pipeline and documentation"

# 5. Relier à votre dépôt GitHub distant (remplacez <nom-du-repo> par le nom choisi)
git remote add origin https://github.com/eya-boukeri/<nom-du-repo>.git

# 6. Envoyer le code sur GitHub
git push -u origin main
```

---

## 👥 Auteurs

- **Eya Boukeri** ([@eya-boukeri](https://github.com/eya-boukeri))
- Travail réalisé dans le cadre du module **Data Mining appliqué aux données biomédicales**.
