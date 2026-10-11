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
  - [7. Améliorations Appliquées du Pipeline](#7-améliorations-appliquées-du-pipeline)
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
- **Fichier Annotations Événements (`<id>_0000h.csv`)** : Événements horodatés en secondes depuis le début de l'enregistrement (`Onset`, `Annotations`, `Duration`).
- **Annotations Référentielles Médicales (`Annotations MAJ.xlsx` / `annotationsMAJ.xls`)** :
  - **Nouvelle colonne `DebutTrace`** : Indique l'horaire précis de début d'enregistrement du tracé EEG.
  - **Valeurs de début (`DebCE`) et fin (`FinCE`) traitées comme des horaires** :
    - **Calcul direct de la durée réelle de chaque crise** :
      $$\text{Durée de la crise } k = \text{Horaire FinCE}_k - \text{Horaire DebCE}_k$$
      *Exemple :* Patient `190304A-E`, début à `10:35:06` et fin à `10:35:13` $\rightarrow$ Durée = $7\text{ secondes}$.
    - **Synchronisation temporelle sur l'axe EEG** ($t = 0$ au début de l'enregistrement) :
      $$t_{\text{EEG}} = (\text{Horaire} - \text{DebutTrace}) \times 86400$$

> ⚠️ **Note de stockage :** Les fichiers CSV de signaux bruts représentent plusieurs centaines de mégaoctets chacun (dépassant la limite GitHub de 100 Mo). Ils sont donc exclus du suivi Git via le fichier `.gitignore`.

---

## 📁 Structure du Dépôt

```plaintext
data_mining/
├── data/
│   ├── .gitkeep
│   ├── Annotations MAJ.xlsx       # Référentiel médical avec DebutTrace & horaires réels
│   ├── annotationsMAJ.xls         # Alias de compatibilité
│   ├── Crises_absences.xlsx       # Base de référence initiale
│   └── (Fichiers CSV volumineux exclus du repo via .gitignore)
├── engine.py                      # Moteur biomédical, extraction multi-canaux & ML
├── web_dashboard/                 # Application Web Interactive & Dashboard de Détection
│   ├── index.html                 # Interface utilisateur biomédicale moderne
│   ├── style.css                  # Design moderne, dark mode & glassmorphism
│   ├── app.js                     # Visualisation dynamique Canvas & communication API
│   └── server.py                  # Serveur HTTP local autonome (Port 5050)
├── output/                        # Résultats, visualisations et modèles exportés
│   └── .gitkeep
├── mini-projet-data-mining.pdf    # Cahier des charges et sujet détaillé
├── mini_projet_eeg.ipynb          # Notebook Jupyter principal (Code, pipeline, widget ipywidgets)
├── requirements.txt               # Dépendances Python nécessaires
├── .gitignore                     # Exclusion de l'environnement virtuel et gros volumes
└── README.md                      # Documentation du projet
```

---

## 🧪 Méthodologie & Étapes

### 1. Restructuration & Prétraitement
Conversion des signaux EEG bruts du format *large* (1 ligne = 1 seconde avec 256 colonnes par canal) au format *long* classique (`Time, EEG canal 1, ..., EEG canal N`) avec horodatage continu en secondes.

### 2. Exploitation des Horaires Réels & Synchronisation du Tracé
- Utilisation de la colonne **`DebutTrace`** pour positionner avec rigueur les crises médicales sur la durée d'enregistrement.
- Détermination exacte de la durée de chaque crise par simple différence des horaires de fin et de début.
- Génération de la variable cible binaire par échantillon : `0` (hors crise) / `1` (pendant la crise sur l'ensemble des crises réelles).

### 3. Segmentation en Fenêtres Glissantes
- Fenêtres temporelles de **$2.0\text{ s}$** ($512\text{ échantillons}$ à $256\text{ Hz}$).
- Recouvrement de **$50\%$** ($1.0\text{ s}$ de décalage).
- Attribution de la classe par fenêtre : classe 1 si au moins **$50\%$** des échantillons internes sont en crise.

### 4. Extraction Multi-Canaux des Caractéristiques
Pour exploiter la synchronisation spatiale globale des décharges épileptiques, les descripteurs sont extraits **simultanément sur l'ensemble des 21 canaux** (19 canaux EEG standard 10-20 + ECG + SLI) :
- **12 descripteurs par canal ($21 \times 12 = 252\text{ caractéristiques}$) :**
  - **Domaine temporel (8) :** Moyenne, Écart-type, Variance, Minimum, Maximum, Amplitude crête-à-crête, RMS, Énergie.
  - **Domaine fréquentiel FFT (4) :**
    - **Delta ($\delta$, 0.5 – 4 Hz) :** Activité lente paroxystique (onde lente à 3 Hz).
    - **Theta ($\theta$, 4 – 8 Hz) :** Activité transitoire épileptogène.
    - **Alpha ($\alpha$, 8 – 13 Hz) :** Rythme de fond et désynchronisation occipito-pariétale.
    - **Beta ($\beta$, 13 – 30 Hz) :** Activité cérébrale rapide.

### 5. Classification Multi-Canaux & Évaluation
- Découpage stratifié des données : **$80\%$ entraînement** / **$20\%$ test**.
- Entraînement des algorithmes sur la totalité des **252 caractéristiques spatiotemporelles** :
  - **Random Forest** : **Accuracy = $98.24\%$**, **Recall = $79.31\%$**, **Precision = $82.14\%$**, **F1 = $80.70\%$**
  - **KNN (Distance pondérée)** : **Accuracy = $97.92\%$**, **Recall = $72.41\%$**, **Precision = $80.77\%$**, **F1 = $76.36\%$**
  - **SVM (Noyau RBF)** : **Accuracy = $93.27\%$**, **Recall = $89.66\%$** *(sensibilité médicale maximale)*
  - **Decision Tree** : **Accuracy = $97.12\%$**, **Recall = $55.17\%$**, **Precision = $76.19\%$**
- **Gain de l'approche multi-canaux :**
  - La précision bondit de $55\%$ à **$82.14\%$** par rapport au traitement mono-électrode.
  - Les artéfacts locaux sont filtrés grâce à la corrélation inter-canaux (frontaux, centraux et occipitaux).

### 6. Reconstruction Temporelle
Fusion des fenêtres consécutives prédites positives pour reconstruire les intervalles temporels de crise `[Début (s), Fin (s)]` et comparaison directe avec les annotations médicales de référence.

### 7. Améliorations Appliquées du Pipeline (Niveau Clinique)

Pour surpasser les limites des classifieurs bruts et éliminer les fausses alertes, trois améliorations concrètes ont été implémentées et validées :

1. **Ratios Spectraux Neurophysiologiques (294 descripteurs) :**
   - Calcul pour chaque canal de l'indice de décharge lente vs rapide :
     $$\text{SDR} = \frac{\text{Delta} + \text{Theta}}{\text{Alpha} + \text{Beta} + 10^{-6}}$$
   - Ratio de dominance de l'onde lente épileptique ($\approx 3\text{ Hz}$) :
     $$\text{RelDelta} = \frac{\text{Delta}}{\sum \text{Puissances} + 10^{-6}}$$
2. **Rééquilibrage par SMOTE (Synthetic Minority Over-sampling) :**
   - Suréchantillonnage contrôlé de la classe minoritaire (crises) sur l'ensemble d'entraînement, fournissant aux algorithmes des frontières de décision plus nettes.
3. **Post-filtrage Temporel par Persistance Clinique :**
   - Élimination des alertes positives isolées durant moins de 2 secondes (artéfacts transitoires de déglutition, clignement ou mouvement).
   - Comblement par hystérésis des micro-décrochages intra-crise (`1 0 1` $\rightarrow$ `1 1 1`).

#### 📊 Synthèse Comparative de la Progression

| Étape du Pipeline | Nombre de Features | Accuracy | Precision | Recall | F1-score | Faux Positifs (FP) | Faux Négatifs (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Mono-Canal Initial (`EEGFp1`)** | 12 | 95.99 % | 55.00 % | 75.86 % | 63.77 % | 18 | 7 |
| **2. Multi-Canaux Brut (21 canaux)** | 252 | 98.24 % | 80.00 % | 82.76 % | 81.36 % | 6 | 5 |
| **3. Multi-Canaux + Ratios + SMOTE** | 294 | 98.40 % | 82.76 % | 82.76 % | 82.76 % | 5 | 5 |
| **4. Système Optimisé Final (+ Post-Filtrage)** | **294** | **99.04 %** | **92.59 %** | **86.21 %** | **89.29 %** | **2** | **4** |

> 🎯 **Bilan :** Le passage du mono-canal au pipeline complet optimisé permet de faire passer le F1-score de **63.77%** à **89.29%**, tout en réduisant les fausses alarmes de **18 à seulement 2**.

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
> Le notebook intègre désormais un **Widget interactif (Partie XIII)** permettant de choisir n'importe quel canal parmi les 21 électrodes et d'entraîner en direct le modèle ML.

### 5. Lancer l'Application Web Dashboard (Interface Dédiée)
```bash
python web_dashboard/server.py
```
Ouvrez ensuite votre navigateur sur : **[http://localhost:5050](http://localhost:5050)** pour accéder à l'interface moderne avec visualiseur de signal haute résolution, zoom interactif sur les crises, matrice de confusion et métriques en temps réel.

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
