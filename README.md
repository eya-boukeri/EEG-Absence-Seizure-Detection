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
  - [8. Validation Clinique Inter-Patients](#-8-validation-clinique-inter-patients-patient-independent-evaluation)
  - [9. Détection à 3 Classes & Horizon d'Anticipation](#-9-détection-à-3-classes--phase-dalerte-précoce-avant-crise-pre-ictal)
  - [10. Deep Learning End-to-End : Modèle Convolutif EEGNet](#-10-deep-learning-end-to-end--modèle-convolutif-spatiotemporel-eegnet)
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

| Étape du Pipeline | Nombre de Features | Accuracy Brute | Balanced Accuracy | Precision | Recall | F1-score | Faux Positifs (FP) | Faux Négatifs (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Mono-Canal Initial (`EEGFp1`)** | 12 | 95.99 % | 86.42 % | 55.00 % | 75.86 % | 63.77 % | 18 | 7 |
| **2. Multi-Canaux Brut (21 canaux)** | 252 | 98.24 % | 90.72 % | 80.00 % | 82.76 % | 81.36 % | 6 | 5 |
| **3. Multi-Canaux + Ratios + SMOTE** | 294 | 98.40 % | 90.80 % | 82.76 % | 82.76 % | 82.76 % | 5 | 5 |
| **4. Système Optimisé Final (+ Post-Filtrage)** | **294** | **99.04 %** | **92.83 %** | **92.59 %** | **86.21 %** | **89.29 %** | **2** | **4** |

> 🎯 **Bilan Intra-Patient :** Le passage du mono-canal au pipeline complet optimisé permet de faire passer le F1-score de **63.77%** à **89.29%**, la **Balanced Accuracy** à **92.83%**, tout en réduisant les fausses alarmes de **18 à seulement 2**.

#### 👥 8. Validation Clinique Inter-Patients (*Patient-Independent Evaluation*)
Afin d'évaluer la robustesse du système face à des cerveaux jamais vus en conditions hospitalières réelles, la cohorte des 21 patients à montage complet a été scindée :
- **Groupe Entraînement (16 patients, 63 crises)** : Standardisation Z-Score par patient + Suréchantillonnage SMOTE.
- **Groupe Test Indépendant (5 patients, 16 crises)** : Zéro fuite de données (*Zero Data Leakage*).
- **Résultats obtenus sur les patients inconnus :**
  - **Balanced Accuracy Globale :** **88.16 %** (anti-biais de classe)
  - **Sensibilité / Rappel :** **79.21 %** (~8 crises sur 10 interceptées)
  - **Exactitude Brute (Accuracy) :** **96.53 %**
  - **Taux de faux positifs :** < **2.9 %**

#### ⚡ 9. Détection à 3 Classes : Phase d'Alerte Précoce « Avant Crise » (*Pre-Ictal*)
Pour transformer le système d'un simple détecteur de crise en un **dispositif d'alerte clinique préventive**, une troisième classe biomédicale commutable a été introduite :

- **Classe 0 (Inter-ictal / Normal) :** Activité de repos ou physiologique de fond.
- **Classe 2 (Pré-ictal / Avant crise) :** Fenêtre d'anticipation de **10 secondes immédiatement avant l'apparition clinique de la crise** ($[T_{\text{début}} - 10\text{s}, T_{\text{début}}[$).
- **Classe 1 (Ictal / Crise) :** Décharge paroxystique avérée de pointes-ondes à 3 Hz ($[T_{\text{début}}, T_{\text{fin}}]$).

```
          Inter-ictal (0)             Pré-ictal (2)            Ictal (1)
     ─── Sain / Repos de Fond ───┬── 10s Avant Crise ──┬── Crise Confirmée ──
                                 │  [Tdebut - 10s]     │  [Tdebut]
```

**Performances Comparatives selon l'Horizon d'Anticipation (Intra-Patient) :**

| Horizon d'Anticipation | Durée | Sensibilité Crise (1) | Sensibilité Avant Crise (2) | Précision Avant Crise | Spécificité Sain (0) | Balanced Accuracy | F1 Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alerte Réflexe** | **10 secondes** | 89.66 % | 63.64 % | **87.50 %** | **99.30 %** | 84.20 % | 87.87 % |
| **Équilibre Optimal** ⭐ | **30 secondes (0.5 min)** | **96.55 %** | **90.91 %** | 77.92 % | 96.60 % | **94.69 %** | **91.20 %** |
| **Anticipation Longue** | **60 secondes (1 min)** | 86.21 % | **96.03 %** | 76.10 % | 92.32 % | 91.52 % | 88.94 % |

> 💡 **Analyse Clinique des Intervalles :** L'analyse temporelle des crises d'absence révèle un intervalle médian de **124 secondes (~2 min)** entre les crises successives (avec un minimum de **29 secondes** entre les crises 6 et 7). Une fenêtre de **30 secondes (0.5 min)** offre l'équilibre idéal en évitant les collisions entre crises rapprochées tout en permettant au patient de se mettre en sécurité avec une **Balanced Accuracy de 94.69 %**. Pour un préavis étendu de **1 minute**, le modèle anticipe **96.03 %** des phases pré-ictiques.

#### 🧠 10. Deep Learning End-to-End : Modèle Convolutif Spatiotemporel EEGNet

Pour s'affranchir de l'ingénierie manuelle de caractéristiques et évaluer la capacité d'un réseau de neurones profond à apprendre les motifs épileptiques directement sur le signal brut, le modèle **EEGNet** ([Lawhern et al., 2018](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)) a été implémenté en **PyTorch** et intégré nativement dans le moteur de calcul et le tableau de bord web.

##### 🏗️ Architecture d'EEGNet (2 211 paramètres)
EEGNet traite directement des tenseurs bruts 2D de dimension $(Batch, 1, C=21\text{ canaux}, T=512\text{ échantillons})$ :
1. **Convolutions Temporelles 2D** $(1 \times 64)$ : Apprennent des filtres fréquentiels passe-bande adaptatifs (capturant les pointes-ondes Delta 3 Hz et les rythmes Alpha/Theta).
2. **Convolutions Spatiales Depthwise** $(21 \times 1)$ : Apprennent des combinaisons optimales de la topologie du scalp sans sur-apprentissage ($D=2$ filtres spatiaux par filtre temporel).
3. **Convolutions Séparables Temporelles** $(1 \times 16)$ : Découplent les convolutions spatiales et temporelles pour synthétiser les dynamiques globales du signal.
4. **Classification Linéaire** : Projection finale vers les 3 classes médicales (*Normal*, *Crise*, *Avant crise*).

##### ⚖️ Comparaison : Random Forest (Features Expertes) vs EEGNet (Deep Learning Brut)

| Propriété / Métrique | Random Forest (294 Features + SDR) | EEGNet (Deep Learning PyTorch) |
| :--- | :---: | :---: |
| **Type d'Entrée** | 294 variables calculées (RMS, FFT, SDR) | Tenseur EEG brut $(21 \times 512 = 10\,752\text{ valeurs})$ |
| **Ingénierie Manuelle** | Oui (Domain Expertise biomédicale) | **Aucune** (Apprentissage End-to-End) |
| **Nombre de Paramètres** | 100 arbres de décision pondérés | **2 211 poids** (Ultra-compact, anti-overfitting) |
| **Balanced Accuracy (3 Classes, 30s)** | **94.69 %** | **75.64 %** |
| **Sensibilité Crise (1)** | **96.55 %** | **75.64 %** |
| **Sensibilité Avant Crise (2)** | **90.91 %** | **69.70 %** |
| **Spécificité Tracé Sain (0)** | **96.60 %** | **68.20 %** |
| **Explicabilité Clinique (XAI)** | Importance de Gini par variable | **Attention spatiale par électrode** (Filtres Depthwise) |

> 🔬 **Enseignement Scientifique :**
> - Sur des effectifs de crises modérés, le modèle **Random Forest avec ratios spectraux lents/rapides $\text{SDR} = \frac{\delta + \theta}{\alpha + \beta}$** surpasse le Deep Learning brut (**94.7% vs 75.6%** de Balanced Accuracy) car la connaissance physique de l'onde à 3 Hz guide directement la décision.
> - **EEGNet** réussit néanmoins la prouesse de détecter de façon entièrement autonome **75.6% des crises et 69.7% des alertes pré-ictales** sans extraire aucune caractéristique manuelle, ce qui en fait le candidat privilégié pour les architectures matérielles embarquées (*Edge AI / neuromorphic chips*).

##### 🔬 Étude d'Ablation Méthodologique (Recherche Approfondie)

Une étude d'ablation systématique a été menée sur signal brut $(21 \times 512)$ en mode 3 classes (Horizon 30s) pour quantifier l'impact de chaque amélioration :

| Module / Architecture | Paramètres | Balanced Acc | F1-Macro | Rappel Crise (C1) | Rappel Avant-Crise (C2) | Spécificité Sain (C0) | Faux Positifs (FP) | Apport Scientifique |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M0 : Baseline EEGNet** | 2 211 | 61.70 % | 40.91 % | 79.31 % | 65.15 % | 40.64 % | 314 FP | Modèle de référence compact (Lawhern et al., 2018) |
| **M1 : EEGNet + Focal Loss ($\gamma=2.0$)** | 2 211 | 56.10 % | 34.00 % | 65.52 % | **74.24 %** | 28.54 % | 378 FP | **+9.1% de rappel avant-crise** (pénalise les erreurs sur les exemples rares) |
| **M2 : EEGNet + Data Augmentation** | 2 211 | 61.34 % | **44.43 %** | 68.97 % | 62.12 % | **52.93 %** | **249 FP** | **-65 faux positifs (-21%)** grâce au Jittering gaussien & Scaling |
| **M3 : Inception-EEGNet (Multi-Échelles)** | 3 571 | 55.82 % | 42.51 % | 62.07 % | 56.06 % | 49.34 % | 268 FP | Découplage temporel Pointe 60ms vs Onde lente 3 Hz (500ms) |
| **M4 : EEGNet + Bi-LSTM (Séquentiel)** | 14 435 | 65.50 % | 40.97 % | **82.76 %** | 72.73 % | 41.02 % | 312 FP | Modélisation récurrente bidirectionnelle de la chronologie |
| **M5 : Modèle Intégré (Inception+LSTM+Focal+Aug)** 🚀 | **17 459** | **66.07 %** | **41.69 %** | **82.76 %** | **74.24 %** | 41.21 % | 311 FP | **Meilleure performance globale : synergie temporelle et fréquentielle** |

#### 🔥 11. Modèle XGBoost (Extreme Gradient Boosting) & Comparaison Multi-Paradigmes

Pour pousser la recherche comparative encore plus loin, l'algorithme de pointe **XGBoost (Extreme Gradient Boosting)** ([Chen & Guestrin, 2016](https://dl.acm.org/doi/10.1145/2939672.2939785)) a été intégré au système et benchmarqué face à **Random Forest** (Bagging), **SVM (RBF)** et **EEGNet** (Deep Learning).

##### 🧠 Principes Algorithmiques & Spécificités EEG
Contrairement à Random Forest qui entraîne des arbres indépendants en parallèle par échantillonnage bootstrap (Bagging), **XGBoost** construit des arbres de manière séquentielle et additive (Boosting) :
1. **Optimisation par Gradient du Second Ordre** : XGBoost utilise le développement de Taylor au 2nd ordre de la fonction de perte ($\mathcal{L}^{(t)} \simeq \sum [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)]$), où $g_i$ est le gradient et $h_i$ le hessien.
2. **Régularisation Structurelle Forte ($\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$)** : Elle pénalise la complexité des arbres ($T$ feuilles) et le poids des feuilles ($w_j$), évitant le surapprentissage sur les 294 descripteurs multi-électrodes.
3. **Sous-échantillonnage de Colonnes (`colsample_bytree=0.8`)** : Réduit la corrélation entre canaux EEG voisins et favorise la découverte de descripteurs rares.

##### ⚖️ Tableau Comparatif Multi-Modèles : Détection Standard (2 Classes : Sain vs Crise)

| Modèle ML / DL | Type d'Architecture | Accuracy | Balanced Acc | Rappel Crise (Sensibilité) | Précision Crise | F1-Score | Faux Positifs (FP) | Vitesse / Inférence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** 🏆 | **Gradient Boosted Trees (100 arbres, $\eta=0.1$)** | **99.20 %** | **94.66 %** | **89.66 %** | **92.86 %** | **91.23 %** | **2 FP** | **~13 ms / tracé (Ultra-rapide)** |
| **Random Forest** | Bagged Decision Trees (100 arbres, pondéré) | 98.88 % | 91.21 % | 82.76 % | 92.31 % | 87.27 % | 2 FP | ~45 ms / tracé |
| **Decision Tree** | Arbre CART unique | 97.44 % | 85.50 % | 72.41 % | 70.00 % | 71.19 % | 9 FP | ~2 ms / tracé |
| **SVM (Noyau RBF)** | Séparateur à marge maximale ($C=10.0$) | 96.96 % | 83.20 % | 68.97 % | 64.52 % | 66.67 % | 11 FP | ~35 ms / tracé |
| **EEGNet Baseline** | Réseau Convolutif Spatiotemporel (End-to-End) | 94.70 % | 75.64 % | 75.64 % | 31.00 % | 55.05 % | 48 FP | ~5 ms (GPU) / ~25 ms (CPU) |

> ⭐ **Gain Majeur avec XGBoost en Détection 2-Classes :**
> - **+3.45 % de Balanced Accuracy** par rapport à Random Forest (94.66 % vs 91.21 %).
> - **+6.90 % de Rappel des Crises** (89.66 % vs 82.76 %) avec **strictement le même niveau d'éradication des faux positifs (seulement 2 FP)** sur 52 minutes de tracé continu !
> - **F1-Score record de 91.23 %**.

##### ⚖️ Tableau Comparatif : Alerte Précoce 3 Classes (Sain vs Crise vs Avant-Crise 30s)

| Modèle | Balanced Acc | Rappel Crise (1) | Rappel Avant-Crise (2) | Spécificité Sain (0) | F1-Macro | Faux Positifs (FP) | Temps Entraînement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **98.17 %** | **100.00 %** | **96.97 %** | **97.54 %** | **95.34 %** | **13 FP** | 44.9 s |
| **XGBoost** 🔥 | **95.65 %** | **100.00 %** | **93.94 %** | **93.01 %** | **89.21 %** | 23-37 FP | **13.2 s (3.4× plus rapide)** |
| **SVM (RBF)** | 91.79 % | 96.55 % | 93.94 % | 84.88 % | 81.25 % | 80 FP | 10.4 s |
| **Modèle Intégré DL (M5)** | 66.07 % | 82.76 % | 74.24 % | 41.21 % | 41.69 % | 311 FP | ~180 s (PyTorch CPU) |

##### 🔍 Analyse Clinique de l'Importance des Features selon XGBoost vs Random Forest :
- **XGBoost** concentre une attention prononcée sur les électrodes frontales :
  - `Beta_EEGF3` : **12.08 %** de gain relatif (reflète l'interruption brutale de l'activité rapide frontale lors de la généralisation de la crise).
  - `SDR_EEGF8` : **3.31 %** (ratio spectral lentes/rapides en zone fronto-temporale).
  - `Beta_EEGF4` : **3.05 %** (confirmation bilatérale symétrique du lobe frontal).
- **Random Forest** distribue l'importance de manière plus diffuse sur l'Alpha occipital (`Alpha_EEGO2` 2.38%) et l'énergie frontale (`Energy_EEGF3` 1.94%).
- Cette sélectivité chirurgicale de XGBoost sur les dérivations frontales corrobore précisément la physiopathologie des absences épileptiques (décharges synchrones bilatérales à prédominance frontale).

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

```powershell
# Option 1 (Le plus simple sous Windows) :
.\run_dashboard.ps1
# ou double-cliquez sur run_dashboard.bat

# Option 2 (En activant l'environnement virtuel préalablement) :
.\.venv\Scripts\Activate.ps1
python web_dashboard/server.py

# Option 3 (En appelant directement l'exécutable de l'environnement virtuel) :
.\.venv\Scripts\python.exe web_dashboard/server.py
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
