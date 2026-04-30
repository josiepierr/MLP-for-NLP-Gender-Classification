# Analyse computationnelle des professions de foi - Législatives 1993


Ce projet applique des méthodes de traitement automatique du langage naturel (NLP) au corpus Archelec des élections législatives françaises de 1993. L'objectif est double : identifier les structures thématiques qui structurent le discours partisan, et examiner si le sexe des candidats laisse une empreinte stylistique mesurable dans leurs professions de foi.

---

## Contexte

Les élections législatives de mars 1993 se tiennent dans un contexte de cohabitation : la gauche au pouvoir depuis 1988 fait face à une vague de droite qui lui coûtera 213 sièges. Le corpus Archelec, constitué par Sciences Po / CEVIPOF, rassemble les professions de foi numérisées de cette élection - l'un des rares moments où chaque candidat, qu'il soit issu d'un grand parti ou d'une formation marginale, dispose du même espace pour s'adresser à l'ensemble des électeurs de sa circonscription.

5 835 documents, 43 colonnes de métadonnées, une fenêtre ouverte sur la diversité idéologique d'une époque.

---

## Structure du projet

```
MLP-for-NLP-Topic-Modeling/
│
├── data/
│   ├── legislatives_1993.zip          # Textes bruts (OCR)
│   ├── meta_archelect_1993.csv        # Métadonnées (parti, sexe, âge, profession…)
│   └── processed/                     # Sorties des notebooks
│       ├── corpus_1993.csv            # Corpus nettoyé + features de base
│       ├── topics_1993.csv            # Proportions LDA par document
│       ├── corpus_1993_with_lda_topics_and_descriptors.csv
│       └── style_features_1993.csv   # Features stylistiques
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Chargement, nettoyage, description
│   ├── 02_topic_modeling.ipynb        # LDA, NMF, BERTopic
│   ├── 03_lda_topic_join_and_analysis.ipynb  # Croisements thèmes × partis × genre
│   └── 04_style_gender.ipynb          # Stylométrie et signal de genre
│
├── src/
│   └── function.py                    # Fonctions partagées (nettoyage, MSTTR…)
│
├── figures/                           # Graphiques produits par les notebooks
|
└── requirements.txt
```

---

## Notebooks

### 01 - Préparation des données
Nous faisons les opérations suivantes : chargement des textes bruts depuis le ZIP, correction des artefacts OCR, normalisation Unicode NFC, tokenisation et lemmatisation avec spaCy (`fr_core_news_md`). Nous construisons ensuite trois représentations textuelles selon l'usage downstream : texte lemmatisé pour la LDA, texte normalisé pour les embeddings, texte brut nettoyé pour la stylométrie. Enfin, nous calculons le MSTTR (richesse lexicale) sur fenêtre de 100 tokens. Le corpus final comprend 5 835 documents avec une moyenne de 310 tokens par document.

### 02 - Modélisation thématique
Nous comparons trois approches : **LDA** (Gensim), **NMF**, **BERTopic**.

### 03 - Analyse croisée thèmes × partis × genre
Ici, nous faisons la jointure des thèmes LDA avec la métadonnée afin d'effectuer des analyses complémentaires : croisements entre les autres variables et les thèmes obtenus.

### 04 - Stylométrie et signal de genre
**Notebook entièrement indépendant**. Opérations principales effectuées : extraction de 20+ features stylistiques issues de la littérature (Argamon & Koppel 2003, Pennebaker 2001) - taux de déterminants, nominalisations, bigrammes POS, hedging, profondeur syntaxique, vocabulaire social/économique, registre vous/tu, etc -, tests de Mann-Whitney avec correction de Bonferroni, régressions OLS avec contrôle du parti, et classification supervisée par régression logistique.

---

## Installation

```bash
pip install -r requirements.txt
```


---

## Ordre d'exécution

Les notebooks doivent être lancés dans l'ordre numérique. Le notebook 04 est indépendant et peut être lancé seul.

```
01 -> 02 -> 03 -> 04 (indépendant)
```

---

## Données

Les données Archelec sont mises à disposition par Sciences Po / CEVIPOF dans le cadre de leurs archives électorales. Pour y accéder : [archelec.sciencespo.fr](https://archelec.sciencespo.fr).

