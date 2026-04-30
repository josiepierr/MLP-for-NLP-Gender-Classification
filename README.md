# Analyse computationnelle des professions de foi - Législatives 1993


Ce projet applique des méthodes de traitement automatique du langage naturel (NLP) au corpus Archelec des élections législatives françaises de 1993. L'objectif est double : identifier les structures thématiques qui structurent le discours partisan, et examiner si le sexe des candidats laisse une empreinte stylistique mesurable dans leurs professions de foi.

---

## Contexte

Les élections législatives de mars 1993 se tiennent dans un contexte de cohabitation : la gauche au pouvoir depuis 1988 fait face à une vague de droite qui lui coûtera 213 sièges. Le corpus Archelec, constitué par Sciences Po / CEVIPOF, rassemble les professions de foi numérisées de cette élection - l'un des rares moments où chaque candidat, qu'il soit issu d'un grand parti ou d'une formation marginale, dispose du même espace pour s'adresser à l'ensemble des électeurs de sa circonscription.

5 835 documents, 43 colonnes de métadonnées, une fenêtre ouverte sur la diversité idéologique d'une époque.

---

## Structure du projet

```
MLP-for-NLP-Gender-Classification/
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
├── report/
│   ├── main.tex                       # Rapport NeurIPS
│   └── references.bib
└── requirements.txt
```

---

## Notebooks

### 01 - Préparation des données
Chargement des textes bruts depuis le ZIP, correction des artefacts OCR, normalisation Unicode NFC, tokenisation et lemmatisation avec spaCy (`fr_core_news_md`). Construction de trois représentations textuelles selon l'usage downstream : texte lemmatisé pour la LDA, texte normalisé pour les embeddings, texte brut nettoyé pour la stylométrie. Calcul du MSTTR (richesse lexicale) sur fenêtre de 100 tokens. Le corpus final comprend 5 835 documents avec une moyenne de 310 tokens par document.

### 02 - Modélisation thématique
Trois approches comparées :
- **LDA** (scikit-learn) : K=10 sélectionné par cohérence Cv maximale (0.654). Les dix thèmes couvrent les grandes familles idéologiques de 1993 - écologie, Front national, extrême gauche, PCF, discours généraliste - ainsi que deux artefacts documentés : 129 professions de foi en allemand (Alsace-Moselle) et le Parti de la Loi Naturelle (mouvement Maharishi).
- **NMF** : résultats plus contrastés que la LDA, avec une meilleure séparation des deux courants écologistes (Verts vs CPNT).
- **BERTopic** : cinq clusters avec un cluster dominant regroupant 72 % des documents, ce qui reflète l'homogénéité sémantique intrinsèque des professions de foi.

### 03 - Analyse croisée thèmes × partis × genre
Jointure des proportions LDA avec les métadonnées. Calcul des indices de spécialisation thématique par log-ratio (thème vs. distribution globale). Résultats saillants : le thème immigration est surreprésenté au FN d'un facteur 8,35 ; le thème lutte ouvrière est surreprésenté chez l'extrême gauche d'un facteur 13,45. Croisement exploratoire thèmes × sexe.

### 04 - Stylométrie et signal de genre
**Notebook entièrement indépendant** (aucune dépendance aux sorties des notebooks précédents). Chargement direct du ZIP, calcul du MSTTR en interne. Extraction de 20+ features stylistiques issues de la littérature (Argamon & Koppel 2003, Pennebaker 2001) : taux de déterminants, nominalisations, bigrammes POS, hedging, profondeur syntaxique, vocabulaire social/économique, registre vous/tu, etc. Tests de Mann-Whitney avec correction de Bonferroni, régressions OLS avec contrôle du parti, et classification supervisée par régression logistique.

---

## Résultats principaux

**Topic modeling.** La LDA avec K=10 produit des thèmes thématiquement cohérents et politiquement interprétables. Les familles partisanes présentent des spécialisations très marquées : le FN monopolise le thème immigration, l'extrême gauche le thème de la lutte des classes, le CPNT et les Verts occupent deux sous-espaces distincts de l'écologie. Le discours généraliste (thème 7) est partagé par tous les partis mais prédomine à droite.

**Signal de genre.** Sur les 20+ features testées, 9 sont robustes aux deux protocoles de test (Mann-Whitney + OLS contrôlé par le parti). Les femmes candidates utilisent davantage de déterminants, de séquences déterminant+nom, de nominalisations et de conjonctions de coordination. Elles ont aussi recours plus fréquemment au hedging (atténuation modale). Les hommes se distinguent par une utilisation plus élevée de noms propres en position prépositionnelle - ce qui traduit plus de références à des lieux, des personnalités et des institutions. La richesse lexicale (MSTTR) et la profondeur syntaxique sont légèrement supérieures chez les femmes après contrôle du parti.

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

### Dépendances principales
- Python 3.10+
- spacy + fr_core_news_md
- scikit-learn
- bertopic
- sentence-transformers
- statsmodels
- matplotlib / seaborn

---

## Ordre d'exécution

Les notebooks doivent être lancés dans l'ordre numérique. Le notebook 04 est indépendant et peut être lancé seul.

```
01 → 02 → 03 → 04 (indépendant)
```

---

## Données

Les données Archelec sont mises à disposition par Sciences Po / CEVIPOF dans le cadre de leurs archives électorales. Elles ne sont pas redistribuées dans ce dépôt. Pour y accéder : [archelec.sciencespo.fr](https://archelec.sciencespo.fr).

