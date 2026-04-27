"""
function.py
===========
Fonctions utilitaires pour le projet NLP Archelec.
Utilisées dans les notebooks d'analyse du corpus Archelec 1993.
"""

import re
import unicodedata
import numpy as np
import pandas as pd


# ============================================================
# 1. NETTOYAGE DES TEXTES
# ============================================================

# Patterns OCR spécifiques au corpus Archelec / Sciences Po CEVIPOF
OCR_PATTERNS = [
    (r"sciences\s+po\s*/\s*fonds\s+cevipof", " "),
    (r"sciences\s+po\s*/\s*fonds\s+cevipov", " "),
    (r"fonds\s+cevipof",                      " "),
    (r"fonds\s+cevipov",                      " "),
    (r"vu[,]?\s*les?\s*candidats?",           " "),
    (r"vu[,]?\s*le\s*candidat",               " "),
    (r"imp\.\s*[^\n]*",                       " "),
    (r"offset[^\n]*",                         " "),
]


def clean_text_for_lda(text: str) -> str:
    """
    Nettoyage orienté LDA (bag-of-words) :
    - mise en minuscules
    - suppression des artefacts OCR Archelec
    - suppression des symboles parasites et de la ponctuation
    - suppression des chiffres isolés
    - normalisation des espaces

    La casse est perdue volontairement — LDA travaille sur
    des fréquences de lemmes, pas sur la forme de surface.
    """
    text = text.lower()
    for pattern, repl in OCR_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"[☒☐•▪■◆●«»\u201c\u201d\u201e+]", " ", text)
    text = re.sub(r"\b\d+\b",     " ", text)
    text = re.sub(r"[^\w\s]",     " ", text)
    text = re.sub(r"\s+",         " ", text).strip()
    return text


def clean_text_for_embeddings(text: str) -> str:
    """
    Nettoyage orienté embeddings (BERTopic / SentenceTransformer) :
    - conserve la casse (les modèles pré-entraînés exploitent les majuscules)
    - même pipeline que pour LDA sinon
    """
    for pattern, repl in OCR_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"[☒☐•▪■◆●«»\u201c\u201d\u201e+]", " ", text)
    text = re.sub(r"\b\d+\b",     " ", text)
    text = re.sub(r"[^\w\s]",     " ", text)
    text = re.sub(r"\s+",         " ", text).strip()
    return text


# ============================================================
# 2. LEMMATISATION
# ============================================================

# Stopwords spécifiques au corpus électoral (artefacts OCR + formules rhétoriques)
EXTRA_STOPS = {
    # Artefacts archivistiques
    "po", "cevipof", "cevipov", "imp", "imprimerie", "offset", "prefet",
    "préfet", "vu",
    # Formules d'adresse omniprésentes
    "madame", "monsieur", "mademoiselle",
    "cher", "chère", "chers", "chères",
    "compatriote", "compatriotes", "concitoyen", "concitoyenne", "concitoyens",
    # Vocabulaire administratif électoral (présent dans 100 % des PF)
    "suppléant", "suppléante", "scrutin",
    "circonscription", "departement", "département",
    "candidature", "legislatif", "législatif", "mars", "election",
    # Sigles partisans résiduels
    "rpr", "udf", "pcf", "mrg", "mdc", "cpnt",
}


def build_lemmatizer(nlp):
    """
    Injecte les stopwords supplémentaires dans le vocabulaire spaCy
    et retourne la fonction de lemmatisation.

    Paramètres
    ----------
    nlp : spacy.Language
        Modèle spaCy français déjà chargé.

    Retour
    ------
    lemmatize : callable
        Fonction str -> list[str]
    """
    for word in EXTRA_STOPS:
        nlp.vocab[word].is_stop = True

    def lemmatize(text: str) -> list:
        """Retourne la liste de lemmes filtrés pour un texte donné."""
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and len(token) > 2
        ]

    return lemmatize


# ============================================================
# 3. NORMALISATION ET CLASSIFICATION DU SOUTIEN PARTISAN
# ============================================================

def strip_accents(text: str) -> str:
    """Supprime les accents. Ex : 'Écologie' -> 'Ecologie'."""
    if pd.isna(text):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(c)
    )


def normalize_support(text: str) -> str:
    """
    Normalise une chaîne de soutien politique :
    minuscules, sans accents, espaces homogènes, séparateurs ';' propres.
    """
    if pd.isna(text) or str(text).strip() == "":
        return "non mentionne"
    text = strip_accents(str(text))
    text = text.lower().replace("'", "'").replace("`", "'")
    text = re.sub(r"\s+",    " ", text)
    text = re.sub(r"\s*;\s*", ";", text)
    return text.strip()


def split_supports(text: str) -> list:
    """Découpe un soutien multi-partis en liste de composantes."""
    text = normalize_support(text)
    if text in ("", "nan", "none", "non mentionne"):
        return ["non mentionne"]
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts or ["non mentionne"]


def classify_single_component(component: str) -> str:
    """
    Classe une composante individuelle de soutien dans une famille politique.

    Familles retournées :
    - 'Extrême gauche'
    - 'Front national'
    - 'PCF'
    - 'Droite parlementaire (RPR/UDF)'
    - 'Gauche gouvernementale (PS/MRG/MDC)'
    - 'Écologistes'
    - 'Indépendants / divers'
    - 'Non mentionné'
    - 'Autres'
    """
    c = normalize_support(component)

    if c in ("", "nan", "none", "non mentionne"):
        return "Non mentionné"

    patterns = {
        "Extrême gauche": [
            r"\blutte ouvriere\b", r"\bligue communiste revolutionnaire\b",
            r"\blcr\b", r"\bparti des travailleurs\b",
            r"\balliance europeenne des travailleurs\b",
            r"\bjeunesses communistes revolutionnaires\b",
            r"\balternative rouge et verte\b",
            r"\balternative democratie socialisme\b",
        ],
        "Front national": [
            r"\bfront national\b", r"\bfn\b",
        ],
        "PCF": [
            r"\bparti communiste francais\b", r"\bpcf\b", r"\bcommuniste\b",
        ],
        "Droite parlementaire (RPR/UDF)": [
            r"\brassemblement pour la republique\b", r"\brpr\b",
            r"\bunion pour la democratie francaise\b", r"\budf\b",
            r"\bcentre des democrates sociaux\b", r"\bcds\b",
            r"\bcentre national des independants\b", r"\bcni\b",
            r"\bdemocratie chretienne francaise\b",
            r"\bparti republicain\b", r"\bmouvement des reformateurs\b",
            r"\bgaulliste\b", r"\bunion des democrates de progres\b",
        ],
        "Gauche gouvernementale (PS/MRG/MDC)": [
            r"\bparti socialiste\b", r"\bps\b",
            r"\bmouvement des radicaux de gauche\b", r"\bmrg\b",
            r"\bmouvement des citoyens\b", r"\bmdc\b",
            r"\bparti radical\b", r"\bsocialiste[s]?\b",
            r"\bparti social democrate\b",
        ],
        "Écologistes": [
            r"\bverts?\b", r"\becologie\b", r"\becologiste[s]?\b",
            r"\bgeneration ecologie\b",
            r"\bnouveaux ecologistes du rassemblement nature et animaux\b",
            r"\bunion nationale ecologiste\b", r"\banimaux?\b",
            r"\bnature\b", r"\bsos environnement\b",
        ],
        "Indépendants / divers": [
            r"\bindependants?\b", r"\bsans etiquette\b",
            r"\bdivers droite\b", r"\bdivers gauche\b",
            r"\bnon inscri[ts]+\b", r"\bapolitique\b",
        ],
    }

    for famille, regexps in patterns.items():
        if any(re.search(p, c) for p in regexps):
            return famille

    return "Autres"


FAMILLES_FINALES = [
    "Droite parlementaire (RPR/UDF)",
    "Gauche gouvernementale (PS/MRG/MDC)",
    "Front national",
    "PCF",
    "Extrême gauche",
    "Écologistes",
    "Indépendants / divers",
    "Non mentionné",
    "Ambigu",
    "Autres",
]


def classify_candidate_support(raw_support: str) -> pd.Series:
    """
    Classe le soutien d'un candidat dans une famille partisane unique.

    Logique de résolution des multi-soutiens :
    - Soutien unique            → famille directe
    - Multi-soutien cohérent   → famille commune
    - Famille principale + résidus → famille principale
    - Plusieurs familles fortes → 'Ambigu' (flag_a_verifier = True)
    """
    components = split_supports(raw_support)
    families   = [classify_single_component(c) for c in components]
    unique_fam = list(dict.fromkeys(families))
    nb         = len(components)

    # Non mentionné
    if unique_fam == ["Non mentionné"]:
        return pd.Series({
            "soutiens_liste": components, "nb_soutiens": nb,
            "type_soutien": "non_mentionne", "familles_detectees": unique_fam,
            "famille_partisane": "Non mentionné", "flag_a_verifier": False,
            "raison_codage": "Soutien non mentionné",
        })

    # Soutien unique
    if nb == 1:
        return pd.Series({
            "soutiens_liste": components, "nb_soutiens": nb,
            "type_soutien": "unique", "familles_detectees": unique_fam,
            "famille_partisane": unique_fam[0], "flag_a_verifier": False,
            "raison_codage": "Soutien unique",
        })

    # Multi-soutien, même famille
    if len(unique_fam) == 1:
        return pd.Series({
            "soutiens_liste": components, "nb_soutiens": nb,
            "type_soutien": "multi_meme_famille", "familles_detectees": unique_fam,
            "famille_partisane": unique_fam[0], "flag_a_verifier": False,
            "raison_codage": "Multi-soutien cohérent",
        })

    # Une famille principale + résidus
    core = [f for f in unique_fam if f not in ("Autres", "Non mentionné")]
    if len(core) == 1:
        return pd.Series({
            "soutiens_liste": components, "nb_soutiens": nb,
            "type_soutien": "multi_famille_principale_plus_residuel",
            "familles_detectees": unique_fam,
            "famille_partisane": core[0], "flag_a_verifier": False,
            "raison_codage": "Famille principale + composantes résiduelles",
        })

    # Plusieurs familles fortes → ambigu
    return pd.Series({
        "soutiens_liste": components, "nb_soutiens": nb,
        "type_soutien": "multi_familles_mixtes", "familles_detectees": unique_fam,
        "famille_partisane": "Ambigu", "flag_a_verifier": True,
        "raison_codage": "Plusieurs familles politiques : vérification manuelle requise",
    })


# ============================================================
# 4. MÉTRIQUES STYLISTIQUES
# ============================================================

def msttr(tokens: list, window: int = 100) -> float:
    """
    Mean Segmental Type-Token Ratio (MSTTR).

    Mesure la richesse lexicale sur des fenêtres de longueur fixe
    afin de s'affranchir de l'effet de la longueur du document.

    Paramètres
    ----------
    tokens : list[str]
        Liste de tokens (lemmes).
    window : int
        Taille de la fenêtre (défaut : 100).

    Retour
    ------
    float : score MSTTR ∈ [0, 1]
    """
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    segments = [tokens[i: i + window] for i in range(0, len(tokens) - window + 1, window)]
    return float(np.mean([len(set(seg)) / window for seg in segments]))
