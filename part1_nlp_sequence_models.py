"""
Atelier 3 - Deep Learning (Part 1)
Arabic NLP Preprocessing Pipeline & Recurrent Sequence Models (RNN, BiRNN, GRU, LSTM)
with Hyperparameter Tuning and BLEU Evaluation.

Université Abdelmalek Essaadi - FST Tanger (Département Génie Informatique)
Filière  : Cycle d'Ingénieur LSI - 3ème Année
Module   : Deep Learning (Pr. Lotfi ELAACHAK)
Étudiant : Mohammed SALHI
"""

import os
import re
import random
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem.arlstem2 import ARLSTem2
from nltk.stem.isri import ISRIStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Reproductibilité
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. Collecte de données en arabe via Scraping (Cybersécurité & Cryptographie)
# =====================================================================
def scrape_arabic_cybersecurity_corpus(csv_path="data/arabic_dataset.csv"):
    """
    Collecte des paragraphes en langue arabe portant sur la Cybersécurité,
    la Sécurité de l'Information et la Cryptographie à l'aide de BeautifulSoup.
    Attribue un score de pertinence compris entre 0.0 et 10.0 basé sur la
    densité et la couverture de mots-clés thématiques.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print("[1/4] Collecte du corpus arabe (Cybersécurité & Sécurité de l'Information)...")

    urls = [
        "https://ar.wikipedia.org/wiki/%D8%A3%D9%85%D9%86_%D8%A7%D9%84%D8%AD%D8%A7%D8%B3%D9%88%D8%A8",          # أمن الحاسوب
        "https://ar.wikipedia.org/wiki/%D8%A3%D9%85%D9%86_%D8%A7%D9%84%D9%85%D8%B9%D9%84%D9%88%D9%85%D8%A7%D8%AA",  # أمن المعلومات
        "https://ar.wikipedia.org/wiki/%D8%AA%D8%B9%D9%85%D9%8A%D8%A9",                                             # تعمية (Cryptographie)
        "https://ar.wikipedia.org/wiki/%D8%A8%D8%B1%D9%85%D8%AC%D9%8A%D8%A9_%D8%AE%D8%A8%D9%8A%D8%AB%D8%A9",        # برمجية خبيثة
        "https://ar.wikipedia.org/wiki/%D8%AC%D8%AF%D8%A7%D8%B1_%D8%AD%D9%85%D8%A7%D9%8A%D8%A9",                    # جدار حماية
    ]

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    seen = set()
    paragraphs = []

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                for p in soup.find_all("p"):
                    raw_txt = p.get_text(" ", strip=True)
                    # Nettoyage des renvois de notes Wikipedia [1], [أ], [بحاجة لمصدر]
                    txt = re.sub(r"\[[^\]]*\]", "", raw_txt)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if len(txt) >= 60 and txt not in seen:
                        seen.add(txt)
                        paragraphs.append(txt)
        except Exception as exc:
            print(f"  Avertissement : échec de récupération pour {url} ({exc})")

    # Pondération thématique (Cybersécurité, Cryptographie, Réseaux)
    weighted_keywords = {
        "أمن": 1.6, "حماية": 1.5, "تشفير": 1.8, "تعمية": 1.8, "اختراق": 1.8,
        "هجوم": 1.6, "هجمات": 1.6, "خبيثة": 1.7, "برمجية": 1.4, "برمجيات": 1.4,
        "جدار": 1.6, "ثغرة": 1.7, "ثغرات": 1.7, "فيروس": 1.6, "فيروسات": 1.6,
        "تهديد": 1.5, "تهديدات": 1.5, "مفتاح": 1.3, "مفاتيح": 1.3, "توثيق": 1.5,
        "مصادقة": 1.5, "خصوصية": 1.4, "معلومات": 1.1, "بيانات": 1.1, "شبكة": 1.2,
        "شبكات": 1.2, "حاسوب": 1.0, "نظام": 0.9, "أنظمة": 0.9, "إنترنت": 1.0,
        "سرية": 1.5, "سلامة": 1.3, "مستخدم": 0.9, "وصول": 1.0, "كلمة": 1.0,
    }

    records = []
    for txt in paragraphs:
        score_sum = 0.0
        for kw, weight in weighted_keywords.items():
            if kw in txt:
                score_sum += weight
        # Normalisation déterministe dans [0.5, 10.0]
        normalized = min(10.0, max(0.5, 0.8 + score_sum * 0.72))
        records.append({"Text": txt, "Score": round(normalized, 1)})

    # Jeu de secours équilibré si exécution hors-ligne
    if len(records) < 40:
        fallback_samples = [
            ("يعد أمن المعلومات وتشفير البيانات وحماية الشبكات من الهجمات السيبرانية والبرمجيات الخبيثة ركيزة أساسية في الأنظمة الحديثة.", 9.6),
            ("تستخدم خوارزميات التعمية الحديثة مفاتيح رقمية معقدة لضمان سرية المعلومات وسلامة البيانات أثناء نقلها عبر الإنترنت.", 9.2),
            ("يعمل جدار الحماية على مراقبة حركة المرور في الشبكة ومنع محاولات الاختراق والوصول غير المصرح به إلى الخوادم والأنظمة.", 8.8),
            ("تستغل البرمجيات الخبيثة مثل الفيروسات وبرامج الفدية الثغرات الأمنية في أنظمة التشغيل للوصول إلى البيانات الحساسة.", 8.5),
            ("تساعد المصادقة متعددة العوامل في تعزيز أمن حسابات المستخدمين وتقليل مخاطر سرقة كلمات المرور والاختراق.", 8.0),
            ("يتطلب تأمين الحوسبة السحابية سياسات صارمة للتحكم في الوصول ومراقبة التهديدات الأمنية بشكل مستمر.", 7.4),
            ("تعتمد البروتوكولات الآمنة مثل بروتوكول طبقة المقابس الآمنة على التشفير لحماية الاتصالات بين المتصفح والخادم.", 6.8),
            ("يُعد تدريب الموظفين على كشف رسائل التصيد الاحتيالي خطوة مهمة للحد من الهجمات الهندسية الاجتماعية.", 6.2),
            ("تقوم أنظمة كشف التسلل بتحليل سجلات الشبكة للتعرف على الأنشطة المشبوهة والتنبيه الفوري لفرق الأمن.", 5.8),
            ("توفر النسخ الاحتياطية المنتظمة للبيانات وسيلة فعالة لاستعادة الأنظمة بعد الأعطال التقنية أو الهجمات.", 5.0),
            ("تطورت هندسة البرمجيات لتشمل اختبارات آلية تضمن جودة التطبيقات واستقرارها في بيئات الإنتاج.", 3.6),
            ("تعتمد قواعد البيانات العلائقية على لغة الاستعلام الهيكلية لتنظيم السجلات وتسهيل البحث السريع.", 2.8),
            ("تعتبر الطاقة الشمسية وطاقة الرياح من أهم مصادر الطاقة المتجددة النظيفة الصديقة للبيئة.", 1.2),
            ("تسهم القراءة المنتظمة في تنمية المهارات اللغوية وتوسيع المعرفة الثقافية لدى الطلاب والباحثين.", 0.8),
        ]
        for text_item, score_item in fallback_samples:
            records.append({"Text": text_item, "Score": score_item})

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  -> Dataset sauvegardé dans {csv_path} ({len(df)} échantillons).")
    print("  -> Aperçu des 3 premières lignes :")
    for i, row in df.head(3).iterrows():
        preview = row["Text"][:95] + ("..." if len(row["Text"]) > 95 else "")
        print(f"     [{i}] Score={row['Score']:<4} | {preview}")
    return df


# =====================================================================
# 2. Pipeline de Prétraitement NLP Arabe
#    (Nettoyage, Tokenization, Stopwords, Lemmatisation, Stemming, Discrétisation)
# =====================================================================
class ArabicTextPreprocessor:
    """
    Pipeline NLP complet pour la langue arabe :
    1. Suppression des diacritiques (Tashkeel) et du Tatweel.
    2. Normalisation orthographique (Alif, Yaa, Taa Marbouta, Hamza).
    3. Tokenization par expression régulière arabe.
    4. Filtrage des mots vides (Stopwords NLTK + liste complémentaire).
    5. Lemmatisation légère via ARLSTem2.
    6. Racinisation (Stemming) via ISRIStemmer.
    7. Discrétisation du score continu (0-10) en 3 classes de pertinence.
    8. Encodage en indices et Padding à longueur fixe.
    """

    DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")
    ARABIC_TOKEN_PATTERN = re.compile(r"[\u0621-\u064A]+")

    def __init__(self):
        nltk.download("stopwords", quiet=True)
        extra_stops = {
            "ان", "انه", "انها", "او", "اي", "ايضا", "بشكل", "بين", "تم", "تلك",
            "حيث", "خلال", "ذلك", "على", "عن", "عند", "في", "قد", "كان", "كانت",
            "كل", "كما", "لا", "له", "لها", "ما", "مع", "من", "هذا", "هذه", "هو",
            "هي", "والتي", "الذي", "التي", "التى", "الى", "إلى", "على", "عبر", "يمكن",
        }
        self.stop_words = set(stopwords.words("arabic")) | extra_stops
        self.lemmatizer = ARLSTem2()
        self.stemmer = ISRIStemmer()
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def normalize(self, text):
        text = self.DIACRITICS_PATTERN.sub("", text)
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"ؤ", "و", text)
        text = re.sub(r"ئ", "ي", text)
        return text

    def tokenize(self, text):
        normalized = self.normalize(text)
        return self.ARABIC_TOKEN_PATTERN.findall(normalized)

    def remove_stopwords(self, tokens):
        return [tok for tok in tokens if tok not in self.stop_words and len(tok) > 1]

    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.stem(tok) for tok in tokens]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(tok) for tok in tokens]

    def preprocess(self, text):
        tokens = self.tokenize(text)
        filtered = self.remove_stopwords(tokens)
        lemmas = self.lemmatize_tokens(filtered)
        stems = self.stem_tokens(lemmas)
        return tokens, lemmas, stems

    @staticmethod
    def discretize_scores(scores):
        """
        Discrétise les scores continus [0, 10] en 3 classes de pertinence :
          - Classe 0 : Pertinence Faible   [0.0, 4.0)
          - Classe 1 : Pertinence Moyenne  [4.0, 7.0)
          - Classe 2 : Pertinence Élevée   [7.0, 10.0]
        """
        bins = [-0.1, 4.0, 7.0, 10.1]
        labels = pd.cut(scores, bins=bins, labels=[0, 1, 2], right=False).astype(int)
        return labels.tolist()

    def build_vocab(self, token_lists, min_freq=2):
        counts = {}
        for tokens in token_lists:
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
        for tok, freq in sorted(counts.items()):
            if freq >= min_freq and tok not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[tok] = idx
                self.idx2word[idx] = tok

    def encode_and_pad(self, tokens, max_len=35):
        unk_id = self.word2idx["<UNK>"]
        pad_id = self.word2idx["<PAD>"]
        ids = [self.word2idx.get(tok, unk_id) for tok in tokens[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        return ids


class ArabicSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 3. Architectures Séquentielles PyTorch : RNN, BiRNN, GRU, LSTM
# =====================================================================
def masked_mean_pooling(outputs, x):
    """Calcule la moyenne des représentations cachées uniquement sur les tokens non-PAD."""
    mask = (x != 0).unsqueeze(-1).float()
    summed = (outputs * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths


class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        emb = self.emb_drop(self.embedding(x))
        out, _ = self.rnn(emb)
        pooled = self.norm(masked_mean_pooling(out, x))
        return self.head(pooled)


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x):
        emb = self.emb_drop(self.embedding(x))
        out, _ = self.rnn(emb)
        pooled = self.norm(masked_mean_pooling(out, x))
        return self.head(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        emb = self.emb_drop(self.embedding(x))
        out, _ = self.gru(emb)
        pooled = self.norm(masked_mean_pooling(out, x))
        return self.head(pooled)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        emb = self.emb_drop(self.embedding(x))
        out, _ = self.lstm(emb)
        pooled = self.norm(masked_mean_pooling(out, x))
        return self.head(pooled)


# =====================================================================
# 4. Entraînement, Optimisation des Hyperparamètres & Évaluation (Métriques + BLEU)
# =====================================================================
def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            total_loss += criterion(logits, y_batch).item()
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "preds": preds,
        "targets": targets,
    }


def train_one_config(model_cls, vocab_size, train_loader, val_loader, config, epochs=22):
    torch.manual_seed(SEED)
    model = model_cls(
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=3,
        dropout=config["dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics = evaluate_loader(model, val_loader, criterion)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_loader(model, val_loader, criterion)
    return model, final_val


def compute_class_alignment_bleu(train_tokens, train_labels, test_tokens, test_preds):
    """
    Calcule le score BLEU (NLTK sentence_bleu avec lissage) mesurant la cohérence
    n-gramme de chaque texte de test par rapport au corpus de référence de la classe
    de pertinence prédite par le modèle.
    """
    smoother = SmoothingFunction().method1
    class_references = {0: [], 1: [], 2: []}
    for toks, lbl in zip(train_tokens, train_labels):
        if toks:
            class_references[lbl].append(toks)

    scores = []
    for hyp, pred_cls in zip(test_tokens, test_preds):
        refs = class_references.get(pred_cls, [])
        if hyp and refs:
            score = sentence_bleu(refs, hyp, weights=(0.5, 0.5), smoothing_function=smoother)
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def run_part1():
    print("=" * 72)
    print("LAB 3 - PARTIE 1 : NLP ARABE & MODÈLES SÉQUENTIELS (Mohammed SALHI)")
    print("=" * 72)

    # Étape 1 : Scraping & Dataset
    df = scrape_arabic_cybersecurity_corpus("data/arabic_dataset.csv")

    # Étape 2 : Pipeline NLP
    print("\n[2/4] Exécution du pipeline NLP (Nettoyage, Lemmatisation, Stemming, Discrétisation)...")
    preprocessor = ArabicTextPreprocessor()

    raw_tokens_all, lemma_tokens_all, stem_tokens_all = [], [], []
    for text in df["Text"]:
        raw_t, lem_t, stm_t = preprocessor.preprocess(text)
        raw_tokens_all.append(raw_t)
        lemma_tokens_all.append(lem_t)
        stem_tokens_all.append(stm_t)

    raw_vocab_size = len({t for doc in raw_tokens_all for t in doc})
    lemma_vocab_size = len({t for doc in lemma_tokens_all for t in doc})
    stem_vocab_size = len({t for doc in stem_tokens_all for t in doc})

    preprocessor.build_vocab(stem_tokens_all, min_freq=2)
    max_len = 35
    sequences = [preprocessor.encode_and_pad(toks, max_len=max_len) for toks in stem_tokens_all]
    labels = preprocessor.discretize_scores(df["Score"])

    print(f"  -> Vocabulaire brut          : {raw_vocab_size} tokens uniques")
    print(f"  -> Après lemmatisation       : {lemma_vocab_size} lemmes uniques (ARLSTem2)")
    print(f"  -> Après stemming            : {stem_vocab_size} racines uniques (ISRIStemmer)")
    print(f"  -> Vocabulaire retenu (f>=2) : {len(preprocessor.word2idx)} tokens")
    print(f"  -> Répartition des classes   : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # Split stratifié Train (64%) / Validation (16%) / Test (20%)
    indices = list(range(len(sequences)))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        indices, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval
    )

    X_train = [sequences[i] for i in idx_train]
    X_val = [sequences[i] for i in idx_val]
    X_test = [sequences[i] for i in idx_test]

    train_loader = DataLoader(ArabicSequenceDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(ArabicSequenceDataset(X_val, y_val), batch_size=16, shuffle=False)
    test_loader = DataLoader(ArabicSequenceDataset(X_test, y_test), batch_size=16, shuffle=False)

    # Étape 3 : Entraînement & Hyperparameter Tuning
    print("\n[3/4] Entraînement et Tuning d'Hyperparamètres (RNN, BiRNN, GRU, LSTM)...")
    search_space = [
        {"embed_dim": 64, "hidden_dim": 64, "lr": 0.003, "dropout": 0.2},
        {"embed_dim": 64, "hidden_dim": 128, "lr": 0.002, "dropout": 0.3},
        {"embed_dim": 96, "hidden_dim": 64, "lr": 0.005, "dropout": 0.3},
    ]

    model_registry = {
        "Simple RNN": VanillaRNNClassifier,
        "Bidirectional RNN": BiRNNClassifier,
        "GRU": GRUClassifier,
        "LSTM": LSTMClassifier,
    }

    criterion = nn.CrossEntropyLoss()
    train_tokens_for_bleu = [stem_tokens_all[i] for i in idx_train]
    test_tokens_for_bleu = [stem_tokens_all[i] for i in idx_test]

    results = {}
    for name, cls in model_registry.items():
        best_model = None
        best_cfg = None
        best_val = None

        for cfg in search_space:
            model, val_res = train_one_config(
                cls, len(preprocessor.word2idx), train_loader, val_loader, cfg, epochs=22
            )
            if best_val is None or val_res["f1"] > best_val["f1"]:
                best_model = model
                best_cfg = cfg
                best_val = val_res

        # Évaluation finale sur le jeu de test
        test_res = evaluate_loader(best_model, test_loader, criterion)
        bleu = compute_class_alignment_bleu(
            train_tokens_for_bleu, y_train, test_tokens_for_bleu, test_res["preds"]
        )
        test_res["bleu"] = bleu
        test_res["best_config"] = best_cfg
        results[name] = test_res

        print(
            f"  -> {name:<18} | Meilleur config: h={best_cfg['hidden_dim']}, "
            f"emb={best_cfg['embed_dim']}, lr={best_cfg['lr']}, drop={best_cfg['dropout']} "
            f"| Val F1={best_val['f1']:.4f} | Test Acc={test_res['accuracy']:.4f}"
        )

    # Étape 4 : Tableau comparatif final
    print("\n[4/4] Résultats Comparatifs sur le Jeu de Test :")
    print("=" * 82)
    print(
        f"{'Modèle':<20} | {'Test Loss':<9} | {'Accuracy':<9} | "
        f"{'Precision':<9} | {'F1-Score':<9} | {'BLEU':<8}"
    )
    print("-" * 82)
    for name, m in results.items():
        print(
            f"{name:<20} | {m['loss']:<9.4f} | {m['accuracy']:<9.4f} | "
            f"{m['precision']:<9.4f} | {m['f1']:<9.4f} | {m['bleu']:<8.4f}"
        )
    print("=" * 82)

    return results


if __name__ == "__main__":
    run_part1()
