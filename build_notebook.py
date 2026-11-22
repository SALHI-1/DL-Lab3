"""
Script de génération du Jupyter Notebook pour l'Atelier 3 - Deep Learning.
Étudiant : Mohammed SALHI (LSI 3ème Année - FST Tanger)
"""

import nbformat as nbf


def build_notebook(output_path="Atelier3_DeepLearning.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. En-tête académique
    cells.append(nbf.v4.new_markdown_cell("""# Université Abdelmalek Essaadi
## Faculté des Sciences et Techniques de Tanger (FSTT)
### Département de Génie Informatique

- **Filière :** Cycle d'Ingénieur LSI (Logiciel et Systèmes Intelligents) — 3ème Année
- **Module :** Deep Learning
- **Encadrant :** Pr. Lotfi ELAACHAK
- **Étudiant :** Mohammed SALHI

---

# Lab 3 : NLP avec Modèles Séquentiels Récurrents & Transformers (GPT-2)
"""))

    # 2. Objectifs
    cells.append(nbf.v4.new_markdown_cell("""## Objectifs de l'Atelier
Conformément à l'énoncé de l'Atelier 3 (`Atelier3.pdf`), ce notebook couvre :
1. **Collecte de données en arabe (Web Scraping)** via `BeautifulSoup` sur la thématique de la **Cybersécurité et Sécurité de l'Information**, avec attribution d'un score de pertinence entre `0` et `10`.
2. **Pipeline de Prétraitement NLP Arabe** complet : nettoyage (Tashkeel, normalisation orthographique), tokenisation, suppression des stopwords, **lemmatisation (`ARLSTem2`)**, **stemming (`ISRIStemmer`)** et discrétisation du score en 3 classes.
3. **Entraînement et Optimisation d'Hyperparamètres** de 4 architectures PyTorch : **Simple RNN**, **Bidirectional RNN**, **GRU** et **LSTM**.
4. **Évaluation Comparative** à l'aide des métriques standards (*Loss*, *Accuracy*, *Precision*, *Recall*, *F1-Score*) ainsi que du **Score BLEU**.
5. **Fine-Tuning du Transformer GPT-2** pré-entraîné sur un jeu de données personnalisé dédié à la cybersécurité et génération de nouveaux paragraphes à partir de phrases d'amorce.
"""))

    # 3. Imports & Configuration
    cells.append(nbf.v4.new_markdown_cell("## 0. Configuration de l'Environnement et Imports"))
    cells.append(nbf.v4.new_code_cell("""import os
import re
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import nltk
from nltk.corpus import stopwords
from nltk.stem.arlstem2 import ARLSTem2
from nltk.stem.isri import ISRIStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Environnement prêt — Device utilisé : {DEVICE}")
"""))

    # 4. Partie 1 — Scraping
    cells.append(nbf.v4.new_markdown_cell("""---
# Partie 1 : Tâche de Classification (NLP Arabe & Modèles Séquentiels)

### 1. Collecte des données en arabe via Scraping (`BeautifulSoup`)
Nous collectons des paragraphes en langue arabe depuis Wikipédia Arabe sur le thème de la **Cybersécurité, Cryptographie et Sécurité des Réseaux**, puis nous calculons un score de pertinence compris entre `0.0` et `10.0`.
"""))

    cells.append(nbf.v4.new_code_cell("""urls = [
    "https://ar.wikipedia.org/wiki/%D8%A3%D9%85%D9%86_%D8%A7%D9%84%D8%AD%D8%A7%D8%B3%D9%88%D8%A8",
    "https://ar.wikipedia.org/wiki/%D8%A3%D9%85%D9%86_%D8%A7%D9%84%D9%85%D8%B9%D9%84%D9%88%D9%85%D8%A7%D8%AA",
    "https://ar.wikipedia.org/wiki/%D8%AA%D8%B9%D9%85%D9%8A%D8%A9",
    "https://ar.wikipedia.org/wiki/%D8%A8%D8%B1%D9%85%D8%AC%D9%8A%D8%A9_%D8%AE%D8%A8%D9%8A%D8%AB%D8%A9",
    "https://ar.wikipedia.org/wiki/%D8%AC%D8%AF%D8%A7%D8%B1_%D8%AD%D9%85%D8%A7%D9%8A%D8%A9",
]

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
seen, paragraphs = set(), []

for url in urls:
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            for p in soup.find_all("p"):
                txt = re.sub(r"\\[[^\\]]*\\]", "", p.get_text(" ", strip=True))
                txt = re.sub(r"\\s+", " ", txt).strip()
                if len(txt) >= 60 and txt not in seen:
                    seen.add(txt)
                    paragraphs.append(txt)
    except Exception as exc:
        print(f"Erreur sur {url} : {exc}")

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
    score_sum = sum(w for kw, w in weighted_keywords.items() if kw in txt)
    score = round(min(10.0, max(0.5, 0.8 + score_sum * 0.72)), 1)
    records.append({"Text": txt, "Score": score})

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(records)
df.to_csv("data/arabic_dataset.csv", index=False, encoding="utf-8")
print(f"Nombre total de textes collectés : {len(df)}")
df.head(5)
"""))

    # 5. Partie 1 — Pipeline NLP
    cells.append(nbf.v4.new_markdown_cell("""### 2. Pipeline de Prétraitement NLP Arabe
Le pipeline met en œuvre toutes les étapes demandées :
1. Suppression des diacritiques (Tashkeel) et du Tatweel (`ـ`).
2. Normalisation des lettres arabes (`إأآ` $\\rightarrow$ `ا`, `ى` $\\rightarrow$ `ي`, `ة` $\\rightarrow$ `ه`, `ؤ` $\\rightarrow$ `و`, `ئ` $\\rightarrow$ `ي`).
3. **Tokenization** arabe par expression régulière.
4. Filtrage des **Stopwords** (NLTK + mots-outils complémentaires).
5. **Lemmatisation** morphologique légère (`ARLSTem2`) suivie du **Stemming** (`ISRIStemmer`).
6. **Discrétisation** du score `[0, 10]` en 3 classes (`0: Faible`, `1: Moyenne`, `2: Élevée`).
7. Construction du vocabulaire (`min_freq=2`) et **Padding** (`max_len=35`).
"""))

    cells.append(nbf.v4.new_code_cell("""class ArabicTextPreprocessor:
    DIACRITICS_PATTERN = re.compile(r"[\\u0617-\\u061A\\u064B-\\u0652\\u0670\\u0640]")
    ARABIC_TOKEN_PATTERN = re.compile(r"[\\u0621-\\u064A]+")

    def __init__(self):
        nltk.download("stopwords", quiet=True)
        extra_stops = {
            "ان", "انه", "انها", "او", "اي", "ايضا", "بشكل", "بين", "تم", "تلك",
            "حيث", "خلال", "ذلك", "على", "عن", "عند", "في", "قد", "كان", "كانت",
            "كل", "كما", "لا", "له", "لها", "ما", "مع", "من", "هذا", "هذه", "هو",
            "هي", "والتي", "الذي", "التي", "التى", "الى", "إلى", "عبر", "يمكن",
        }
        self.stop_words = set(stopwords.words("arabic")) | extra_stops
        self.lemmatizer = ARLSTem2()
        self.stemmer = ISRIStemmer()
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def normalize(self, text):
        text = self.DIACRITICS_PATTERN.sub("", text)
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"ؤ", "و", text)
        text = re.sub(r"ئ", "ي", text)
        return text

    def preprocess(self, text):
        tokens = self.ARABIC_TOKEN_PATTERN.findall(self.normalize(text))
        filtered = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        lemmas = [self.lemmatizer.stem(t) for t in filtered]
        stems = [self.stemmer.stem(t) for t in lemmas]
        return tokens, lemmas, stems

    @staticmethod
    def discretize_scores(scores):
        return pd.cut(scores, bins=[-0.1, 4.0, 7.0, 10.1], labels=[0, 1, 2], right=False).astype(int).tolist()

    def build_vocab(self, docs, min_freq=2):
        freq = {}
        for doc in docs:
            for tok in doc:
                freq[tok] = freq.get(tok, 0) + 1
        for tok, count in sorted(freq.items()):
            if count >= min_freq and tok not in self.word2idx:
                self.word2idx[tok] = len(self.word2idx)

    def encode_and_pad(self, tokens, max_len=35):
        ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        return ids + [0] * max(0, max_len - len(ids))

preprocessor = ArabicTextPreprocessor()
raw_all, lemma_all, stem_all = [], [], []
for txt in df["Text"]:
    r, l, s = preprocessor.preprocess(txt)
    raw_all.append(r)
    lemma_all.append(l)
    stem_all.append(s)

preprocessor.build_vocab(stem_all, min_freq=2)
max_len = 35
sequences = [preprocessor.encode_and_pad(toks, max_len=max_len) for toks in stem_all]
labels = preprocessor.discretize_scores(df["Score"])

print(f"Tokens uniques bruts   : {len({t for d in raw_all for t in d})}")
print(f"Lemmes uniques         : {len({t for d in lemma_all for t in d})}")
print(f"Racines uniques        : {len({t for d in stem_all for t in d})}")
print(f"Vocabulaire (freq>=2)  : {len(preprocessor.word2idx)}")
print(f"Distribution classes   : {pd.Series(labels).value_counts().sort_index().to_dict()}")
"""))

    # 6. Partie 1 — Architectures PyTorch
    cells.append(nbf.v4.new_markdown_cell("""### 3. Architectures Récurrentes en PyTorch (RNN, BiRNN, GRU, LSTM)
Chaque modèle combine :
- Une couche `nn.Embedding(padding_idx=0)` suivie d'un `Dropout`.
- Une couche récurrente (`RNN`, `BiRNN`, `GRU` ou `LSTM`).
- Un **Masked Mean Pooling** (moyenne des états cachés uniquement sur les tokens non-PAD) suivi d'une `LayerNorm` et d'une tête de classification linéaire.
"""))

    cells.append(nbf.v4.new_code_cell("""def masked_mean_pooling(outputs, x):
    mask = (x != 0).unsqueeze(-1).float()
    return (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        out, _ = self.rnn(self.emb_drop(self.embedding(x)))
        return self.head(self.norm(masked_mean_pooling(out, x)))

class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x):
        out, _ = self.rnn(self.emb_drop(self.embedding(x)))
        return self.head(self.norm(masked_mean_pooling(out, x)))

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        out, _ = self.gru(self.emb_drop(self.embedding(x)))
        return self.head(self.norm(masked_mean_pooling(out, x)))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        out, _ = self.lstm(self.emb_drop(self.embedding(x)))
        return self.head(self.norm(masked_mean_pooling(out, x)))
"""))

    # 7. Partie 1 — Tuning & Évaluation
    cells.append(nbf.v4.new_markdown_cell("""### 4. Tuning d'Hyperparamètres et Évaluation (Métriques Standards & Score BLEU)
Nous effectuons une recherche par grille sur un découpage stratifié **Train (64%) / Validation (16%) / Test (20%)**, sélectionnons la meilleure configuration selon le **F1-Score** de validation, puis évaluons chaque modèle sur le jeu de test avec les métriques standards et le **Score BLEU**.
"""))

    cells.append(nbf.v4.new_code_cell("""class ArabicSequenceDataset(Dataset):
    def __init__(self, seqs, lbls):
        self.X = torch.tensor(seqs, dtype=torch.long)
        self.y = torch.tensor(lbls, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

indices = list(range(len(sequences)))
idx_tv, idx_test, y_tv, y_test = train_test_split(indices, labels, test_size=0.20, random_state=SEED, stratify=labels)
idx_tr, idx_val, y_tr, y_val = train_test_split(idx_tv, y_tv, test_size=0.20, random_state=SEED, stratify=y_tv)

train_loader = DataLoader(ArabicSequenceDataset([sequences[i] for i in idx_tr], y_tr), batch_size=16, shuffle=True)
val_loader = DataLoader(ArabicSequenceDataset([sequences[i] for i in idx_val], y_val), batch_size=16, shuffle=False)
test_loader = DataLoader(ArabicSequenceDataset([sequences[i] for i in idx_test], y_test), batch_size=16, shuffle=False)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(yb.cpu().tolist())
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)
    return {"loss": total_loss / max(1, len(loader)), "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "preds": preds}

def compute_bleu(train_tokens, train_labels, test_tokens, test_preds):
    smoother = SmoothingFunction().method1
    class_refs = {0: [], 1: [], 2: []}
    for toks, lbl in zip(train_tokens, train_labels):
        if toks:
            class_refs[lbl].append(toks)
    scores = [
        sentence_bleu(class_refs[p], h, weights=(0.5, 0.5), smoothing_function=smoother)
        for h, p in zip(test_tokens, test_preds)
        if h and class_refs[p]
    ]
    return float(np.mean(scores)) if scores else 0.0

search_space = [
    {"embed_dim": 64, "hidden_dim": 64, "lr": 0.003, "dropout": 0.2},
    {"embed_dim": 64, "hidden_dim": 128, "lr": 0.002, "dropout": 0.3},
    {"embed_dim": 96, "hidden_dim": 64, "lr": 0.005, "dropout": 0.3},
]

registry = {
    "Simple RNN": VanillaRNNClassifier,
    "Bidirectional RNN": BiRNNClassifier,
    "GRU": GRUClassifier,
    "LSTM": LSTMClassifier,
}

criterion = nn.CrossEntropyLoss()
train_toks = [stem_all[i] for i in idx_tr]
test_toks = [stem_all[i] for i in idx_test]

rows = []
for name, cls in registry.items():
    best_model, best_val_f1, best_cfg = None, -1.0, None
    for cfg in search_space:
        torch.manual_seed(SEED)
        m = cls(len(preprocessor.word2idx), cfg["embed_dim"], cfg["hidden_dim"], 3, cfg["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        local_best_f1, local_state = -1.0, None
        for _ in range(22):
            m.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = criterion(m(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            vm = evaluate(m, val_loader, criterion)
            if vm["f1"] > local_best_f1:
                local_best_f1 = vm["f1"]
                local_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(local_state)
        if local_best_f1 > best_val_f1:
            best_val_f1, best_model, best_cfg = local_best_f1, m, cfg

    tm = evaluate(best_model, test_loader, criterion)
    bleu = compute_bleu(train_toks, y_tr, test_toks, tm["preds"])
    rows.append({
        "Model": name,
        "Best Hyperparams": f"h={best_cfg['hidden_dim']}, emb={best_cfg['embed_dim']}, lr={best_cfg['lr']}",
        "Test Loss": round(tm["loss"], 4),
        "Accuracy": round(tm["accuracy"], 4),
        "Precision": round(tm["precision"], 4),
        "F1-Score": round(tm["f1"], 4),
        "BLEU Score": round(bleu, 4),
    })

results_df = pd.DataFrame(rows)
results_df
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualisation comparative des 4 architectures
ax = results_df.set_index("Model")[["Accuracy", "F1-Score", "BLEU Score"]].plot(
    kind="bar", figsize=(9, 4.5), color=["#0f766e", "#2563eb", "#d97706"], rot=0
)
ax.set_title("Comparaison des Modèles Séquentiels — Corpus Arabe Cybersécurité (Mohammed SALHI)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
"""))

    # 8. Partie 2 — GPT-2
    cells.append(nbf.v4.new_markdown_cell("""---
# Partie 2 : Transformer & Génération de Texte avec GPT-2

En suivant le tutoriel de référence ([gist mf1024](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7)), nous :
1. Chargeons le modèle **GPT-2** pré-entraîné et son tokenizer BPE.
2. Construisons un dataset personnalisé de 20 paragraphes techniques sur la **Cybersécurité, la Cryptographie et l'IA appliquée à la détection des menaces**.
3. Effectuons un **Fine-Tuning** avec masquage des tokens de padding (`-100`), optimiseur **AdamW** et scheduler linéaire avec warmup.
4. Générons de nouveaux paragraphes cohérents à partir de phrases données (avant et après fine-tuning).
"""))

    cells.append(nbf.v4.new_code_cell("""tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

def generate_paragraph(model, tokenizer, prompt, max_new_tokens=60, temperature=0.75, top_k=40, top_p=0.92):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

torch.manual_seed(SEED)
base_prompt = "Modern cybersecurity architectures rely on"
print("Génération AVANT fine-tuning :")
print(generate_paragraph(gpt2_model, tokenizer, base_prompt, max_new_tokens=55))
"""))

    cells.append(nbf.v4.new_markdown_cell("### 1. Fine-Tuning de GPT-2 sur le Dataset Personnalisé"))
    cells.append(nbf.v4.new_code_cell("""documents = [
    "Modern cybersecurity architectures rely on zero-trust principles that continuously verify every user, device, and network request before granting access.",
    "Zero-trust security and cryptographic protocols protect sensitive enterprise data across hybrid cloud environments and distributed microservices.",
    "Artificial intelligence in threat detection enables security operations centers to identify anomalous network traffic and zero-day attacks in real time.",
    "End-to-end encryption ensures that confidential communications remain unreadable to intermediaries by using strong symmetric and asymmetric keys.",
    "Public key infrastructure establishes digital trust through cryptographic certificates that authenticate servers, applications, and software updates.",
    "Intrusion detection systems combine signature matching with deep learning models to uncover lateral movement inside corporate networks.",
    "Multi-factor authentication significantly reduces the risk of credential stuffing, phishing compromises, and unauthorized privilege escalation.",
    "Web application firewalls inspect incoming HTTP traffic to block SQL injection, cross-site scripting, and automated denial-of-service floods.",
    "Ransomware defense strategies emphasize network segmentation, immutable offline backups, and continuous endpoint detection and response.",
    "Post-quantum cryptography introduces lattice-based mathematical schemes designed to resist future decryption attacks from quantum computers.",
    "Secure software development lifecycles integrate automated static analysis, dependency scanning, and penetration testing into continuous integration pipelines.",
    "Cloud security posture management continuously audits identity permissions, storage buckets, and container images to prevent dangerous misconfigurations.",
    "Security information and event management platforms aggregate telemetry logs from firewalls, endpoints, and identity providers for rapid incident response.",
    "Adversarial machine learning studies how attackers craft subtle input perturbations to evade malware classifiers and neural network detectors.",
    "Federated learning allows multiple organizations to collaboratively train cyber threat intelligence models without sharing raw private logs.",
    "Hardware security modules safeguard root cryptographic keys inside tamper-resistant physical devices for banking and government infrastructure.",
    "Vulnerability management programs prioritize software patches based on exploitability scores, asset criticality, and active threat intelligence.",
    "Network micro-segmentation limits the blast radius of a breach by enforcing granular access policies between individual workloads.",
    "Digital forensics and incident response teams reconstruct attack timelines from memory dumps, disk artifacts, and encrypted packet captures.",
    "Privacy-preserving analytics leverage differential privacy and homomorphic encryption so organizations can extract insights without exposing personal records.",
]

class CyberGPT2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=48):
        self.samples = []
        for t in texts:
            enc = tokenizer(t + " " + tokenizer.eos_token, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

gpt_loader = DataLoader(CyberGPT2Dataset(documents, tokenizer), batch_size=2, shuffle=True)

for p in gpt2_model.transformer.parameters(): p.requires_grad = False
for block in gpt2_model.transformer.h[-2:]:
    for p in block.parameters(): p.requires_grad = True
for p in gpt2_model.transformer.ln_f.parameters(): p.requires_grad = True
for p in gpt2_model.lm_head.parameters(): p.requires_grad = True

trainable = [p for p in gpt2_model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=5e-5, weight_decay=0.01)
epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, len(gpt_loader) * epochs // 10), num_training_steps=len(gpt_loader) * epochs)

loss_history = []
gpt2_model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in gpt_loader:
        out = gpt2_model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    avg_loss = total_loss / len(gpt_loader)
    loss_history.append(avg_loss)
    print(f"Époque {epoch+1}/{epochs} — Loss : {avg_loss:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 2. Génération de Nouveaux Paragraphes après Fine-Tuning"))
    cells.append(nbf.v4.new_code_cell("""prompts = [
    "Modern cybersecurity architectures rely on",
    "Zero-trust security and cryptographic protocols",
    "Artificial intelligence in threat detection",
]

for idx, prompt in enumerate(prompts, 1):
    torch.manual_seed(SEED + idx)
    print(f"[Exemple {idx}] Amorce : {prompt}")
    print(generate_paragraph(gpt2_model, tokenizer, prompt, max_new_tokens=60))
    print("-" * 65)
"""))

    # 9. Synthèse individuelle
    cells.append(nbf.v4.new_markdown_cell("""---
# Synthèse Personnelle — Mohammed SALHI

Cet atelier pratique m'a permis de maîtriser la chaîne complète du Traitement Automatique du Langage Naturel (NLP) en PyTorch, depuis la collecte brute jusqu'aux modèles génératifs modernes :

1. **Spécificités du NLP Arabe :** La richesse morphologique et les variations orthographiques de l'arabe imposent un prétraitement rigoureux. L'association d'une normalisation complète, de la lemmatisation légère (`ARLSTem2`) et de la racinisation (`ISRIStemmer`) réduit nettement la dimensionnalité du vocabulaire et améliore la capacité de généralisation des modèles.
2. **Comparaison des Architectures Récurrentes & Hyperparameter Tuning :**
   - L'utilisation du **Masked Mean Pooling** évite que les tokens de padding ne dégradent la représentation finale des séquences courtes.
   - Le **Bidirectional RNN** enrichit la représentation grâce au double contexte (gauche et droite), tandis que **GRU** et **LSTM** stabilisent l'apprentissage via leurs mécanismes de portes.
   - L'optimisation des hyperparamètres (`hidden_dim`, `embed_dim`, `lr`, `dropout`) sur un jeu de validation dédié s'avère essentielle pour tirer le meilleur parti de chaque architecture.
3. **Apport des Transformers et du Fine-Tuning GPT-2 :**
   - Contrairement aux réseaux récurrents limités par le calcul séquentiel, l'architecture **Transformer** exploite l'auto-attention multi-têtes pour modéliser efficacement le contexte global.
   - Le masquage propre des tokens de padding (`labels = -100`) et l'usage d'un scheduler linéaire avec warmup permettent une convergence rapide et propre de **GPT-2** sur un domaine spécialisé (cybersécurité), produisant des paragraphes cohérents et techniquement pertinents grâce au décodage par *Top-K* et *Nucleus Sampling (Top-P)*.
"""))

    nb["cells"] = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook généré avec succès : {output_path}")


if __name__ == "__main__":
    build_notebook()
