"""
Atelier 3 - Deep Learning (Part 2)
Transformer Architecture & Text Generation with Fine-Tuned Pretrained GPT-2.

Université Abdelmalek Essaadi - FST Tanger (Département Génie Informatique)
Filière  : Cycle d'Ingénieur LSI - 3ème Année
Module   : Deep Learning (Pr. Lotfi ELAACHAK)
Étudiant : Mohammed SALHI
Tutoriel de référence : https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# Reproductibilité
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. Préparation du Dataset Personnalisé (Cybersécurité, Cloud & IA)
# =====================================================================
def build_custom_dataset(dataset_path="data/custom_dataset.txt"):
    """
    Crée un jeu de données personnalisé dédié à la cybersécurité moderne,
    à la cryptographie et à l'intelligence artificielle appliquée à la détection
    des menaces. Chaque document est terminé par le marqueur <|endoftext|>.
    """
    documents = [
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

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + " <|endoftext|>\n")

    print(f"  -> Dataset personnalisé créé : {dataset_path} ({len(documents)} paragraphes).")
    return documents


class CyberGPT2Dataset(Dataset):
    """
    Dataset PyTorch pour le fine-tuning causal de GPT-2 avec masquage propre
    des tokens de padding (-100) afin que la perte ne porte que sur les vrais tokens.
    """

    def __init__(self, texts, tokenizer, max_length=48):
        self.samples = []
        for text in texts:
            enc = tokenizer(
                text + " " + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =====================================================================
# 2. Génération de Paragraphes (Temperature, Top-K & Nucleus Top-P)
# =====================================================================
def generate_paragraph(
    model,
    tokenizer,
    prompt,
    max_new_tokens=60,
    temperature=0.75,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.2,
):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# =====================================================================
# 3. Fine-Tuning GPT-2 (AdamW + Warmup Linear Scheduler)
# =====================================================================
def fine_tune_gpt2(model, dataset, epochs=5, batch_size=2, lr=5e-5):
    """
    Fine-tune sélectif des 2 derniers blocs Transformer, de la normalisation
    finale (ln_f) et de la tête de modélisation du langage (lm_head).
    """
    for param in model.transformer.parameters():
        param.requires_grad = False
    for block in model.transformer.h[-2:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.transformer.ln_f.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    epoch_losses = []
    model.train()
    print(f"\n  Début du fine-tuning GPT-2 ({epochs} époques sur {DEVICE})...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"  Époque {epoch + 1}/{epochs} - Training Loss : {avg_loss:.4f}")

    print("  Fine-tuning terminé avec succès.")
    return epoch_losses


# =====================================================================
# Exécution Principale - Partie 2
# =====================================================================
def run_part2():
    print("=" * 72)
    print("LAB 3 - PARTIE 2 : TRANSFORMER & GÉNÉRATION GPT-2 (Mohammed SALHI)")
    print("=" * 72)

    print("\n[1/3] Chargement du modèle pré-entraîné GPT-2 et de son tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    base_prompt = "Modern cybersecurity architectures rely on"
    torch.manual_seed(SEED)
    print("\n[2/3] Génération AVANT fine-tuning :")
    print(f'  Amorce : "{base_prompt}"')
    before_text = generate_paragraph(model, tokenizer, base_prompt, max_new_tokens=55)
    print(f"  Résultat (GPT-2 de base) :\n  {before_text}\n")

    # Construction et chargement du dataset personnalisé
    docs = build_custom_dataset("data/custom_dataset.txt")
    dataset = CyberGPT2Dataset(docs, tokenizer, max_length=48)

    # Fine-tuning
    print("\n[3/3] Fine-Tuning de GPT-2 sur le dataset Cybersécurité & IA...")
    losses = fine_tune_gpt2(model, dataset, epochs=5, batch_size=2, lr=5e-5)

    # Génération de nouveaux paragraphes après fine-tuning
    prompts = [
        "Modern cybersecurity architectures rely on",
        "Zero-trust security and cryptographic protocols",
        "Artificial intelligence in threat detection",
    ]

    print("\n" + "=" * 72)
    print("GÉNÉRATION DE NOUVEAUX PARAGRAPHES APRÈS FINE-TUNING")
    print("=" * 72)
    generated_samples = []
    for idx, prompt in enumerate(prompts, 1):
        torch.manual_seed(SEED + idx)
        text_out = generate_paragraph(model, tokenizer, prompt, max_new_tokens=60)
        generated_samples.append((prompt, text_out))
        print(f"\n[Exemple {idx}]")
        print(f'Phrase d\'amorce : "{prompt}"')
        print(f"Paragraphe généré :\n{text_out}")
        print("-" * 60)

    return losses, before_text, generated_samples


if __name__ == "__main__":
    run_part2()
