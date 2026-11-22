"""
Génère les figures SVG modernes pour le rapport de l'Atelier 3 - Deep Learning.
Étudiant : Mohammed SALHI (LSI 3ème Année - FST Tanger)
"""

import os


def generate_svgs(gpt2_losses=None):
    os.makedirs("images", exist_ok=True)

    # 1. Graphique comparatif des 4 modèles récurrents (Partie 1)
    # Valeurs réelles issues de l'exécution de part1_nlp_sequence_models.py
    models_data = [
        ("Simple RNN", 0.5333, 0.5250, 0.2433),
        ("BiRNN", 0.4667, 0.4275, 0.2483),
        ("GRU", 0.5333, 0.5078, 0.2345),
        ("LSTM", 0.4333, 0.4206, 0.2367),
    ]

    base_y = 380.0
    scale = 325.0  # 0.80 -> 260px -> y=120
    groups_svg = []
    start_x = 140
    step_x = 165

    for i, (label, acc, f1, bleu) in enumerate(models_data):
        gx = start_x + i * step_x
        h_acc, h_f1, h_bleu = acc * scale, f1 * scale, bleu * scale
        y_acc, y_f1, y_bleu = base_y - h_acc, base_y - h_f1, base_y - h_bleu
        groups_svg.append(f"""  <g transform="translate({gx}, 0)">
    <rect x="0" y="{y_acc:.1f}" width="34" height="{h_acc:.1f}" rx="4" fill="#0f766e"/>
    <text x="17" y="{y_acc - 8:.1f}" font-size="11" font-weight="bold" fill="#115e59" text-anchor="middle">{acc:.3f}</text>

    <rect x="42" y="{y_f1:.1f}" width="34" height="{h_f1:.1f}" rx="4" fill="#2563eb"/>
    <text x="59" y="{y_f1 - 8:.1f}" font-size="11" font-weight="bold" fill="#1d4ed8" text-anchor="middle">{f1:.3f}</text>

    <rect x="84" y="{y_bleu:.1f}" width="34" height="{h_bleu:.1f}" rx="4" fill="#d97706"/>
    <text x="101" y="{y_bleu - 8:.1f}" font-size="11" font-weight="bold" fill="#b45309" text-anchor="middle">{bleu:.3f}</text>

    <text x="59" y="405" font-size="14" font-weight="600" fill="#0f172a" text-anchor="middle">{label}</text>
  </g>""")

    svg_comparison = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 480" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <text x="425" y="38" text-anchor="middle" font-size="21" font-weight="bold" fill="#0f172a">Lab 3 - Partie 1 : Comparaison des Modèles Séquentiels</text>
  <text x="425" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Classification de Pertinence en Arabe — Corpus Cybersécurité &amp; Cryptographie (146 textes)</text>

  <!-- Légende -->
  <rect x="235" y="82" width="16" height="14" rx="3" fill="#0f766e"/>
  <text x="258" y="94" font-size="13" fill="#334155" font-weight="600">Accuracy</text>

  <rect x="355" y="82" width="16" height="14" rx="3" fill="#2563eb"/>
  <text x="378" y="94" font-size="13" fill="#334155" font-weight="600">F1-Score (pondéré)</text>

  <rect x="535" y="82" width="16" height="14" rx="3" fill="#d97706"/>
  <text x="558" y="94" font-size="13" fill="#334155" font-weight="600">Score BLEU</text>

  <!-- Grille -->
  <line x1="100" y1="380" x2="780" y2="380" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="100" y1="315" x2="780" y2="315" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>
  <text x="86" y="320" font-size="12" fill="#64748b" text-anchor="end">0.20</text>

  <line x1="100" y1="250" x2="780" y2="250" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>
  <text x="86" y="255" font-size="12" fill="#64748b" text-anchor="end">0.40</text>

  <line x1="100" y1="185" x2="780" y2="185" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>
  <text x="86" y="190" font-size="12" fill="#64748b" text-anchor="end">0.60</text>

  <line x1="100" y1="120" x2="780" y2="120" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>
  <text x="86" y="125" font-size="12" fill="#64748b" text-anchor="end">0.80</text>

{chr(10).join(groups_svg)}

  <text x="425" y="452" text-anchor="middle" font-size="12" fill="#64748b">Étudiant : Mohammed SALHI — LSI 3ème Année — FST Tanger</text>
</svg>"""

    with open("images/models_comparison.svg", "w", encoding="utf-8") as f:
        f.write(svg_comparison)

    # 2. Courbe de perte GPT-2 (5 époques) — valeurs réelles de part2_gpt2_text_generation.py
    if gpt2_losses is None:
        gpt2_losses = [5.3453, 4.8093, 4.4675, 4.2392, 4.0298]

    min_v = max(0.0, min(gpt2_losses) - 0.35)
    max_v = max(gpt2_losses) + 0.35
    span = max(0.5, max_v - min_v)
    y_top, y_bot = 95.0, 315.0
    xs = [165 + i * 115 for i in range(len(gpt2_losses))]
    pts = []
    for x_pos, val in zip(xs, gpt2_losses):
        y_pos = y_bot - ((val - min_v) / span) * (y_bot - y_top)
        pts.append((x_pos, y_pos, val))

    polyline_str = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
    polygon_str = f"{polyline_str} {pts[-1][0]:.1f},{y_bot:.1f} {pts[0][0]:.1f},{y_bot:.1f}"

    grid_lines = []
    for step in range(4):
        frac = step / 3.0
        yv = y_bot - frac * (y_bot - y_top)
        lv = min_v + frac * span
        grid_lines.append(
            f'  <line x1="115" y1="{yv:.1f}" x2="670" y2="{yv:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>\n'
            f'  <text x="102" y="{yv + 4:.1f}" font-size="12" fill="#64748b" text-anchor="end">{lv:.2f}</text>'
        )

    dots_svg = []
    for idx, (x, y, val) in enumerate(pts, 1):
        color = "#0f766e" if idx == len(pts) else "#1d4ed8"
        dots_svg.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>\n'
            f'  <text x="{x:.1f}" y="{y - 14:.1f}" font-size="12" font-weight="bold" fill="{color}" text-anchor="middle">{val:.4f}</text>\n'
            f'  <text x="{x:.1f}" y="340" font-size="13" font-weight="600" fill="#334155" text-anchor="middle">Époque {idx}</text>'
        )

    svg_loss = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 405" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <text x="380" y="36" text-anchor="middle" font-size="20" font-weight="bold" fill="#0f172a">Lab 3 - Partie 2 : Convergence du Fine-Tuning GPT-2</text>
  <text x="380" y="60" text-anchor="middle" font-size="13" fill="#475569">Évolution de la perte (Cross-Entropy masquée) sur 5 époques — Dataset Cybersécurité &amp; IA</text>

  <line x1="115" y1="315" x2="670" y2="315" stroke="#cbd5e1" stroke-width="1.5"/>
{chr(10).join(grid_lines)}

  <polygon points="{polygon_str}" fill="#0f766e" fill-opacity="0.12"/>
  <polyline points="{polyline_str}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>

{chr(10).join(dots_svg)}

  <text x="380" y="382" text-anchor="middle" font-size="12" fill="#64748b">Optimiseur AdamW (lr=5e-5) + Warmup Linear Scheduler — Étudiant : Mohammed SALHI</text>
</svg>"""

    with open("images/gpt2_training_loss.svg", "w", encoding="utf-8") as f:
        f.write(svg_loss)

    # 3. Schéma du Pipeline NLP Arabe
    svg_pipeline = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 230" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
    </marker>
  </defs>
  <text x="480" y="32" text-anchor="middle" font-size="18.5" font-weight="bold" fill="#0f172a">Pipeline de Traitement NLP Arabe &amp; Classification Séquentielle</text>

  <g transform="translate(24, 62)">
    <rect width="152" height="96" rx="10" fill="#f0fdfa" stroke="#0f766e" stroke-width="2"/>
    <text x="76" y="32" text-anchor="middle" font-size="13" font-weight="bold" fill="#115e59">1. Web Scraping</text>
    <text x="76" y="54" text-anchor="middle" font-size="11" fill="#334155">BeautifulSoup</text>
    <text x="76" y="72" text-anchor="middle" font-size="10.5" fill="#64748b">Cybersécurité (AR)</text>
  </g>
  <line x1="178" y1="110" x2="210" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <g transform="translate(216, 62)">
    <rect width="152" height="96" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
    <text x="76" y="32" text-anchor="middle" font-size="13" font-weight="bold" fill="#1d4ed8">2. Normalisation</text>
    <text x="76" y="54" text-anchor="middle" font-size="11" fill="#334155">Tashkeel &amp; Tatweel</text>
    <text x="76" y="72" text-anchor="middle" font-size="10.5" fill="#64748b">Alif / Yaa / Hamza</text>
  </g>
  <line x1="370" y1="110" x2="402" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <g transform="translate(408, 62)">
    <rect width="152" height="96" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="76" y="32" text-anchor="middle" font-size="13" font-weight="bold" fill="#b45309">3. Morphologie</text>
    <text x="76" y="54" text-anchor="middle" font-size="11" fill="#334155">Stopwords NLTK</text>
    <text x="76" y="72" text-anchor="middle" font-size="10.5" fill="#64748b">ARLSTem2 + ISRI</text>
  </g>
  <line x1="562" y1="110" x2="594" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <g transform="translate(600, 62)">
    <rect width="152" height="96" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="2"/>
    <text x="76" y="32" text-anchor="middle" font-size="13" font-weight="bold" fill="#6d28d9">4. Discrétisation</text>
    <text x="76" y="54" text-anchor="middle" font-size="11" fill="#334155">Score 0-10 -&gt; 3 Classes</text>
    <text x="76" y="72" text-anchor="middle" font-size="10.5" fill="#64748b">Vocab &amp; Padding (35)</text>
  </g>
  <line x1="754" y1="110" x2="786" y2="110" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>

  <g transform="translate(792, 62)">
    <rect width="144" height="96" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>
    <text x="72" y="32" text-anchor="middle" font-size="13" font-weight="bold" fill="#b91c1c">5. Modèles + Tuning</text>
    <text x="72" y="54" text-anchor="middle" font-size="11" fill="#334155">RNN / BiRNN</text>
    <text x="72" y="72" text-anchor="middle" font-size="10.5" fill="#64748b">GRU / LSTM + BLEU</text>
  </g>

  <text x="480" y="202" text-anchor="middle" font-size="12" fill="#64748b">Étudiant : Mohammed SALHI — LSI 3ème Année — FST Tanger</text>
</svg>"""

    with open("images/part1_architecture.svg", "w", encoding="utf-8") as f:
        f.write(svg_pipeline)

    print("Schémas SVG générés avec succès dans images/.")


if __name__ == "__main__":
    generate_svgs()
