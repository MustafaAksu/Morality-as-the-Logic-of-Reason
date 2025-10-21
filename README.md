# Morality as the Logic of Reason

**Authors:**  
Mustafa Aksu  
*with AI collaborators: Grok (xAI) and ChatGPT (OpenAI)*  

**Preprint:** [arXiv:25xx.xxxxx](https://arxiv.org/abs/25xx.xxxxx)  
**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)

---

## 📘 Overview

This repository contains the source files, figures, and simulation code accompanying the paper  
**“Morality as the Logic of Reason”** — an interdisciplinary study integrating cognitive science,  
game theory, information theory, and AI ethics.

The paper models moral reasoning as a *resonance phenomenon* in multi-agent systems,  
formally linking motivation, cooperation, and entropy minimization:

\[
E_c = \frac{H-A}{(1+k t)^n}, \qquad 
S^{\mathsf{R}} = -\sum_{i,j} r_{ij}\ln(r_{ij})
\]

- **Temporal dimension:** explains moral delay and procrastination through temporal discounting.  
- **Strategic dimension:** models cooperation using iterated-game dynamics (Tit-for-Tat, Generous-TFT).  
- **Relational dimension:** defines *relational entropy* \(S^{\mathsf{R}}\) as a measure of moral order.  
- **AI application:** proposes *Moral Memory*, *Meta-Learning*, and *Supervised Resonance* as design principles for ethical AI.

---

## 🧩 Repository Structure

```
Morality-as-the-Logic-of-Reason/
│
├── paper/
│   ├── morality_logic_reason.tex     # LaTeX source
│   ├── ipd_chart.pdf                 # Figure 1 – Cooperation vs. δ
│   ├── entropy_chart.pdf             # Figure 2 – Sᴿ vs. rᵢⱼ
│   ├── figures.tex                   # LaTeX figure inclusion file
│   └── LICENSE                       # CC-BY 4.0
│
├── code/
│   ├── gtft_simulation.ipynb         # Generous-TFT simulation (10-agent IPD)
│   ├── utils.py                      # Helper functions
│   └── data/                         # (optional) CSV results
│
├── README.md
└── .gitignore                        # excludes build artifacts
```

---

## 🔬 Reproducing Results

### Figures
1. **Iterated Prisoner’s Dilemma (IPD):**
   - Run `code/gtft_simulation.ipynb` to generate cooperation-rate data.
   - The resulting CSV can be rendered as `ipd_chart.pdf`.

2. **Relational Entropy Curve:**
   - Execute the entropy-calculation cell in the notebook  
     or `python utils.py --entropy-curve`  
     to regenerate `entropy_chart.pdf`.

### Building the Paper
```bash
cd paper
pdflatex morality_logic_reason.tex
pdflatex morality_logic_reason.tex
```
> For arXiv, upload the **source** (tex + figures). Do not upload PDF-only.

---

## 🧠 Citation
If you use this work, please cite:

```bibtex
@article{Aksu2025MoralityLogicReason,
  title   = {Morality as the Logic of Reason},
  author  = {Aksu, Mustafa and AI Collaborators (Grok and ChatGPT)},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {25xx.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

---

## 🤝 Collaboration & Discussion
This repository welcomes academic collaboration, replication studies, and interdisciplinary dialogue.  
Issues and pull requests may include:
- improvements to the simulation code,
- alternative moral-resonance metrics,
- translations or educational materials.

For correspondence: **your-email@example.com**

---

## 🌌 Statement of Intent
This research emerges from a collaboration between human and artificial intelligences,  
seeking to model morality as a measurable form of resonance — a bridge between reason,  
ethics, and universal order.

> *“Fear breeds isolation; trust amplifies collective resonance.”*
