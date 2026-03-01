"""
CUBE dataset utilities.

Available datasets for VLM-RL experiments:
  - MathVista      (AI4Math/MathVista)
  - MMStar         (lmms-lab/MMStar)
  - ChartQA        (HuggingFaceM4/ChartQA)
  - MMBench        (lmms-lab/MMBench)
  - ScienceQA      (HuggingFaceM4/ScienceQA)
  - MMMU-Pro       (lmms-lab/MMMU_Pro)
  - VQAv2          (HuggingFaceM4/VQAv2)
  - OK-VQA         (Multimodal-Fatima/OK-VQA_train)

Usage:
    python datasets/download.py --list
    python datasets/download.py --dataset mathvista --split testmini
"""
