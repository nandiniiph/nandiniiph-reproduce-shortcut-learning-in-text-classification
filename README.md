# Reproduction Report
“Navigating the Shortcut Maze: Shortcut Learning in Text Classification with BERT”  
**Original Paper**  
Navigating the Shortcut Maze: A Comprehensive Analysis of Shortcut Learning in Text Classification by Language Models  
**Authors**  
Yuqing Zhou et al.  
**Link to Paper**  
https://arxiv.org/abs/2305.07467  
**Link to Original Code**  
https://github.com/yuqing-zhou/shortcut-learning-in-text-classification  
**Report by**  
Nandini Putri Hanifa Jannah  

## 1. Summary and Reproduction Goal
**Paper Summary**

This paper investigates shortcut learning in modern language models such as BERT. Shortcut learning occurs when a model relies on spurious patterns in the training data (e.g., the presence of specific words) rather than learning meaningful semantic representations.

The authors demonstrate that although models can achieve high accuracy on standard test sets, their performance drops significantly when shortcuts are removed. This reveals a lack of robustness and poor generalization to out-of-distribution (OOD) data.

**Reproduction Goal**
The goal of this reproduction is to validate the shortcut learning phenomenon reported in the paper by:
- Training a BERT-based text classifier on a shortcut-augmented dataset
- Evaluating the model on:
      - A Test set (shortcut present)
      - An Anti-Test set (shortcut removed)
      - An Original Test set
- Verifying that model performance drops substantially when shortcuts are removed
This reproduction was conducted in a CPU-only environment, unlike the original experiments which were primarily run on GPUs.

## 2. Environment Setup
**Hardware**
- Device: CPU-only
- Note: Training time is significantly longer compared to GPU-based experiments.

**Software**
- Python: 3.10+
- Key Libraries:
      - PyTorch
      - HuggingFace Transformers
      - Datasets
      - Evaluate
      - Weights & Biases (offline mode)

**Model Configuration**
- Model: bert-base-uncased
- Epochs: 3
- Batch size: 16
- Optimizer & scheduler: HuggingFace Trainer defaults
- Logging: W&B offline

## 3. Step-by-Step Reproduction
**1️⃣ Clone the Repository**
git clone https://github.com/yuqing-zhou/shortcut-learning-in-text-classification
cd shortcut-learning-in-text-classification

**2️⃣ Install Dependencies**
pip install torch transformers datasets evaluate wandb

**3️⃣ Dataset Configuration**
- Dataset: Yelp Reviews
- Shortcut type: occurrence
- Shortcut subtype: single-word
- Dataset preprocessing is handled automatically by the provided scripts

**4️⃣ Run Training**
python code/bert.py
When prompted by Weights & Biases:
(3) Don't visualize my results
The experiment runs entirely on CPU.

**5️⃣ Evaluation**
The trained model is evaluated on three different test sets:
- Test – shortcut present
- Anti-Test – shortcut removed
- Original Test – original, natural data

## 4. Results and Analysis
Experimental Results
| Test Type                            | Accuracy | Macro F1 |
|-------------------------------------|----------|----------|
| **Test (shortcut present)**         | 0.6446   | 0.6386   |
| **Anti-Test (shortcut removed)**    | 0.3950   | 0.3577   |
| **Original Test**                   | 0.5758   | 0.5695   |


**Analysis**
- Test Set
  The model achieves relatively high performance because it exploits shortcut cues introduced during training.
- Anti-Test Set
  Performance drops sharply (by approximately 25–30 points), indicating strong reliance on shortcut features rather than semantic understanding.
- Original Test Set
  Performance is lower than the shortcut-based test set, as the original data does not contain artificial shortcut signals.
These trends closely match the findings reported in the original paper, although absolute values are lower due to fewer training epochs and the use of a CPU-only environment.

## 5. Key Challenges and Insights
**Technical Challenges**
- CPU-only training resulted in significantly longer training times
- Minor configuration adjustments were required for local execution
- Weights & Biases logging was performed in offline mode

**Insights Gained**
- Shortcut learning is a robust and reproducible phenomenon
- High test accuracy does not necessarily indicate true language understanding
- Anti-test evaluation is critical for assessing model robustness
- The qualitative trends remain consistent despite hardware limitations

## 6. Conclusion
This reproduction successfully confirms the core claims of the original paper:
- BERT can achieve high accuracy by exploiting dataset shortcuts
- Removing shortcuts causes severe performance degradation
- Standard evaluation metrics may overestimate real model understanding
These findings highlight the importance of robustness-oriented evaluation in NLP research.
