# Reliable-and-Trustworthy-AI_Assignment_1
## Adversarial Attack: FGSM & PGD

This repository contains a PyTorch implementation of adversarial attacks, specifically **FGSM** and **PGD**, on **MNIST** and **CIFAR-10** datasets.

The project evaluates the robustness of simple CNN against both **Targeted** and **Untargeted** adversarial perturbations.

### 🚀 Key Features

* **Models**: Simple CNN (2~3 conv layers) architectures for MNIST and CIFAR-10.
    * Accuracy: 99.18% on MNIST, 81.18% on CIFAR-10
* **Attacks**:
    * **FGSM** (Targeted/Untargeted)
    * **PGD** (Targeted/Untargeted)
* **Auto Evaluation**: Calculates Attack Success Rate over 200 test samples.
* **Visualization**: Generates side-by-side comparisons of original images, adversarial examples, and perturbations.

## 📁 Project Structure

```text
.
├── test.py              # Main script for training, attacking, and evaluation
├── results/             # Directory where visualization PNGs are saved
├── log.txt              # Output logs from the experiment execution
├── report.pdf           # Analysis report
├── requirements.txt
└── .gitignore
```

## 💻 How to Run

Simply run the test.py. This will automatically
1. Train the CNN models for MNIST and CIFAR-10.
2. Run 4 types of attacks (FGSM/PGD, Targeted/Untargeted).
3. Print success rates in the console.
4. Save visualization plots in the results/ folder.

## 📊 Experimental Results
| Dataset | Attack Type     | Success Rate |
| :--- |:----------------|:-------------|
| **MNIST** | FGSM Targeted   | 19.50%       |
| **MNIST** | FGSM Untargeted | 87.00%       |
| **MNIST** | PGD Targeted    | 100%         |
| **MNIST** | PGD Untargeted  | 100%         |
| **CIFAR-10** | FGSM Targeted   | 57.50%       |
| **CIFAR-10** | FGSM Untargeted | 100%         |
| **CIFAR-10** | PGD Targeted    | 100%         |
| **CIFAR-10** | PGD Untargeted  | 100%         |