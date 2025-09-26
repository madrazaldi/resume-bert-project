---
title: Resume Classifier
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Transformer-Based Resume Classification: Setup Guide
This guide provides instructions to set up and run the resume classification project on a local machine with an NVIDIA GPU.
## 🛠️ Setup and Installation

**1. Clone the repository:**

```bash
git clone https://github.com/madrazaldi/resume-bert-project.git
cd resume-bert-project
```

**2. Create and activate a Python virtual environment:**

```Bash
python -m venv .venv
source .venv/bin/activate
```
_(On Windows, use `.venv\Scripts\activate`)_

**3. Install PyTorch with CUDA support:** _(This command is for CUDA 12.1. Ensure your NVIDIA driver is installed and adjust the CUDA version in the URL if necessary.)_

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**4. Install project dependencies:**
```bash
pip install transformers datasets scikit-learn pandas matplotlib seaborn nltk accelerate protobuf
```


## 🚀 Usage

The project is designed to be run in two sequential steps.

**Step 1: Preprocess the Data** This script cleans the raw `Resume.csv`, performs exploratory data analysis, and generates the `train.csv`, `validation.csv`, and `test.csv` files required for training.

```
python eda_and_preprocessing.py
```

**Step 2: Run the Full Training and Evaluation Sweep** This is the main script. It will train and evaluate all six model configurations defined in the project. This process is computationally intensive and will take a significant amount of time to complete.

```Bash
python resume_classifier_full_sweep.py
```

After the script finishes, the results for each model—including the classification report and confusion matrix—will be saved in a corresponding subdirectory inside the `results/` folder.