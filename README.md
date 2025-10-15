---
title: Resume Classifier
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# End-to-End Resume Classifier

This repository contains the full pipeline for an end-to-end machine learning project. It includes scripts to preprocess raw resume data, train and evaluate multiple Transformer models, and deploy the best-performing model as a full-stack web application using FastAPI and Docker.

## 📋 Project Workflow

The project is designed to be run sequentially. The complete workflow is as follows:

1. **Machine Learning Pipeline:** Preprocess the raw dataset and run the full experimental sweep to train all six model configurations.
    
2. **Local Web App Development:** Run the web application locally for testing, using the models trained in the previous step.
    
3. **Deployment with Docker:** Package the final application into a Docker container for easy and reproducible deployment.
    

## 🚀 Step-by-Step Guide

### **1. Setup and Installation**

- **Clone the repository:**
    
    ```
    git clone https://github.com/madrazaldi/resume-bert-project
    cd https://github.com/madrazaldi/resume-bert-project
    ```
    
- **Create and activate a Python virtual environment:**
    
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
    
- **Install PyTorch with CUDA support (Required for Training):**
    
    ```
    pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
    
- **Install All Project Dependencies:**
    
    ```
    # Install web app dependencies
    pip install -r app/requirements.txt
    # Install ML pipeline dependencies
    pip install scikit-learn pandas matplotlib seaborn nltk accelerate
    ```
    
- **Download the Dataset:** Place the `Resume.csv` file in the root of the project directory.
    

### **2. Part 1: Machine Learning Pipeline**

This is the first and most critical part of the project. You must run these scripts to generate the cleaned data and the trained models that the web application depends on.

- **Preprocess the Data:** This script cleans `Resume.csv` and creates the `train.csv`, `validation.csv`, and `test.csv` files.
    
    ```
    python eda_and_preprocessing.py
    ```
    
- **Run the Full Training and Evaluation Sweep:** This script trains all six model configurations and saves the results (including classification reports and confusion matrices) to the `results/` directory. This is a long, computationally intensive process.
    
    ```
    python resume_classifier_full_sweep.py
    ```
    

### **3. Part 2: Local Web App Development**

After the models have been trained, you can run the web application locally for testing.

- **Start the backend server:** From the project root directory, run:
    
    ```
    uvicorn app.api:app --reload
    ```
    
- Access the web app in your browser at `http://localhost:8000`.
    

### **4. Part 3: Deployment with Docker**

This is the final step for creating a portable, self-contained version of your application.

- **Prerequisites:** Ensure **Docker** and **Docker Compose** are installed and running on your machine.
    
- **Build and Run the Container:** From the project root directory, run:
    
    ```
    docker-compose up --build
    ```