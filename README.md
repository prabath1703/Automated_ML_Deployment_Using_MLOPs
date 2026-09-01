# 🚀 Automated Machine Learning Deployment System Using MLOps

> **An end-to-end MLOps pipeline for automated machine-learning
> training, experiment tracking, model selection, containerized
> inference, CI/CD, deployment, and monitoring.**

```{=html}
<p align="center">
```
`<strong>`{=html}Machine Learning • MLOps • DevOps • Cloud • CI/CD •
Docker • FastAPI • MLflow • DVC • AWS`</strong>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Project Overview

This project demonstrates how a traditional machine-learning workflow
can be transformed into an **automated MLOps lifecycle**.

Instead of manually training a model, saving it, and deploying it
independently, the system brings together:

-   Data ingestion
-   Feature selection and preprocessing
-   Multiple ML models
-   Cross-validation
-   Model evaluation
-   Experiment tracking
-   Model artifact management
-   API-based inference
-   Docker containerization
-   GitHub Actions CI/CD
-   Cloud deployment
-   Prediction logging
-   Data-drift monitoring

The project uses the **Telco Customer Churn dataset** and trains
classification models to predict whether a customer is likely to churn.

------------------------------------------------------------------------

# 🎯 Project Objective

The main objective is to build a reproducible ML system in which a code
change can move through an engineering pipeline:

``` text
Code / Data Change
       ↓
     GitHub
       ↓
 GitHub Actions
       ↓
 Testing / Training
       ↓
 Model Evaluation
       ↓
 Best Model Selection
       ↓
 Docker Image
       ↓
 FastAPI Inference Service
       ↓
 Cloud Deployment
       ↓
 Prediction Logging
       ↓
 Data Drift Monitoring
```

This project focuses not only on **machine-learning accuracy**, but also
on the engineering practices required to make an ML application
reproducible, deployable, and maintainable.

------------------------------------------------------------------------

# 🧠 Machine Learning Problem

## Customer Churn Prediction

The system predicts whether a telecom customer is likely to churn.

### Input Features

  Feature            Description
  ------------------ -----------------------------------------------------------
  `tenure`           Number of months the customer has stayed with the company
  `MonthlyCharges`   Customer's monthly service charge
  `TotalCharges`     Total amount charged to the customer

### Models

The project evaluates multiple classification algorithms:

  Model                 Purpose
  --------------------- ----------------------------------------------
  Logistic Regression   Interpretable linear classification baseline
  Random Forest         Non-linear ensemble classification model

### Validation

The training workflow uses:

-   **Stratified K-Fold Cross-Validation**
-   **5 folds**
-   Classification evaluation metrics

### Evaluation Metrics

  Metric      Meaning
  ----------- -----------------------------------------------------
  Accuracy    Percentage of total predictions that are correct
  Precision   How many predicted churners were actually churners
  Recall      How many actual churners were successfully detected
  F1 Score    Balance between precision and recall

> **Important:** Accuracy alone is not sufficient for churn prediction
> because the class distribution can make accuracy look better than the
> actual minority-class performance. Precision, recall, and F1 are
> therefore tracked as well.

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
                         ┌──────────────────────┐
                         │       Developer      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       GitHub         │
                         │  Source + Versioning │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   GitHub Actions     │
                         │      CI/CD           │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Training / Evaluation│
                         │  + Automated Checks  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       MLflow         │
                         │ Tracking / Registry  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Best ML Model      │
                         │   churn_model.joblib │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Docker         │
                         │ Containerized API    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │    REST Inference    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Cloud Runtime     │
                         │      AWS EC2         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Prediction Logging   │
                         │ + Drift Monitoring   │
                         └──────────────────────┘
```

------------------------------------------------------------------------

# 📂 Project Structure

``` text
MLOPS-Deployment/
│
├── .dvc/                       # DVC internal metadata
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD workflow
│
├── data/
│   └── raw/                    # Raw ML dataset
│
├── docker/                     # Docker-related deployment files
│
├── logs/                       # Application / prediction logs
│
├── mlruns/                     # MLflow experiment tracking artifacts
│
├── models/
│   └── churn_model.joblib      # Trained ML model artifact
│
├── monitoring/                 # Monitoring and drift-analysis components
│
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application entry point
│   │   ├── routes.py           # API routes/endpoints
│   │   └── utils.py            # API helper functions
│   │
│   └── train.py                # Model training and evaluation pipeline
│
├── tests/
│   └── test_api.py             # API tests
│
├── .dvcignore                  # Files ignored by DVC
├── .gitignore                  # Files ignored by Git
├── dvc.lock                    # DVC pipeline lock information
├── dvc.yaml                    # DVC pipeline configuration
├── mlflow.db                   # Local MLflow backend database
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

------------------------------------------------------------------------

# 🛠️ Technology Stack

  --------------------------------------------------------------------------
  Category                Technology              Role in Project
  ----------------------- ----------------------- --------------------------
  Programming             Python                  Main development language

  ML                      Scikit-learn            Model training and
                                                  evaluation

  Data                    Pandas                  Data loading and
                                                  manipulation

  Numerical Computing     NumPy                   Numerical operations

  Model Serialization     Joblib                  Saving/loading trained
                                                  models

  Experiment Tracking     MLflow                  Track experiments,
                                                  metrics, parameters and
                                                  models

  Data Versioning         DVC                     Data/pipeline versioning
                                                  structure

  API                     FastAPI                 Serve ML predictions
                                                  through REST APIs

  ASGI Server             Uvicorn                 Run the FastAPI
                                                  application

  Testing                 Pytest                  Automated API/software
                                                  tests

  Containerization        Docker                  Package application and
                                                  dependencies

  CI/CD                   GitHub Actions          Automate
                                                  testing/build/deployment
                                                  workflow

  Cloud                   AWS EC2                 Cloud runtime for the API
                                                  deployment

  Monitoring              Evidently AI            Offline data-drift
                                                  analysis

  Version Control         Git + GitHub            Source-code versioning and
                                                  collaboration
  --------------------------------------------------------------------------

------------------------------------------------------------------------

# 🔍 What Each Tool Does

## 🐍 Python

Python is the primary language used throughout the system.

It connects the ML, API, automation, monitoring, and deployment
components into one workflow.

------------------------------------------------------------------------

## 📊 Pandas

Used for:

-   Reading datasets
-   Cleaning data
-   Selecting features
-   Creating DataFrames
-   Preparing model inputs

Example:

``` python
import pandas as pd

data = pd.read_csv("data/raw/data.csv")
```

------------------------------------------------------------------------

## 🔢 NumPy

Provides numerical and array-based operations used by the ML ecosystem.

------------------------------------------------------------------------

## 🤖 Scikit-learn

Scikit-learn provides the machine-learning algorithms and evaluation
utilities.

The project uses:

-   Logistic Regression
-   Random Forest
-   StratifiedKFold
-   Accuracy
-   Precision
-   Recall
-   F1 Score

------------------------------------------------------------------------

## 📦 Joblib

Joblib serializes the trained model so it can be loaded later without
retraining.

The resulting artifact is:

``` text
models/churn_model.joblib
```

The API loads this artifact during inference.

------------------------------------------------------------------------

# 📈 MLflow

MLflow is used for **ML experiment tracking and model lifecycle
management**.

Instead of simply training a model and writing down its accuracy
manually, MLflow allows experiments to be recorded systematically.

### MLflow tracks things such as:

-   Model parameters
-   Evaluation metrics
-   Experiment runs
-   Model artifacts
-   Model versions / registry information

Conceptually:

``` text
Training Run
    │
    ├── Parameters
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1 Score
    └── Model Artifact
```

The local project contains:

``` text
mlruns/
mlflow.db
```

which support the local MLflow setup.

------------------------------------------------------------------------

# 🗃️ DVC

**DVC --- Data Version Control** --- brings version-control concepts to
datasets and ML pipelines.

Git is excellent for source code, while DVC can be used to manage large
datasets and reproducible ML pipeline stages.

This project includes:

``` text
.dvc/
.dvcignore
dvc.yaml
dvc.lock
```

### DVC's role

``` text
Git
 ↓
Code Versioning

DVC
 ↓
Data / Pipeline Versioning
```

> The current project includes the DVC configuration and pipeline
> structure. A fully configured remote DVC storage backend is a future
> improvement.

------------------------------------------------------------------------

# ⚡ FastAPI

FastAPI provides the **REST API layer** for model inference.

The API acts as the bridge between a client and the trained ML model.

``` text
Client
  ↓
HTTP Request
  ↓
FastAPI
  ↓
Input Processing
  ↓
ML Model
  ↓
Prediction
  ↓
JSON Response
```

Relevant files:

``` text
src/api/app.py
src/api/routes.py
src/api/utils.py
```

------------------------------------------------------------------------

# 🚀 Uvicorn

Uvicorn is the ASGI server used to run the FastAPI application.

Conceptually:

``` text
Uvicorn
   ↓
FastAPI
   ↓
ML Inference
```

------------------------------------------------------------------------

# 🐳 Docker

Docker packages the application together with its dependencies into a
reproducible container.

Without containerization:

``` text
Developer Machine
Python Version
Libraries
OS Dependencies
```

can differ from:

``` text
Server
Python Version
Libraries
OS Dependencies
```

Docker reduces this environment mismatch.

### Container workflow

``` text
Application
    +
Dependencies
    +
Runtime Configuration
        ↓
     Docker Image
        ↓
   Docker Container
```

------------------------------------------------------------------------

# 🔄 GitHub Actions

GitHub Actions automates the software/ML deployment workflow.

The workflow is defined under:

``` text
.github/workflows/deploy.yml
```

A typical automated flow is:

``` text
git push
   ↓
GitHub Actions starts
   ↓
Install dependencies
   ↓
Run checks/tests
   ↓
Train/evaluate
   ↓
Build deployment artifact/container
   ↓
Deploy
```

This reduces the need for manually repeating deployment commands after
every change.

------------------------------------------------------------------------

# ☁️ AWS EC2

AWS EC2 provides a virtual server in the cloud.

The project originally used an EC2-based deployment for hosting the
FastAPI service.

Conceptually:

``` text
Docker Container
       ↓
AWS EC2
       ↓
Public API
```

> **Note:** The AWS free-tier allocation used during development has now
> been exhausted, so the EC2 deployment should be considered part of the
> project's demonstrated cloud architecture rather than a permanently
> running free service.

------------------------------------------------------------------------

# 📡 API Layer

The FastAPI service is responsible for:

1.  Receiving prediction requests
2.  Validating input
3.  Loading/using the trained model
4.  Performing inference
5.  Returning the prediction
6.  Supporting application-level logging

API source:

``` text
src/api/
├── app.py
├── routes.py
└── utils.py
```

------------------------------------------------------------------------

# 📊 Monitoring

Machine-learning systems need monitoring because production data can
change over time.

This project includes a monitoring workflow based on
prediction/reference data.

Conceptually:

``` text
Reference Data
      │
      │
      ├──────────────┐
      │              │
      ▼              ▼
Reference       New Prediction Data
      │              │
      └──────┬───────┘
             ▼
       Drift Analysis
             │
             ▼
      Monitoring Report
```

The monitoring layer is designed to compare production/prediction data
against reference data and identify potential data-distribution changes.

**Evidently AI** is used for offline drift analysis.

------------------------------------------------------------------------

# 🧪 Testing

Tests are stored in:

``` text
tests/
└── test_api.py
```

Testing is important because an ML application is still a software
system.

The test layer helps validate that API behavior remains correct as the
project changes.

Run:

``` bash
pytest
```

------------------------------------------------------------------------

# 🔁 Complete MLOps Lifecycle

The complete lifecycle can be understood as:

``` text
                ┌──────────────┐
                │     DATA     │
                └──────┬───────┘
                       ↓
                Data Preparation
                       ↓
                Feature Selection
                       ↓
                Model Training
                       ↓
              Cross Validation
                       ↓
                Model Evaluation
                       ↓
                Best Model
                       ↓
             MLflow Experiment Log
                       ↓
                Model Artifact
                       ↓
                 FastAPI API
                       ↓
                   Docker
                       ↓
                CI/CD Pipeline
                       ↓
                 Cloud Deploy
                       ↓
              Prediction Logging
                       ↓
               Drift Monitoring
                       │
                       └───────────────┐
                                       ↓
                                Model Improvement
```

------------------------------------------------------------------------

# 📈 Model Results

The project evaluates Logistic Regression and Random Forest using
multiple classification metrics.

Example recorded results from the project:

  Model                   Accuracy   F1 Score
  --------------------- ---------- ----------
  Logistic Regression     \~78.73%   \~52.64%
  Random Forest           \~78.63%   \~52.23%

Additional metrics such as precision and recall are also evaluated
during the training workflow.

> Results can vary slightly depending on the exact dataset version,
> preprocessing, random state, and training environment.

------------------------------------------------------------------------

# 💻 Local Setup

## 1. Clone the Repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MLOPS-Deployment
```

------------------------------------------------------------------------

## 2. Create a Virtual Environment

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🧠 Train the Model

From the project root, run:

``` bash
python src/train.py
```

The training pipeline is responsible for the model-training/evaluation
workflow.

The trained model artifact is expected under:

``` text
models/churn_model.joblib
```

------------------------------------------------------------------------

# 🌐 Run the FastAPI Application

The API entry point is:

``` text
src/api/app.py
```

Run the application using Uvicorn.

Depending on how the application is configured, a typical command is:

``` bash
uvicorn src.api.app:app --reload
```

The API will normally be available at:

``` text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation when enabled:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

# 🐳 Run With Docker

Build the Docker image using the project's Docker configuration.

A typical workflow is:

``` bash
docker build -t mlops-churn-api .
```

Then run:

``` bash
docker run -p 8000:8000 mlops-churn-api
```

> Use the project's actual Dockerfile path/configuration if your Docker
> setup uses a custom location or command.

------------------------------------------------------------------------

# 📊 MLflow

To inspect experiment tracking locally, start the MLflow UI using the
project's configured backend/artifact setup.

A typical local command is:

``` bash
mlflow ui
```

Then open:

``` text
http://127.0.0.1:5000
```

> If the project's MLflow configuration specifies a custom backend URI
> or artifact location, use that configuration rather than replacing it
> with a default command.

------------------------------------------------------------------------

# 🧪 Run Tests

``` bash
pytest
```

------------------------------------------------------------------------

# 📱 Public Demo / Streamlit UI

A Streamlit frontend can be added as a **presentation/demo layer** on
top of the existing ML system.

The intended architecture is:

``` text
                   Public User
                       │
                       ▼
                Streamlit UI
                       │
                       ▼
                  Prediction
                       │
                       ▼
              churn_model.joblib
```

This is useful for portfolio demonstrations because a recruiter can
interact with the ML model through a browser without needing to
understand the underlying API.

> The Streamlit interface is a separate presentation layer and does not
> replace the existing FastAPI/MLOps architecture.

------------------------------------------------------------------------

# 🔐 Configuration & Security

Before publishing the repository, make sure that you **never commit**:

-   AWS access keys
-   AWS secret keys
-   GitHub tokens
-   API keys
-   Passwords
-   `.env` files containing secrets
-   Private credentials

Use environment variables or GitHub Actions Secrets for sensitive
configuration.

Example:

``` text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

should never be hard-coded into source files.

------------------------------------------------------------------------

# 🔄 Development Workflow

A recommended developer workflow for this project is:

``` text
1. Modify code
      ↓
2. Run tests locally
      ↓
3. Train/evaluate if ML code changed
      ↓
4. Inspect MLflow results
      ↓
5. Test FastAPI
      ↓
6. Build/test Docker image
      ↓
7. git add
      ↓
8. git commit
      ↓
9. git push
      ↓
10. GitHub Actions
      ↓
11. Automated CI/CD
```

------------------------------------------------------------------------

# 🧩 Engineering Responsibilities by Layer

  Layer              Responsibility                Main Tools
  ------------------ ----------------------------- ----------------
  Data Layer         Store and prepare ML data     Pandas, DVC
  ML Layer           Train and evaluate models     Scikit-learn
  Experiment Layer   Track ML experiments          MLflow
  Artifact Layer     Store serialized model        Joblib
  API Layer          Expose inference service      FastAPI
  Server Layer       Serve the API                 Uvicorn
  Test Layer         Validate software behavior    Pytest
  Container Layer    Package application           Docker
  CI/CD Layer        Automate workflow             GitHub Actions
  Cloud Layer        Host service                  AWS EC2
  Monitoring Layer   Analyze data drift            Evidently AI
  Version Control    Track source changes          Git, GitHub
  UI Layer           Browser-based demonstration   Streamlit

------------------------------------------------------------------------

# ⭐ Key Engineering Concepts Demonstrated

This project demonstrates practical exposure to:

### Machine Learning

-   Supervised learning
-   Binary classification
-   Feature selection
-   Cross-validation
-   Model comparison
-   Classification metrics

### MLOps

-   Experiment tracking
-   Model artifacts
-   Model lifecycle concepts
-   Data/pipeline versioning
-   Reproducibility
-   Monitoring

### DevOps

-   Git
-   GitHub
-   CI/CD
-   Automated testing
-   Docker
-   Deployment automation

### Backend Engineering

-   REST APIs
-   FastAPI
-   Uvicorn
-   Request/response handling
-   API testing

### Cloud Engineering

-   AWS EC2
-   Linux server deployment
-   Containerized cloud workloads

------------------------------------------------------------------------

# ⚠️ Current Limitations

This project is intentionally documented with its current limitations
rather than hiding them.

  Area                             Current Status
  -------------------------------- ----------------------------------------------------------
  DVC Remote Storage               Not fully configured
  Terraform / IaC                  Not implemented
  Kubernetes                       Not implemented
  AWS SageMaker                    Not implemented
  Kubeflow                         Not implemented
  Airflow                          Not implemented
  Prometheus / Grafana             Not implemented
  Automatic Model Retraining       Not implemented
  Production-Scale Observability   Limited
  AWS EC2                          Free-tier development deployment; not permanently hosted

These limitations also define clear opportunities for future engineering
improvements.

------------------------------------------------------------------------

# 🚀 Future Improvements

The project can be extended toward a production-grade MLOps platform by
adding:

-   [ ] Terraform for Infrastructure as Code
-   [ ] AWS S3 for data/model artifacts
-   [ ] SageMaker integration
-   [ ] Kubernetes deployment
-   [ ] Helm charts
-   [ ] GitOps with Argo CD
-   [ ] Prometheus metrics
-   [ ] Grafana dashboards
-   [ ] Centralized logging
-   [ ] Automated model retraining
-   [ ] Model performance monitoring
-   [ ] DVC remote storage
-   [ ] Feature store
-   [ ] Airflow/Kubeflow orchestration
-   [ ] Authentication and authorization
-   [ ] Production-grade API gateway
-   [ ] Scalable cloud deployment

------------------------------------------------------------------------

# 🎓 What This Project Demonstrates

This is more than a machine-learning notebook.

It demonstrates how an ML model can move through an engineering
lifecycle:

``` text
Experiment
   ↓
Training
   ↓
Evaluation
   ↓
Tracking
   ↓
Packaging
   ↓
API
   ↓
Container
   ↓
CI/CD
   ↓
Cloud
   ↓
Monitoring
```

The main learning outcome is understanding that **deploying an ML model
is only one part of MLOps**. A reliable ML system also needs version
control, reproducibility, testing, automation, monitoring, and
maintainability.

------------------------------------------------------------------------

# 👨‍💻 Author

**Damarla Sai Prabath**

Final Year B.Tech --- Computer Science & Engineering (AI & ML)

Interested in:

-   MLOps
-   DevOps
-   Cloud Engineering
-   Machine Learning
-   Backend Engineering
-   AI Systems

### Connect

-   GitHub: `YOUR_GITHUB_URL`
-   LinkedIn: `YOUR_LINKEDIN_URL`

------------------------------------------------------------------------

# 📜 License

Add the project's license here if you decide to publish one.

Example:

``` text
License: MIT
Author: Damarla Sai Prabath
```

------------------------------------------------------------------------

# ⭐ If You Find This Project Useful

If this project helps you understand MLOps, feel free to ⭐ star the
repository and explore the implementation.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<strong>`{=html}Built to demonstrate the complete journey from Machine
Learning experimentation to deployable MLOps
engineering.`</strong>`{=html}
```{=html}
</p>
```
