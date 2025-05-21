<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

# Group members
| STT    | MSSV          | Họ và Tên              |  Email                  |
| ------ |:-------------:| ----------------------:|-------------------------:
| 1      | 22520273      | Nguyễn Viết Đức        |22520273@gm.uit.edu.vn   |
| 2      | 22520459      | Đoàn Văn Hoàng         |22520459@gm.uit.edu.vn   |
| 3      | 22520862      | Huỳnh Nhật Minh        |22520862@gm.uit.edu.vn   |

# MLOps Project: YooChoose Recommendation System

This project implements an end-to-end MLOps pipeline for a recommendation system using the YooChoose dataset. It leverages modern MLOps practices to manage data, train models, evaluate performance, and automate workflows.

## Table of Contents
[Overview](#overview)
[Project Structure](#project-structure)
[Prerequisites](#prerequisites)
[Setup Instructions](#setup-instructions)

  - [Local Machine Setup](#local-machine-setup)
  - [Server Setup](#server-setup)
  - [DVC Remote Storage](#dvc-remote-storage)
  - [Airflow Automation](#airflow-automation)
[Pipeline Details](#pipeline-details)
[Usage](#usage)
[Troubleshooting](#troubleshooting)
[Contributing](#contributing)
[License](#license)


## Overview

This project builds a recommendation system using the [YooChoose](https://darel13712.github.io/rs_datasets/Datasets/yoochoose/) dataset, which contains user click and purchase data. The pipeline includes data preprocessing, model training, evaluation, and deployment, with the following key components:

- *Data Versioning*: Managed with DVC, with remote storage on a server.
- *Workflow Automation*: Automated using Apache Airflow.
- *Experiment Tracking*: Tracked using MLFlow.
- *Code Quality*: Enforced with pre-commit hooks (black, flake8, and custom commit message validation).
- *Environment*: Uses RAPIDS (cudf, cupy) for GPU-accelerated data processing and PyTorch for model training.

The pipeline is designed to run either on a local machine (via SSH from the server) or directly on the server.

## Project Structure

```text
/mlops_cs317/
├── .git/                    # Git repository for version control
├── .dvc/                    # DVC configuration and cache for data version control
├── .gitignore               # Specifies files and directories to ignore in Git
├── .pre-commit-config.yaml  # Configuration for pre-commit hooks
├── configs/
│   ├── config.yaml          # General configuration (e.g., hyperparameters, paths)
│   └── dvc.yaml             # DVC pipeline definitions
├── data/
│   ├── raw/                 # Raw, unprocessed data files
│   │   ├── yoochoose-clicks.dat
│   │   ├── yoochoose-buys.dat
│   │   └── dataset-README.txt
│   ├── processed/           # Processed data ready for training/evaluation
│   │   ├── clicks_train.parquet
│   │   ├── clicks_test.parquet
│   │   ├── item_encoder.pkl
│   │   ├── train_sessions.pkl
│   │   └── test_sessions.pkl
│   └── test/                # Test data for model evaluation
│       └── yoochoose-test.dat
├── scripts/
│   ├── check_commit_message.sh  # Validates commit message format
│   ├── preprocess.py        # Script for data preprocessing
│   ├── train.py             # Script for model training
│   ├── evaluate.py          # Script for model evaluation
│   ├── test_commit.py       # Tests for pre-commit hooks
│   └── utils/               # Utility scripts and helpers
│       ├── _init_.py
│       ├── data_loader.py
│       └── metrics.py
├── models/
│   └── model.pth            # Trained model checkpoint
├── metrics/
│   ├── metrics.json         # Training performance metrics
│   └── eval_metrics.json    # Evaluation performance metrics
├── tests/                   # Unit tests for scripts and utilities
├── notebooks/
│   └── MLOps.ipynb          # Jupyter notebook for exploratory analysis
└── requirements.txt         # Python dependencies for the project
```
### Server Structure (mlops@192.168.28.39:/home/mlops_CS317/)

```text
/home/mlops_CS317/
├── dvcstore/                # DVC remote storage for data files
│   └── files/               # Versioned data files
│       ├── yoochoose-clicks.dat
│       ├── yoochoose-test.dat
│       ├── clicks_train.parquet
│       ├── clicks_test.parquet
│       └── ...              # Other DVC-managed files
├── project/                 # Optional: Local project copy for server pipelines
│   └── [same as local structure]
├── mlruns/                  # MLflow experiment tracking directory
├── airflow/                 # Airflow setup for pipeline orchestration
│   ├── dags/
│   │   └── mlops_pipeline.py  # Airflow DAG for MLOps pipeline
│   ├── logs/                # Airflow logs
│   ├── airflow.db           # Airflow metadata database
│   └── airflow.cfg          # Airflow configuration file
├── scripts/
│   ├── start_mlflow.sh      # Script to launch MLflow server
│   └── start_airflow.sh     # Script to launch Airflow services
├── .ssh/                    # SSH configuration for secure access
└── logs/                    # General server logs
```
## Prerequisites

- *Local Machine*:
  - Ubuntu on WSL (Windows Subsystem for Linux)
  - Python 3.10
  - Conda (for environment management)
  - Git
  - DVC
  - SSH client

- *Server*:
  - Ubuntu
  - Python 3.10
  - Conda
  - Apache Airflow 2.10.5
  - MLFlow
  - SSH server (for DVC remote storage)

## Setup Instructions

### Local Machine Setup

1. *Clone the Repository*
   
   git clone <repository-url>
   cd mlops_cs317
   

2. *Set Up the Conda Environment*
   <!--  Create environment with GPU -->
   <!-- Install conda through cmd: 
        -Step1: curl -o miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
        -Step2: start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /AddToPath=1 /S /D=%UserProfile%\Miniconda3
        -Step3: conda --version (version: conda 25.*.*)(to check whether conda had installed or not) 
    -->
   conda create -n mlops_env python=3.10
   conda activate mlops_env
   pip install -r requirements.txt


3. *Install DVC*
   
   pip install "dvc[ssh]"
   

4. *Set Up Pre-commit Hooks*
   
   pre-commit install
   pre-commit install --hook-type commit-msg
   

5. *Initialize DVC*
   
   dvc init
   

### Server Setup

1. *SSH into the Server*
   
   ssh mlops@192.168.28.39
   

2. *Install Conda (if not installed)*
   
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   

3. *Set Up the Conda Environment*
   
   conda create -n rapids-test python=3.10
   conda activate rapids-test
   pip install -r requirements.txt
   

4. *Install Airflow*
   
   pip install apache-airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   

5. *Start Airflow Services*
   
   nohup airflow webserver --port 8080 &
   nohup airflow scheduler &
   

6. *Start MLFlow Server*
   
   nohup mlflow ui --backend-store-uri file:///home/mlops/mlruns --host 0.0.0.0 --port 5000 &
   

### DVC Remote Storage

1. *Create the DVC Storage Directory on the Server*
   
   ssh mlops@192.168.28.39
   mkdir -p /home/mlops/dvcstore
   chmod 755 /home/mlops/dvcstore
   exit
   

2. *Set Up SSH Key-based Authentication*
   - On the local machine:
     
     ssh-keygen -t rsa -b 4096
     ssh-copy-id mlops@192.168.28.39
     

3. *Configure the DVC Remote*
   - On the local machine:
     
     dvc remote add -d myremote ssh://mlops@192.168.28.39/home/mlops/dvcstore
     

4. *Push Data to the Remote*
   
   dvc add data/raw/* data/test/* data/processed/*
   git add data/raw/*.dvc data/test/*.dvc data/processed/*.dvc .dvc/config
   git commit -m "setup: configure dvc remote and add data files"
   dvc push
   

### Airflow Automation

1. *Test the DAG*
   
   airflow dags test mlops_pipeline 2025-04-12T00:00:00
   

2. *Watch the DAG*
   - Access the Airflow UI at http://192.168.28.39:8080, log in (username: admin, password: admin), and trigger the DAG.

## Pipeline Details

The pipeline is defined in configs/dvc.yaml and consists of three stages:

1. *Preprocess*:
   - *Command*: python scripts/preprocess.py
   - *Inputs*: data/raw/yoochoose-clicks.dat, data/test/yoochoose-test.dat
   - *Outputs*: data/processed/clicks_train.parquet, data/processed/clicks_test.parquet, data/processed/item_encoder.pkl, data/processed/train_sessions.pkl, data/processed/test_sessions.pkl
   - *Description*: Processes raw click data into parquet files and session sequences, using RAPIDS for GPU acceleration.

2. *Train*:
   - *Command*: python scripts/train.py
   - *Inputs*: data/processed/train_sessions.pkl, data/processed/test_sessions.pkl
   - *Outputs*: models/model.pth, metrics/metrics.json
   - *Description*: Trains a recommendation model using PyTorch with negative sampling loss. Logs metrics to MLFlow (http://192.168.28.39:5000).

3. *Evaluate*:
   - *Command*: python scripts/evaluate.py
   - *Inputs*: models/model.pth, data/processed/test_sessions.pkl
   - *Outputs*: metrics/eval_metrics.json
   - *Description*: Evaluates the trained model on test data and logs metrics.

## Usage

1. *Run the Pipeline Manually*
   
   dvc repro
   

2. *Monitor with Airflow*
   - Access the Airflow UI at http://192.168.28.39:8080.
   - Trigger the mlops_pipeline DAG to run the pipeline hourly.

3. *View Experiment Metrics*
   - Access the MLFlow UI at http://192.168.28.39:5000 to view training and evaluation metrics.


## Contributing

Fork the repository.

2. Create a feature branch: git checkout -b feature/your-feature.
3. Commit changes with the required format: git commit -m "feature(your-feature): add new functionality"
   - Commit message format: <type>(<scope>): <description> (e.g., feature(preprocess): add new data transformation).
   - Valid types: feature, fixbug, setup, release, fix, refactor, doc.
   - No uppercase letters allowed.
4. Push to the branch: git push origin feature/your-feature.
Create a pull request.


## License

This project is licensed under the MIT License.

---


### 7. *Cấu trúc thư mục cập nhật*

#### Trên máy local (/mnt/c/Users/HP/mlops_cs317/):

```text
/mnt/c/Users/HP/mlops_cs317/
├── README.md                # Newly added
├── .git/
├── .dvc/
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── config.yaml
│   └── dvc.yaml
├── data/
│   ├── raw/
│   │   ├── yoochoose-clicks.dat
│   │   ├── yoochoose-buys.dat
│   │   └── dataset-README.txt
│   ├── processed/
│   │   ├── clicks_train.parquet
│   │   ├── clicks_test.parquet
│   │   ├── item_encoder.pkl
│   │   ├── train_sessions.pkl
│   │   └── test_sessions.pkl
│   └── test/
│       └── yoochoose-test.dat
├── scripts/
│   ├── check_commit_message.sh
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test_commit.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── metrics.py
├── models/
│   └── model.pth
├── metrics/
│   ├── metrics.json
│   └── eval_metrics.json
├── tests/
├── notebooks/
│   └── MLOps.ipynb
└── requirements.txt
```
#### In server (/home/mlops/):
No changes needed unless you’re syncing the entire project to the server.
