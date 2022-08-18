# Project overview   
This is a simple pet-project to demonstrate some knowledge of Deep Learning and some MLOps practices and processes.   
The final version will be a web-site where you will be able to draw a ukrainian letter or digit and neural network will recognize it and determine wheter it's lowercase or uppercase (only for letters). All drawn samples will be automatically collected to improve model perfomance in a future.   
All data versioning, managing and preprocessing is done using DVC. I performed hyperparameter optimization with Optuna and allthe experiment tracking with MLFlow.  
   
**NOTE**: when the project will be finished, i will create a normal README.

# Current progress
- [x] **Data processing and DVC integration**
    - [x] Rewrite plain functions as CLI-compatible scripts
    - [x] Create remote S3 storage
    - [x] Create data pre-processing pipeline
    - [x] Add `train` and `evaluate` stages
- [x] **Model training**
    - [x] Create baseline model and pretrain it on MNIST.
    - [x] Using transfer learning fine-tune pretrained model to recognize both digits and letters as well as uppercase/lowercase classification (Multi-Output CNN).
- [ ] **Backend**
    - [x] Add hyperparameter logging and model tracking using MLFlow.
    - [ ] Create Docker Compose with Minio S3 for artifact storage, PosteSQL for model registry, MLFlow server and backend code with FastAPI
    - [ ] Deploy Docker container to some cloud VM.
- [ ] **Frontend**
    - [ ] Create web interface (most likely with streamlit)

