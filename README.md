# Project overview
This is a simple pet-project to demonstrate some knowledge of Deep Learning and some MLOps practices and processes.   
The final version will be a web-site where you will be able to draw a ukrainian letter or digit and neural network will recognize it and determine wheter it's lowercase or uppercase (only for letters). All drawn samples will be automatically collected to improve model perfomance in a future.   
All data versioning, managing and preprocessing is done using DVC. I performed hyperparameter optimization with Optuna and all the experiment tracking with MLFlow.  
   
**NOTE**: I will update this README as I make progress.

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
    
# About
Below I will go over the various parts of the project, explaining some of the key points.
## Project Structure
```nohighlight
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling. 
│   └── raw            <- The original, immutable data .
├── models             <- Trained and serialized models, hyperparameters.
├── notebooks          <- Jupyter notebooks.
│   ├── eda_and_viz.ipynb       <- Notebook with EDA and visualizations.
│   ├── mnist_training.ipynb    <- Notebook with training base model on MNIST data.
│   ├── model_training.ipynb    <- Notebook with test training of final model.
│   └── optuna.ipynb            <- Notebook with hyperparameter optimization using Optuna.
├── src                <- Source code for use in this project.
│   ├── data           <- Scripts related to data processing or generation.
│   │   ├── clean_data.py       <- Script to drop unneeded columns.
│   │   ├── datasets.py         <- File with definitions of datasets.
│   │   ├── make_dataset.py     <- Script to generate final variant of dataset.
│   │   ├── merge_pictures.py   <- Script to merge pictures from different sources into one folder.
│   │   ├── prepare_glyphs.py   <- Script to prepare pictures of ukr. handwriting to be compatible with MNIST format.
│   │   ├── prepare_mnist.py    <- Script to decode needed amount of MNIST data into pictures.
│   │   ├── split_train_test.py <- Script to perform train/test split.
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── features.py         <- File with functions to generate new features.
│   └── models         <- Scripts to train and evaluate models
│       ├── evaluate.py         <- Script to evaluate trained model on unseen data.
│       ├── functional.py       <- File with utility functions used in training and validation.
│       ├── models.py           <- File with model architecture definitions.
│       └── train.py            <- Script to train model.
├── dvc.lock           <- File required for DVC data versioning.
├── dvc.yaml           <- File with definition of DVC data pre-processing pipeline.
├── poetry.lock        <- File that locks project dependencies to their current versions.
└── pyproject.toml     <- File with project settings.
```
## Data pre-processing and DVC

