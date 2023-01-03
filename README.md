# Project overview
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ukrainian-handwriting-classification.streamlit.app/)  
An end-to-end application for classifying Ukrainian handwriting that aims to demonstrate my knowledge of Deep Learning and MLOps tools and practices.  
On the web page, which is made using [streamlit](https://github.com/streamlit), the user can draw a Ukrainian letter or digit, and the neural network will try to recognize it and estimate whether it is an uppercase or lowercase symbol. For this task, I trained a lightweight multi-output CNN.  
Here are some of the tools that I've used during the development cycle: [DVC](https://github.com/iterative/dvc) to handle all the data versioning and preprocessing, [MLFlow](https://github.com/mlflow/mlflow) for experiment tracking and further model deployment, [Optuna](https://github.com/optuna/optuna) for hyperparameter optimization.  
On the server side of the application, I created a model API using [FastAPI](https://github.com/tiangolo/fastapi), configured the MLFlow tracking server and model registry that are powered up by [PostgreSQL](https://www.postgresql.org/) and [Minio S3](https://github.com/minio/minio), which also serves as a storage for user input, for further re-training.  
All of the above is wrapped up in reproducible [docker](https://github.com/docker) containers that are orchestrated using docker-compose and deployed to the Amazon EC2 instance.
## Demonstration
![](https://i.imgur.com/GovRbld.gif)

# Getting Started
## Prerequisites
To be able to work on this project, you need to have the following tools installed/configured on your machine: [`Poetry 1.2+`](https://python-poetry.org/), `Docker`, `AWS` credentials configured using [`AWS CLI`](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
## Steps to reproduce
- Clone the repository
- To install all the dependencies execute: 
	```shell
	poetry install
	```
	If you don't need dev dependencies (things like `jupyter`, `matplotlib`, `blue auto-formatter`, etc.) execute:
	```shell
	poetry install --without dev
	```
- To get all the training data from dvc remote, execute:
	```shell
	poetry run dvc pull
	```
- To create .env file with predefined environmental variables, execute:
	```shell
	cat .env.example > .env
	```
- To start the microservices defined in docker compose (MLflow tracking server, model API, etc.) execute:
	```shell
	sudo docker-compose up -d --build
	```
	Now you can accesss MLFlow UI at http://localhost:5000 and Minio UI at http://localhost:9001 (use credentials specified in a `.env` file) 
- To execute all the preprocessing steps and train a model run:
	```shell
	poetry run dvc repro
	```

# About
Below I will go over the various parts of the project, explaining some key points.
## Project Structure
```nohighlight
├── README.md           <- The top-level README for developers using this project.
│
├── data
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- The final, canonical data sets for modeling. 
│   └── raw             <- The original, immutable data.
│
├── Docker              <- Folder to store docker volumes, Dockerfiles and files needed to build images.
│   ├── mlflow_image
│   │   └── Dockerfile          <- Dockerfile to build image with mlflow server.
│   ├── model_api_image
│   │   └── Dockerfile          <- Dockerfile to build image with model API.
│   └── nginx.conf              <- Nginx configuration file for minio.
│
├── models              <- Trained and serialized models, hyperparameters.
│   ├── best_params.json        <- Hyperparameters to train models.
│   ├── final_model.pth         <- Weights of the model currently used in an application.
│   ├── mnist_model.pt          <- Weights of the model pretrained on the MNIST dataset.
│   └── model_heads.pth         <- Weights of the model where 2 classification heads where trained to convergence.
│
├── notebooks           <- Jupyter notebooks.
│   ├── eda_and_viz.ipynb       <- Notebook with EDA and visualizations.
│   ├── mnist_training.ipynb    <- Notebook with training base model on MNIST data.
│   └── optuna.ipynb            <- Notebook with hyperparameter optimization using Optuna.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting.
│
├── src                 <- Source code for use in this project.
│   ├── backend         <- All backend related code..
│   │   ├── main.py             <- Logic of a backend server.
│   │   └── utils.py            <- File with utility functions.
│   ├── data            <- Scripts related to data processing or generation.
│   │   ├── clean_data.py       <- Script to drop unneeded columns.
│   │   ├── datasets.py         <- File with definitions of datasets.
│   │   ├── make_dataset.py     <- Script to generate final variant of dataset.
│   │   ├── merge_pictures.py   <- Script to merge pictures from different sources into one folder.
│   │   ├── prepare_glyphs.py   <- Script to prepare pictures of ukr. handwriting to be compatible with MNIST format.
│   │   ├── prepare_mnist.py    <- Script to decode needed amount of MNIST data into pictures.
│   │   └── split_train_test.py <- Script to perform train/test split.
│   ├── features        <- Scripts to turn raw data into features for modeling.
│   │   └── features.py         <- File with functions to generate new features.
│   ├── frontend        <- All frontend related code.
│   │   ├── pages       <- Folder with web-site additional pages.
│   │   │   └── ❓ About.py     <- File with web-site about page.
│   │   ├── 🏠 Home.py          <- File with web-site main page.
│   │   └── utils.py            <- File with utility functions.
│   └── models          <- Scripts to train and evaluate models.
│       ├── functional.py       <- File with utility functions used in training and validation.
│       ├── models.py           <- File with model architecture definitions.
│       └── train_and_eval.py   <- Script to train and evaluate the model.
│
├── docker-compose.yaml <- File with definition of microservices in docker-compose.   
├── dvc.lock            <- File required for DVC data versioning.
├── dvc.yaml            <- File with definition of DVC data pre-processing pipeline.
├── poetry.lock         <- File that locks project dependencies to their current versions.
└── pyproject.toml      <- File with project settings.
```
## Data pre-processing and DVC
Original dataset was taken from [here](https://www.kaggle.com/datasets/lynnporu/rukopys). It consists of 1081 samples of Ukrainian handwritten letters, both uppercase and lowercase. You can see some examples below:    
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/10_raw_samples.png)   
The key point is that this dataset doesn't have handwritten digits included, so to fix this situation I decided to add 50 samples of each digit (from 0 to 9) to the dataset. Resulting in a following distribution of samples:    
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/data_distribution.png)    
Now the question arises how to correctly bring the data from different datasets to a one general form. You can find a bunch of pre-processing scripts in a corresponding `src/data` folder, but you don't need to worry about understanding and executing them correctly. I created a DVC pipeline that allows you to go from raw to ready-to-train data using only one command (I'll explain how to do it in a corresponding [section](#getting-started)).    
Pipeline looks like this:    
```mermaid
flowchart TD
	node1["clean_data"]
	node2["data/raw/MNIST.dvc"]
	node3["data/raw/glyphs.csv.dvc"]
	node4["data/raw/glyphs.dvc"]
	node5["make_dataset"]
	node6["merge_pictures"]
	node7["models/mnist_model.pt.dvc"]
	node8["prepare_glyphs"]
	node9["prepare_mnist"]
	node10["train_and_evaluate"]
	node11["train_test_split"]
	node1-->node5
	node2-->node9
	node3-->node1
	node3-->node8
	node4-->node8
	node5-->node11
	node6-->node10
	node7-->node10
	node8-->node6
	node9-->node5
	node9-->node6
	node11-->node10
```
Let me break it down for you. As you can see the first three steps are executed in parallel:
1. **Clean data** - takes raw .csv file with Ukrainian handwriting as an input and filters out all unnecessary information for our task.
2. **Prepare glyphs** - takes folder containing raw images of Ukrainian handwriting as an input and converts them to MNIST format (inverted 28x28 images).
3. **Prepare MNIST** - takes folder containing raw byte-encoded MNIST images and produces equal amount of .png images per class, as well as .csv file with metadata about these pictures (label, filename, etc.)
4. **Make dataset** - takes cleaned .csv file from stage(1) and .csv file with MNIST metadata from stage(3) and joins them resulting in a final dataset.
5. **Merge pictures** - takes folder with processed images from stages(2) and (3) and merges them into one directory.
6. **Train/test split** - splits .csv file from stage(4) into train and test subsets.
7. **Train and evaluate** - fine-tunes model pre-trained on MNIST on a training data from stages (5) and (6) and evaluates its performance on test data also from stage(6).

That's it, even if it looks a little difficult, in fact, all the stages are quite simple. Take a look at how our final images look like after all the processing (without augmentations ofcourse):    
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/10_proc_samples.png)   
## Model training
Actually I've used 2 models to solve my problem. As you can see on the plots above, amount of available data is very little. 1.5k of unequally distributed samples (43 classes and some of them don't have uppercase analogs at all) doesn't allow us to generalize well. So, my solution was pretty straightforward, pretrain the model on MNIST first (because letter images are pretty much the same) and then fine-tune it to solve multi-output classification problem.
### MNIST Model
MNIST classification problem was solved long time ago, so I have nothing special to say here. With pretty much default hyperparameters I was able to reach `accuracy = 99.39%` only after 25 epochs of training. More than enough for our task. You can see the nn's architecture below:   
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/mnist_model_h.svg)
### Multi-output CNN
Here, I slightly modified the architecture above. Let's see how it looks now:  
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/complete_model_h.svg)
As you can see, I've replaced one classification head with two FCN layers. First has 43 outputs (33 Ukrainian letters and 10 digits) and the second one has only 1 output to predict whether sample is uppercase and lowercase.   
### Loss functions and hyperparameters
A few words about loss functions, after experimenting with different weighting strategies, to give more weight to a label classification task, I've discovered that one can achieve the most stable training process by just summing up to loss functions. Thus, the final loss looked like this: $L = CE + BCE$.    
All the hyperparameters where fine-tuned with [`optuna`](https://github.com/optuna/optuna) framework, you can check out this code at `notebooks/optuna.ipynb`.
## Final Evaluation Results
After training the model above for 15 epochs, I was able to achieve this results on test dataset of 300 samples:
- `Label classification accuracy = 94.3%`
- `Is uppercase classification accuracy = 92.6%`

Also, I've prepared confusion matrices to visualize model predictions:
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/lbl_cm.png)
![](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/is_upp_cm.png)
    
    
I would not call the obtained results ideal, yes, there is room for improvement (that's why I'm collecting samples drawn by user actually), but still, I'm satisfied with the obtained metric values.
I have a very lightweight model, trained for only 15 epochs. On my laptop GPU training lasts for a minute at its best. It generalizes pretty good on both tasks simultaneously. On the first confusion matrix you can see that model sometimes confuses such Ukrainian letters as, for example, `г` and `ґ`.
At the same time, we did pretty good on lowercase/uppercase classification too. On the corresponding confusion matrix you can see than we have only `2` false positives and `21` false negatives.
   
## MLFlow, FastAPI and Docker
### MLFlow and FastAPI
In this project I also use MLFlow for experiment tracking and registering models for production environment. My MlFlow workflow is built according to the following scenario:   
![](https://mlflow.org/docs/latest/_images/scenario_4.png)
In this architecture all storages and MLFlow tracking server itself are located on a remote host(s). Our code only acts as a client that makes requests to the tracking server, which logs metadata about runs into a database and stores artifacts (plots and model weights) in a remote S3 storage. I am using `PostgreSQL` as a database and `Minio` as a S3 storage.   
I also wrote a simple API to work with a model using `FastAPI`. It loads a model version that is currently in `Production` stage in the MLFlow model registry.     
### Docker
I am running all these microservices using `docker-compose`, so you can reproduce a fully functional service with only a few commands (see [Getting Started](#getting-started) section).    
Take a look at the scheme that depicts how my `docker-compose` is organized:
```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TB
  VDockerpostgres{{./Docker/postgres/}} x-. /var/lib/postgresql/data .-x postgres[(postgres)]
  VDockerminio{{./Docker/minio/}} x-. /buckets .-x minio
  VDockernginxconf{{./Docker/nginx.conf}} -. /etc/nginx/nginx.conf .-x nginx
  minioclient[minio_client] --> minio
  mlflowserver[mlflow_server] --> postgres
  mlflowserver --> minioclient
  mlflowserver --> nginx
  modelapi[model_api] --> mlflowserver
  modelapi --> nginx
  nginx --> minio
  P0((5432)) -.-> postgres
  P1((9000)) -.-> nginx
  P2((9001)) -.-> nginx
  P3((5000)) -.-> mlflowserver
  P4((8000)) -.-> modelapi

  classDef volumes fill:#fdfae4,stroke:#867a22
  class VDockerpostgres,VDockerminio,VDockernginxconf volumes
  classDef ports fill:#f8f8f8,stroke:#ccc
  class P0,P1,P2,P3,P4 ports
```
It may look kind of confusing, let me break it down for you. I will go from top to bottom:   
- You can see that `model_api` microservice is mapped to the port `8000` on host machine to port `8000` in container (`'8000:8000'`). This is a docker container with our FastAPI server code.
- It depends on `mlflow_server` microservice, that runs on port `5000`. This is a docker container that runs MLFLow Tracking Server inside. You can find `Dockerfile` to build this image [here](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/Docker/mlflow_image/Dockerfile).
- Then we see a dependency on `PostgreSQL` database. It uses volume, to save data even if container is stopped. Path inside a hexagon indicates a volume location on the host machine, and label matches location inside a container.
- Next, `mlflow_server` also depends on `minio_client` microservice. This is a small container to automatically create user and S3-bucket when first launching a `docker-compose`.
- Obviously, to create users and buckets, `minio_client` must depend on `minio` microservice, that gives us access to a Minio API on port `9000` and graphic UI on port `9001`. You can see another volume here.
- Finally, you may have noticed that we have `nginx` microservice that actually exposes ports `9000` and `9001` to a host machine. In this setup, `nginx` works as a load balancer and proxifies all requests to Minio API and Minio UI.     

Now you can deploy this `docker-compose` to some cloud and experiment with different models, while all the needed data will be tracked and store remotely. Or build some services using model API. You can change usernames, passwords, bucket names etc. defined in a `.env` to suit your needs.
