import streamlit as st


ABOUT_STR = (
    'This is a simple pet-project to demonstrate some knowledge of Deep'
    'Learning with some MLOps practices and processes.  \n\n'
    'You can find all the source code on my [github]'
    '(https://github.com/nazavr322/ukr-handwriting-classification).'
)

st.set_page_config(
    page_title='About ⏺  Ukrainian Handwriting Classification',
    page_icon='📝',
    menu_items={
        'Get help': None,
        'Report a bug': 'https://github.com/nazavr322/ukr-handwriting-classification',
        'About': ABOUT_STR,
    },
)

st.title('About the project ❓')
st.markdown(
    ABOUT_STR
    + ' Below you will find detailed overview and explanation of different project parts.'
)

tab1, tab2, tab3 = st.tabs(
    [
        '📊 Data processing and DVC',
        '🧠 Model training',
        '🌐 MLFlow, FastAPI and Docker',
    ]
)
with tab1:
    st.header('📊 Data Processing and DVC')

    st.subheader('Dataset')
    st.markdown(
        'Original dataset was taken from [here](https://www.kaggle.com/datasets/lynnporu/rukopys). It consists of 1081 samples of Ukrainian handwritten letters, both uppercase and lowercase. You can see some examples below:'
    )
    st.image(
        'https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/10_raw_samples.png?raw=true',
        '10 raw samples from original dataset',
    )

    st.markdown(
        "The key point is that this dataset doesn't have handwritten digits included, so to fix this situation I decided to add 50 samples of each digit from [MNIST](http://yann.lecun.com/exdb/mnist/) to the original dataset. Resulting in a following distribution of samples:"
    )
    st.image(
        'https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/data_distribution.png?raw=true',
        'Distribution of samples after adding digits to a dataset.',
    )

    st.subheader('DVC processing pipeline')
    st.markdown(
        'Now the question arises how to correctly transform the data from different datasets to a one general form. I created a [DVC](https://dvc.org/) pipeline that allows you to go from raw to ready-to-train data using only one command. Our pipeline looks like this:'
    )
    st.image(
        'https://mermaid.ink/svg/pako:eNptkj9vwjAQxb9K5Jl_aQslGTrRoUO7wFZX0ck5wMJxLPsCQojvXicQy6SdLL_38-nunS9M1CWynG1VfRJ7sJRsVlxz0l5OvzkTCkEXJRBw9tMbT95opamF0_Tz62O9mZRHEQHPMbBTZ7N3E-GOA-rlH-qRmHuiggN2DTikyFq0FtodFkYKaiy6yHxtTX8qN620dFR0l4mhQf2l54xFAxaLWwORmUVmVyXy0pk3yYLUBeiywCOoBghjIg0Eoe_AGSUfKozHb92IIdS7kIUQ70I6FJYhwKEw75-EN4temYVs_ijLu7IIkw97y4ZEmoYqbNTuoQJZ-m904TpJ_Nx7rHwYedIu2B444_rqucb4PeJ7Kam2LCfb4IhBQ_X6rEV_vzErCTsLFcu3oBxefwHbCPKr',
        'Schematic look of a DVC data processing pipeline',
    )
    st.markdown(
        'Let me break it down for you. As you can see first three steps are executed in parallel:  \n'
        '1. **Clean data** - takes raw .csv file with ukrainian handwriting as an input and filters out all unnecessary information for our task.  \n'
        '2. **Prepare glyphs** - takes folder containing raw images of ukrainian handwriting as an input and converts them to MNIST format (inverted 28x28 images).  \n'
        '3. **Prepare MNIST** - takes folder containing raw byte-encoded MNIST images and produces equal amount of .png images per class, as well as .csv file with metadata about these pictures (label, filename, etc.)  \n'
        '4. **Make dataset** - takes cleaned .csv file from stage(1) and .csv file with MNIST metadata from stage(3) and joins them resulting in a final dataset.  \n'
        '5. **Merge pictures** - takes folder with processed images from stages(2) and (3) and merges them into one directory.  \n'
        '6. **Train/test split** - splits .csv file from stage(4) into train and test subsets.  \n'
        '7. **Train and evaluate** - fine-tunes model pre-trained on MNIST on a training data from stages (5) and (6) and evaluates its performance on test data also from stage(6).'
    )

    st.markdown(
        "That's it, even if it looks a little difficult, in fact, all the stages are quite simple. Take a look at how our final images look like after all the processing (without augmentations ofcourse):"
    )
    st.image(
        'https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/10_proc_samples.png?raw=true',
        '10 processed samples that are passed to a model',
    )
with tab2:
    st.header('🧠 Model training')

    st.markdown(
        "Actually I've used 2 models to solve my problem. As you can see on the plots from the previous page, amount of available data is very little. 1.5k of unequally distributed samples (43 classes and some of them don't have uppercase analogs at all) doesn't allow us to generalize well. So, my solution was pretty straightforward, pretrain the model on MNIST first (because letter images are pretty much the same) and then fine-tune it to solve multi-output classification problem."
    )

    st.subheader('MNIST Model')
    st.markdown(
        "MNIST classification problem was solved long time ago, so I have nothing special to say here. With pretty much default hyperparameters I was able to reach `accuracy = 99.39%` only after 25 epochs of training. More than enough for our task. You can see the nn's architecture below:"
    )
    st.image(
        'https://raw.githubusercontent.com/nazavr322/ukr-handwriting-classification/2a873aba19e103a8007f106fb25d6c9856842143/reports/figures/mnist_model_h.svg',
        'Architecture of MNIST Model. Made with Netron.',
    )

    st.subheader('Multi-Output CNN')
    st.markdown(
        "Here, I slightly modified the architecture above. Let's see how it looks now:"
    )
    st.image(
        'https://raw.githubusercontent.com/nazavr322/ukr-handwriting-classification/2a873aba19e103a8007f106fb25d6c9856842143/reports/figures/complete_model_h.svg',
        'Architecture of Multi-Output CNN. Made with Netron.',
    )
    st.markdown(
        "As you can see, I've replaced one classification head with two FCN layers. First has 43 outputs (33 ukrainian letters and 10 digits) and the second one has only 1 output to predict whether sample is uppercase and lowercase."
    )

    st.subheader('Loss functions and hyperparameters')
    st.markdown(
        "A few words about loss functions, after experimenting with different weighting strategies, to give more weight to a label classification task, I've discovered that one can achieve the most stable training process just summing up to loss functions. Thus, the final loss looked like this: $L = CE + BCE$."
    )
    st.markdown(
        'All the hyperparameters where fine-tuned with [`optuna`](https://github.com/optuna/optuna) framework'
    )

    st.subheader('Final Evaluation Results')
    st.markdown(
        'After training the model above for 30 epochs, I was able to achieve this results on test dataset of 300 samples:  \n'
        '- `Label classification accuracy = 94.3%`  \n'
        '- `Is uppercase classification accuracy = 92.6%`'
    )
    st.markdown(
        "Also, I've prepared confusion matrices to visualize model predictions:"
    )
    st.image(
        'https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/lbl_cm.png?raw=true',
        'Confusion matrix for label classfication',
    )
    st.image(
        'https://github.com/nazavr322/ukr-handwriting-classification/blob/main/reports/figures/is_upp_cm.png?raw=true',
        'Confusion matrix for case classification',
    )

    st.markdown(
        "I would not call the obtained results ideal, yes, there is room for improvement (that's why I'm collecting samples drawn by user actually), but still, I'm satisfied with the obtained metric values.  \n"
        'We have a very lightweight model, trained for only 15 epochs. On my laptop GPU training lasts for a minute at its best. It generalizes pretty good on both tasks simultaneously. On the first confusion matrix you can see that model sometimes confuses such ukrainian letters as, for example, `г` and `ґ`.  \n'
        'At the same time, we did pretty good on lowercase/uppercase classification too. On the corresponding confusion matrix you can see than we have only `2` false positives and `21` false negatives. And this is without knowing uppercase variants for 17 letters at all!'
    )
with tab3:
    st.header('🌐 MLFlow, FastAPI and Docker')

    st.subheader('MlFlow and FastAPI')
    st.markdown(
        'In this project I also use [MLFlow](https://mlflow.org/) for experiment tracking and registering models for production environment. My MlFlow workflow is built according to the following scenario:'
    )
    st.image(
        'https://mlflow.org/docs/latest/_images/scenario_4.png',
        'Diagram depicts how MLFlow workflow is organised.',
    )
    st.markdown(
        'In this architecture all storages and MLFlow tracking server itself are located on a remote hosts. Our code only acts as a client that makes requests to the tracking server, which logs metadata about runs into a database and stores artifacts (plots and model weights) in a remote S3 storage. I am using `PostgreSQL` as a database and `Minio` as a S3 storage.  \n'
        'I also wrote a simple API to work with a model using `FastAPI`. It loads a model version that is currently in `Production` stage in the MLFlow model registry. '
    )

    st.subheader('Docker')
    st.markdown(
        'I am running all these microservices using `docker-compose`, so you can reproduce a fully functional service with only a few commands.  \n'
        'Take a look at the scheme that depicts how my `docker-compose` is organized:'
    )
    st.image(
        'https://mermaid.ink/svg/pako:eNptU8Gu2jAQ_BXLT4ggGRICbWkOPVT0bqlVL-QJ-TkOz8Kxqe1Qqij_XschwaRVLrszs7NrZ91AqgoGMzibNVxym4Fmbt9ZxeYZmBesJLWw87adzXJZCvWbvhNtwY-vuQTg517RM9MXZexJM9M0q7hH4gGK2xbclisQX4mOBX8biV8iLoglYLW8gQE7REO0eA3sKzeVCrx9Phq_1Q60xht5JqiUJy5vVMkyqPbYqgOdQ2fALO3BgPJuPu3cvC0VnEl78PGxT17Bcvnl0bQS3f0Ypq9MH_rk2Ge9cDjcVPtw6W3_yz-GcT9LkAs_-ODoovsYQUWoe672wfPYOImiD9tNuli465iMiddR9DlJkoEbfXDqifW_xMa5BRXTsfA2inYhfx8zlx1LBTFmz0pwVaKumAElFyJ7KYuSsC0yVqszy152Hz-RNB310zVE4d6g6SoM1k_tLkrbsdmu-8ZmlNJHJ5wgvEY4RXiD8LYvgwhWrhnhhXtDTafNoX8_OcxcWBB9zmEuW6erL27n2beCW6VhVhJhGIKktur7H0lhZnXNBtGek5Mm1V3V_gWxWT1Z',
        'Schematic look of the Docker-Compose',
    )
    st.markdown(
        'It may look kind of confusing, let me break it down for you. We go from top to bottom:  \n'
        "- You can see that `model_api` microservice is mapped to the port `8000` on host machine to port `8000` in container (`'8000:8000'`). This is a docker container with our FastAPI server code.  \n"
        '- It depends on `mlflow_server` microservice, that runs on port `5000`. This is a docker container that runs MLFLow Tracking Server inside. You can find `Dockerfile` to build this image [here](https://github.com/nazavr322/ukr-handwriting-classification/blob/main/Docker/mlflow_image/Dockerfile).  \n'
        '- Then we see a dependency on `PostgreSQL` database. It uses volume, to save data even if container is stopped. Path inside a hexagon indicates a volume location on the host machine, and label matches location inside a container.  \n'
        '- Next, `mlflow_server` also depends on `minio_client` microservice. This is a small container to automatically create user and S3-bucket when first launching a `docker-compose`.  \n'
        '- Obviously, to create users and buckets, `minio_client` must depend on `minio` microservice, that gives us access to a Minio API on port `9000` and graphic UI on port `9001`. You can see another volume here.  \n'
        '- Finally, you may have noticed that we have `nginx` microservice that actually exposes ports `9000` and `9001` to a host machine. In this setup, `nginx` works as a load balancer and proxifies all requests to Minio API and Minio UI.'
    )
