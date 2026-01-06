# Facial Emotion Recognition using Deep Learning

This repository contains the source code, experimental data, and deployment artifacts for a Deep Neural Network (DNN) designed to recognize facial emotions. The project is part of the "Deep Learning and Software Engineering" course and is divided into two main milestones: experimentation (Milestone 2) and deployment via a Dockerized User Interface (Milestone 3).

## Repository Structure

The project is organized as follows:

* **`dataset/`**: The dataset used for training and testing the model.
    * **Structure**: Divided into `train` and `test` directories.
    * **Classes**: Images are categorized into 7 emotion subfolders: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

* **`milestone2/`** (Experimentation Phase):
    * `main_exp1.py` & `main_exp2.py`: Python scripts executing the Design of Experiments (DoE) to find optimal hyperparameters.
    * `graphs.py`: Utility script to generate visualizations from experiment results.
    * `res_doe_1.csv` & `res_doe_2.csv`: Logged results containing metrics from the experiments.
    * `requirements.txt`: Python dependencies specific to the experimentation phase.

* **`milestone3/`** (Training & Deployment Phase):
    * `train.py`: The main script to train the final DNN model using the best hyperparameters found.
    * **`docker-app/`**: Contains the standalone web application for deployment.
        * `app.py`: Flask application serving the UI and handling predictions.
        * `Dockerfile`: Configuration file to containerize the application.
        * `fer_model_weights.h5`: Pre-trained weights of the Facial Emotion Recognition (FER) model.
        * `requirements.txt`: Dependencies required to run the web app.
        * `templates/index.html`: The HTML frontend interface.

---

## Milestone 2: Experiments & Analysis

This section covers the work done to optimize the model architecture and hyperparameters.

### Reproduction of Experiments

To re-run the experiments or analyze the graphs:

1.  **Setup Environment**:
    Navigate to the `milestone2/` folder and install the specific requirements.
    ```bash
    cd milestone2
    pip install -r requirements.txt
    ```

2.  **Run Experiments**:
    Execute the experiment scripts to perform the training loops and log results to CSV files.
    ```bash
    python main_exp1.py
    python main_exp2.py
    ```

3.  **Generate Graphs**:
    Use the `graphs.py` script to visualize the performance metrics from the generated CSV files.
    ```bash
    python graphs.py
    ```

---

## Milestone 3: Deployment

This milestone provides a simple User Interface (UI) allowing users to upload an image and get an emotion prediction. The application is containerized to ensure it runs consistently on any machine.

### Prerequisites
* **Docker Desktop** must be installed and running.

### Quick Start with Docker

1.  **Navigate to the Application Directory**:
    Open your terminal and move to the docker-app folder:
    ```bash
    cd milestone3/docker-app
    ```

2.  **Build the Docker Image**:
    Build the image using the provided Dockerfile. We tag it as `fer-app`.
    ```bash
    docker build -t fer-app .
    ```

3.  **Run the Container**:
    Start the container and map port **5000** of the container to port **5000** on your host machine.
    ```bash
    docker run -p 5000:5000 fer-app
    ```

4.  **Use the Application**:
    Open your web browser and navigate to:
    [http://localhost:5000](http://localhost:5000)

### Running Locally

If you prefer to run the Flask app directly in a Python environment:
1.  Navigate to `milestone3/docker-app`.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run the server: `python app.py`.

## Model Training

If you wish to retrain the final model from scratch using the full dataset:

1.  Ensure you are in the `milestone3/` directory.
2.  Verify that the `dataset/` folder is accessible relative to the script.
3.  Run the training script:
    ```bash
    python train.py
    ```
    *This will save the trained weights (e.g., `fer_model_weights.h5`) which can then be moved to the `docker-app` folder for deployment.*

## Authors

* **Group 5**
* Espinosa Florian
* Reynaud Anthéo
