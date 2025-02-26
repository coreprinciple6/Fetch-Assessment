# Sentence Transformers & Multi-Task Learning Assessment by Fetch

This repository contains all the code related to the take-home assessment from Fetch on the topic **Sentence Transformers & Multi-Task Learning**. The objective of this project is to demonstrate how to expand a pre-trained sentence transformer model for a multi-task learning scenario. The repository includes data preparation, model expansion, training, evaluation, and visualization components.

---

## Repository Structure

- **graphs/**  
  Contains all the graphs generated during training and evaluation.

- **task1.py**  
  Contains the solution for Task 1. Running this script will:
  - Load the sentence transformer model.
  - Test sample sentences using the model.
  - Compute the cosine similarity matrix for the resultant embeddings.
  - Save the cosine similarity graph in the `graphs` folder.

- **task2.py**  
  Contains the model class where the sentence transformer model has been expanded to address a multi-task learning problem.

- **task4_dataset.py**  
  Provides the data preparation class that creates a dummy dataset for training and evaluating the model.

- **task4_train.py**  
  Contains functions for training, evaluating, and visualizing the model performance. This script holds the bulk of the project logic for multi-task learning.

- **task4_main.py**  
  The main entry point that combines all the above components:
  - Loads the expanded model.
  - Loads and prepares the dummy dataset.
  - Trains and evaluates the model.
  - Saves the generated graphs to the `graphs` folder.

- **Explanation.pdf**  
  A detailed report covering:
  - An explanation of the project components.
  - Code design choices.
  - Rationale behind the implementation decisions.

---

## Getting Started

### Prerequisites

- **Python 3.7+**
- Required Python libraries (see `requirements.txt`):
  torch,
  transformers,
  matplotlib,
  sklearn,
  numpy,
  pandas,
  docker,

### Installation

1. **Clone and nstall the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
---

## Running the Project

### Running Task 1

Task 1 demonstrates how to generate and visualize cosine similarity matrices from sentence embeddings.

```bash
python task1.py
```
After running, check the `graphs` folder for the saved cosine similarity graph.

### Running Multi-Task Learning (Task 4)

To run the complete multi-task learning pipeline:

```bash
python task4_main.py
```

### Using Docker

For an easier and reproducible setup, you can run the project within a Docker container.
- Ensure docker is installed and running
- build the docker image by running the following command
  ```bash
  docker build -t sentence-transformer-mtl .
  ```
- Run the docker container by running the following command
  ```bash
  docker run --rm -it -v $(pwd)/graphs:/app/graphs sentence-transformer-mtl
  ```


