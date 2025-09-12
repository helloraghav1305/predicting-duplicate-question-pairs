## Duplicate Question Pair Detection using Transformers

This repository contains a Kaggle-based implementation for detecting **duplicate question pairs** using a pre-trained transformer model.

The goal is to predict whether two given questions are semantically similar - a critical task for platforms to avoid redundant questions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Example Predictions](#example-predictions)

---

## Project Overview

This project uses a subset of the Questions Pairs Dataset on Kaggle to identify whether two questions are duplicates.

The pipeline includes:

- Data preprocessing using Pandas
- Tokenization with Hugging Face Transformers
- Training a pre-trained model using the `Trainer` API
- Evaluation with accuracy calculation

---

## Dataset

- **Source**: Kaggle - Question Pairs Dataset
- **Input Columns to consider**: `question1`, `question2`
- **Label Column**: `is_duplicate`

The dataset is preprocessed as follows:

- Remove rows with missing questions
- Concatenate into a single string format:  **QUES1:** question1 **QUES2:** question2

---

## Model

- **Model Used**: `microsoft/deberta-v3-small`
- **Framework**: Hugging Face Transformers
- **Task**: Sequence Classification (binary)

## Training Details

| Parameter              | Value     |
| ---------------------- | --------- |
| Batch size             | 32        |
| Epochs                 | 3         |
| Learning rate          | 8e-5      |
| Scheduler              | Cosine    |
| Weight Decay           | 0.01      |
| Mixed Precision (fp16) | ✅ Enabled |

## Evaluation

Evaluation is performed on a test split and a separate evaluation set

- **Metric**: Accuracy (via `sklearn.metrics.accuracy_score`)
- **Sample Output**: 
Test Set Accuracy: 0.86

## Installation & Setup

✅ **Requirements**
- Python 3.8+
- transformers
- datasets
- scikit-learn
- pandas
- numpy

## Project Structure

| File/Folder            | Description                                |
| ---------------------- | ------------------------------------------ |
| notebooks              | Contains jupyter notebook for this project |
| .gitignore             | Prevents loading unnecessary files         |
| README.md              | Project documentation (this file)          |
| requirements.txt       | List of Python packages required           |

## Example Predictions

| predicted_labels[:10]                  |
| -------------------------------------- |
| Output: [0, 1, 0, 0, 1, 1, 0, 1, 0, 1] |

## Future Improvements

- Use the full dataset for improved performance
- Add metrics like F1-score, Precision, Recall
- Try other pre-trained models like RoBERTa
- Deploy as a web app using Streamlit or Gradio

## Attribution

- Model: [deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) (MIT License)
- Dataset: [Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset)


