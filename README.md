# Baseline Text Classification Pipeline for Financial Ledger Descriptions

This project demonstrates a reproducible Python text classification workflow for labeling noisy financial ledger descriptions as either **Personnel** or **Non-Personnel** expenses.

The goal is to show an end-to-end machine learning workflow, including text preprocessing, abbreviation normalization, train/test validation, TF-IDF vectorization, baseline model training, evaluation, and a conceptual data mutation diagram.

## Project Overview

Financial ledger descriptions are often messy, abbreviated, and inconsistent. This project builds a baseline text classification pipeline that converts raw text descriptions into standardized expense categories.

The workflow includes:

1. Loading financial ledger description data
2. Cleaning and normalizing raw text
3. Mapping common abbreviations to more meaningful terms
4. Splitting the data into training and test sets
5. Converting text into TF-IDF features
6. Training a baseline linear classification model
7. Evaluating model performance using precision, recall, and F1-score
8. Explaining the data transformation lifecycle through a data mutation diagram

## Repository Structure

```text
text-classification-ledger-expenses/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample_ledger_expenses.csv
│
├── notebooks/
│   └── text_classification_pipeline.ipynb
│
└── outputs/
    └── data_mutation_layout.png
```

## Data

The sample data included in this repository are synthetic examples created for demonstration purposes. No proprietary or employer-provided dataset is included.

The dataset contains two columns:

* `raw_text`: noisy or abbreviated financial ledger description
* `category`: target label, either `Personnel` or `Non-Personnel`

Example records may include abbreviated descriptions such as teacher salary, office supplies, substitute teacher costs, utility bills, repairs, or other school district operating expenses.

## Methods

The project uses a simple and interpretable machine learning pipeline:

* Text preprocessing
* Abbreviation mapping
* Train/test split
* TF-IDF vectorization
* Baseline linear classifier
* Classification report with precision, recall, and F1-score

The pipeline is intentionally simple and reproducible. The emphasis is on software engineering hygiene, data transformation logic, model validity, and clear communication of the workflow.

## Key Skills Demonstrated

This project demonstrates experience with:

* Python
* pandas
* scikit-learn
* Text preprocessing
* TF-IDF feature engineering
* Supervised text classification
* Train/test validation
* Model evaluation
* Data cleaning
* Reproducible notebook development
* Communicating data transformations to technical and non-technical audiences

## How to Run

First, install the required packages:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebooks/text_classification_pipeline.ipynb
```

Run the notebook from top to bottom.

## Code Sample Highlights

Recommended files to review:

1. `notebooks/text_classification_pipeline.ipynb`
   End-to-end text classification workflow, including preprocessing, model training, and evaluation.

2. `data/sample_ledger_expenses.csv`
   Synthetic example data used for demonstration.

3. `outputs/data_mutation_layout.png`
   Conceptual diagram showing how raw text is transformed through preprocessing, tokenization, vectorization, model prediction, and human review logic.

## Notes on Data Leakage Prevention

The train/test split is performed before model fitting. The TF-IDF vectorizer is fit only on the training data and then applied to the test data through a scikit-learn pipeline. This helps prevent information from the test set from leaking into the model training process.

## Author

Yu-Chun Wang
Ph.D. Candidate in Statistics
George Mason University
