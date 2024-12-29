# Text Classification Project Using Naive Bayes and Neural Networks

## Project Description

This project implements and compares two distinct approaches to text classification: a Naive Bayes classifier with advanced n-gram modeling and a Neural Network with enhanced feature engineering. Our implementation focuses on classification accuracy, utilizing a comprehensive dataset of 120,000 documents evenly distributed across four classes.

## Dataset Description

We worked with a large-scale text classification dataset comprising 120,000 documents, carefully balanced across four distinct classes with 30,000 documents each. This dataset was split using 5-fold cross-validation, ensuring robust model evaluation. For each fold, we maintain a training set of 96,000 documents and a validation set of 24,000 documents, preserving the original class distribution.

## Technical Implementation

### Data Preprocessing Pipeline

Our preprocessing pipeline includes three main steps:
- Text cleaning with special characters handling and normalization
- Tokenization using NLTK's optimized tokenizer
- Stop word filtering through vectorized operations

### Feature Engineering

We developed two complementary feature extraction approaches:

1. **N-gram Feature Extraction**
   - Implements unigram, bigram, and trigram models
   - Utilizes TF-IDF weighting for feature importance
   - Optimizes vocabulary selection through statistical analysis

2. **Statistical Feature Analysis**
   We identified key statistical features through PCA, focusing on:
   - Text structure metrics
   - Document length characteristics
   - Sentence complexity measures

### Model Architectures

#### Naive Bayes Implementation
Our Naive Bayes classifier incorporates several advanced techniques:
- Laplace smoothing with α=1.0 for handling unseen events
- Good-Turing estimation for improved probability estimation of rare events
- N-gram interpolation weighted model (0.5/0.3/0.2 for uni/bi/trigrams)

#### Neural Network Implementations

1. **Base Architecture**
   - Input layer adapts to vocabulary size (5, 10, or 15 words)
   - Hidden layer with 100 neurons and ReLU activation
   - Output layer using LogSoftmax for multi-class classification
   - Dropout regularization at 20%

2. **Enhanced Architecture**
   - Combines TF-IDF features with statistical measures
   - Implements normalized feature scaling
   - Adds dedicated input neurons for structural features
   - Maintains computational efficiency through vectorized operations

## Performance Results

Our experimental results demonstrate:

### Naive Bayes Results
- Best configuration: 2-gram with Laplace smoothing
- Accuracy: 90.67% (±0.21%)
- Per-class performance:
  - Class 1: 95.58% accuracy, 90.12% recall
  - Class 2: 98.39% accuracy, 98.22% recall
  - Class 3: 94.04% accuracy, 86.58% recall
  - Class 4: 94.40% accuracy, 89.90% recall

### Neural Network Results
- Best configuration: 15 words vocabulary
- Accuracy: 87.90% (±0.22%)
- Per-class performance:
  - Class 1: 93.83% accuracy, 88.13% recall
  - Class 2: 96.79% accuracy, 94.27% recall
  - Class 3: 92.40% accuracy, 84.50% recall
  - Class 4: 92.19% accuracy, 83.50% recall

## Installation and Setup

For Unix:
source nlp_env/bin/activate

For Windows:
.\nlp_env\Scripts\activate

Install dependencies:
pip install -r requirements.txt

## Command Line Interface

Available arguments:
--mode [ann/nbayes/annspe/all]: Execution mode
--output_dir [folder]: Output directory for results
--n_folds [number]: Number of folds for cross-validation

## Dependencies

Required packages:
- Python 3.11 or higher
- PyTorch >= 2.0.0 
- scikit-learn
- NLTK
- NumPy and Pandas
- tqdm >= 4.65.0
- matplotlib
- seaborn
- jupyter

## Future Development

1. **Model Enhancements**
   - Implementation of adaptive interpolation weights
   - Integration of attention mechanisms in neural models
   - Development of hybrid architectures

2. **Feature Engineering**
   - Addition of semantic embeddings
   - Implementation of character-level n-grams
   - Development of domain-specific features

3. **Performance Optimization**
   - Integration of parallel processing for n-gram computation
   - Implementation of dynamic vocabulary selection
   - Enhancement of memory management for large datasets

## Project Structure

    .
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── feature_extraction.py
    │   ├── naive_bayes.py
    │   ├── neural_network.py
    │   └── utils.py
    ├── results/
    │   ├── metrics/
    │   │   ├── naive_bayes/
    │   │   └── ann/
    │   └── models/
    ├── main.py
    └── requirements.txt
