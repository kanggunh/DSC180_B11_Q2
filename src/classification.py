import pandas as pd
import pickle
import os
import re

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def balanced_error_rate(y_true, y_pred):
    """
    Calculate the Balanced Error Rate (BER) for binary classification.

    The Balanced Error Rate is computed as the average of the misclassification rates 
    for each class, considering both sensitivity and specificity. It is a useful metric 
    for imbalanced datasets, as it gives equal importance to both classes.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted binary labels.

    Returns:
    float: The balanced error rate.
    """

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ber = 1 - (sensitivity + specificity) / 2
    return ber

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    # Predictions on test and train sets
    """
    Evaluate the performance of a model on a given test set.

    Parameters:
    model: A trained model
    X_train (array-like): Training data
    X_test (array-like): Testing data
    y_train (array-like): True labels for training data
    y_test (array-like): True labels for testing data
    model_name (str): Name of the model to identify it in the output

    Returns:
    dict: A dictionary containing the model name, train accuracy, test accuracy, test recall, and balanced error rate.

    Prints classification report, confusion matrix, train accuracy, test accuracy, test recall, and balanced error rate.
    """
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Accuracy scores
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    # Classification report for recall
    report = classification_report(y_test, y_pred_test, output_dict=True)
    test_recall = report['1']['recall'] 
    
    # Balanced Error Rate for test set
    test_ber = balanced_error_rate(y_test, y_pred_test)
    
    print(f"\nEvaluation Report for {model_name}:\n")
    print("Classification Report (Test Set):\n", classification_report(y_test, y_pred_test))
    print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, y_pred_test))
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Recall): {test_recall}")
    print("Balanced Error Rate (Test Set):", test_ber)
    
    # Return metrics as dictionary for further use
    return {
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_ber": test_ber
    }

def train_classification_models():
    """
    Train and evaluate multiple classification models using scientific text data labeled as relevant or irrelevant.

    This function reads text data from a CSV file, processes it using TF-IDF vectorization,
    and trains several classification models including Logistic Regression, Naive Bayes,
    SVM, Random Forest, and XGBoost. Each model is evaluated using metrics such as 
    accuracy, recall, and balanced error rate. The models are then fine-tuned using 
    hyperparameter grid search, and the tuned models are reevaluated. The results are 
    saved to a CSV file for further analysis.

    Steps involved:
    1. Load and preprocess the data, balancing the classes by undersampling the majority class.
    2. Convert the text data into TF-IDF features.
    3. Split the data into training and testing sets.
    4. Train and evaluate multiple classification models, storing their performance metrics.
    5. Perform hyperparameter tuning using GridSearchCV for each model.
    6. Evaluate the tuned models and store their performance metrics.
    7. Save the final results to a CSV file.

    Returns:
    None
    """

    training_data_path = "../data/classification/classifier_training.csv"
    data = pd.read_csv(training_data_path)

    majority_class = data[data['label'] == 0]  # Irrelevant papers
    minority_class = data[data['label'] == 1]  # Relevant papers

    undersample_ratio = 0.5  # Keep 50% of the majority class
    num_majority_to_keep = int(len(minority_class) / undersample_ratio)
    # Randomly sample the majority class to match the size of the minority class
    majority_sampled = majority_class.sample(n=num_majority_to_keep, random_state=42)

    # Combine the undersampled majority class with the full minority class
    df_balanced = pd.concat([majority_sampled, minority_class])

    # Shuffle the dataset to mix the classes
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    data = df_balanced

    # Convert the text data to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=4000)  # Adjust max_features as needed
    X = tfidf_vectorizer.fit_transform(data['text'])  
    y = data['label'] 

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'XGBoost']

    # list to keep record of performance metric
    recall_before = []
    accuracy_before = []
    ber_before = []

    recall_after = []
    accuracy_after = []
    ber_after = []

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    temp = evaluate_model(lr_model, X_train, X_test, y_train, y_test, model_name="Logistic Regression")
    recall_before = recall_before + [temp.get('test_recall')]
    accuracy_before = accuracy_before + [temp.get('test_accuracy')]
    ber_before = ber_before + [temp.get('test_ber')]

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    temp = evaluate_model(nb_model, X_train, X_test, y_train, y_test, model_name="Naive Bayes")
    recall_before = recall_before + [temp.get('test_recall')]
    accuracy_before = accuracy_before + [temp.get('test_accuracy')]
    ber_before = ber_before + [temp.get('test_ber')]

    # Support Vector Machine
    svm_model = LinearSVC(random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)
    temp = evaluate_model(svm_model, X_train, X_test, y_train, y_test, model_name="SVM")
    recall_before = recall_before + [temp.get('test_recall')]
    accuracy_before = accuracy_before + [temp.get('test_accuracy')]
    ber_before = ber_before + [temp.get('test_ber')]

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    temp = evaluate_model(rf_model, X_train, X_test, y_train, y_test, model_name="Random Forest")
    recall_before = recall_before + [temp.get('test_recall')]
    accuracy_before = accuracy_before + [temp.get('test_accuracy')]
    ber_before = ber_before + [temp.get('test_ber')]

    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, random_state=42)
    xgb_model.fit(X_train, y_train)
    temp = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, model_name="XGBoost")
    recall_before = recall_before + [temp.get('test_recall')]
    accuracy_before = accuracy_before + [temp.get('test_accuracy')]
    ber_before = ber_before + [temp.get('test_ber')]

    # number of cross validation folds
    cv_n = 5

    param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'newton-cg', 'sag', 'saga'],
    'max_iter': [50, 100, 200]
    }

    grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=cv_n, scoring='accuracy')
    grid_search_lr.fit(X_train, y_train)
    print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)
    best_lr_model = grid_search_lr.best_estimator_

    param_grid_nb = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0]
    }

    grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=cv_n, scoring='accuracy')
    grid_search_nb.fit(X_train, y_train)
    print("Best Parameters for Naive Bayes:", grid_search_nb.best_params_)
    best_nb_model = grid_search_nb.best_estimator_

    param_grid_rf = {
    'n_estimators': [100, 200, 250],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=cv_n, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)
    print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
    best_rf_model = grid_search_rf.best_estimator_

    param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'max_iter': [1, 10, 50, 100, 250, 500]
    }

    grid_search_svm = GridSearchCV(LinearSVC(random_state=42, class_weight='balanced'), param_grid_svm, cv=cv_n, scoring='accuracy')
    grid_search_svm.fit(X_train, y_train)
    print("Best Parameters for SVM:", grid_search_svm.best_params_)
    best_svm_model = grid_search_svm.best_estimator_

    param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 6],
    'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
                                param_grid_xgb, cv=cv_n, scoring='accuracy')
    grid_search_xgb.fit(X_train, y_train)
    print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)
    best_xgb_model = grid_search_xgb.best_estimator_

    temp = evaluate_model(best_lr_model, X_train, X_test, y_train, y_test, model_name="Tuned Logistic Regression")
    recall_after = recall_after + [temp.get('test_recall')]
    accuracy_after = accuracy_after + [temp.get('test_accuracy')]
    ber_after = ber_after + [temp.get('test_ber')]

    temp = evaluate_model(best_nb_model, X_train, X_test, y_train, y_test, model_name="Tuned Naive Bayes")
    recall_after = recall_after + [temp.get('test_recall')]
    accuracy_after = accuracy_after + [temp.get('test_accuracy')]
    ber_after = ber_after + [temp.get('test_ber')]

    temp = evaluate_model(best_svm_model, X_train, X_test, y_train, y_test, model_name="Tuned SVM")
    recall_after = recall_after + [temp.get('test_recall')]
    accuracy_after = accuracy_after + [temp.get('test_accuracy')]
    ber_after = ber_after + [temp.get('test_ber')]

    temp = evaluate_model(best_rf_model, X_train, X_test, y_train, y_test, model_name="Tuned Random Forest")
    recall_after = recall_after + [temp.get('test_recall')]
    accuracy_after = accuracy_after + [temp.get('test_accuracy')]
    ber_after = ber_after + [temp.get('test_ber')]

    temp = evaluate_model(best_xgb_model, X_train, X_test, y_train, y_test, model_name="Tuned XGBoost")
    recall_after = recall_after + [temp.get('test_recall')]
    accuracy_after = accuracy_after + [temp.get('test_accuracy')]
    ber_after = ber_after + [temp.get('test_ber')]

    # dataframe for further comparison with other models
    tf_idf = pd.DataFrame({
        'model_name':models,
        'BER': ber_after,
        'recall': recall_after,
        'accuracy': accuracy_after
    })

    # save it to the respective folder
    file_path = "data/classification/classification_results.csv"
    tf_idf.to_csv(file_path, index=False)


def passivator_frequency(text):
    """
    Calculate the frequency of specific passivator-related terms in a given text.

    This function takes a string input, converts it to lowercase, splits it into words,
    and counts the occurrences of the terms "passivation", "passivator", and "passivating".

    Parameters:
    text (str): The input text in which to count the occurrences of passivator-related terms.

    Returns:
    int: The total count of occurrences of the specified passivator-related terms in the input text.
    """

    passivator_names = ["passivation", "passivator", "passivating"]
    words = text.lower().split()
    return sum([words.count(name) for name in passivator_names])


def keyword_classification(text):
    """
    Classify a given text according to the presence of keywords related to passivation and power conversion efficiency (PCE).

    This function takes a string input, converts it to lowercase, and checks if any of the keywords in the passivator_names list and pces list are present in the text. Additionally, it checks if the total count of occurrences of the specified passivator-related terms in the input text is greater than 2.

    Parameters:
    text (str): The input text to classify.

    Returns:
    bool: Whether the text is classified as relevant or not.
    """
    passivators = ["passivation", "passivator", "passivating"]
    pces = ["pce", "power conversion efficiency"]
    return any(keyword in text.lower() for keyword in passivators) and any(keyword in text.lower() for keyword in pces) and passivator_frequency(text) > 2

def apply_keyword_classification():
    """
    Apply the keyword classification model to the scraped papers to determine if the papers are relevant to our specific domain or not.

    This function reads in the scraped papers from the data/scraping_and_conversion folder, applies the keyword classification model to each paper, and saves the results to a new CSV file in the data/classification folder.

    Parameters:
    None

    Returns:
    None
    """
    scraped_papers_path = "../data/scraping_and_conversion/scraped_papers.csv"
    df = pd.read_csv(scraped_papers_path)
    df["is_relevant"] = df["text"].apply(keyword_classification)
    relevant_papers = df[df["is_relevant"] == True]
    relevant_papers.to_csv("..data/classification/relevant_papers.csv", index=False)