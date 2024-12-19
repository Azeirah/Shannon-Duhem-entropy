import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from surprise import calculate_surprise
logging.basicConfig(level=logging.INFO)

def create_anomaly_dataset(X, y, normal_digit=1, n_samples=1000, random_state=42):
    """Create dataset with normal and anomalous digits using random sampling."""
    rng = np.random.RandomState(random_state)
    
    # Get indices for normal and anomalous samples
    normal_idx = np.where(y == str(normal_digit))[0]
    anomaly_idx = np.where(y != str(normal_digit))[0]
    
    # Randomly sample from both classes
    if len(normal_idx) > n_samples:
        normal_idx = rng.choice(normal_idx, size=n_samples, replace=False)
    if len(anomaly_idx) > n_samples:
        anomaly_idx = rng.choice(anomaly_idx, size=n_samples, replace=False)
    
    # Extract samples and create labels
    normal_X = X[normal_idx]
    anomaly_X = X[anomaly_idx]
    normal_y = np.zeros(len(normal_X))
    anomaly_y = np.ones(len(anomaly_X))
    
    # Combine and shuffle datasets
    X_combined = np.vstack([normal_X, anomaly_X])
    y_combined = np.hstack([normal_y, anomaly_y])
    
    # Shuffle the combined dataset
    shuffle_idx = rng.permutation(len(y_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]
    
    return X_combined, y_combined

def evaluate_surprise_detection(X_train, X_test, y_test):
    """Evaluate surprise-based anomaly detection performance."""
    surprise_scores = np.array([
        calculate_surprise(img.reshape(28, 28)).mean() 
        for img in X_test
    ])
    return compute_metrics(y_test, surprise_scores)

def evaluate_isolation_forest(X_train, X_test, y_test, random_state=42):
    """Evaluate Isolation Forest performance."""
    # Fit Isolation Forest on all training data
    iso_forest = IsolationForest(
        random_state=random_state,
        contamination=0.5,  # Dataset is balanced
        n_jobs=-1
    )
    iso_forest.fit(X_train.reshape(len(X_train), -1))
    
    # Get anomaly scores (-1 for anomalies, 1 for normal samples)
    scores = -iso_forest.score_samples(X_test.reshape(len(X_test), -1))
    return compute_metrics(y_test, scores)

def evaluate_one_class_svm(X_train, X_test, y_test):
    """Evaluate One-Class SVM performance."""
    # Fit One-Class SVM on all training data
    ocsvm = OneClassSVM(kernel='rbf', nu=0.5)  # nu=0.5 since dataset is balanced
    ocsvm.fit(X_train.reshape(len(X_train), -1))
    
    # Get anomaly scores
    scores = -ocsvm.decision_function(X_test.reshape(len(X_test), -1))
    return compute_metrics(y_test, scores)

def compute_metrics(y_true, scores):
    """Compute evaluation metrics for any anomaly detection method."""
    auc_score = roc_auc_score(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    
    return {
        'auc_score': auc_score,
        'avg_precision': avg_precision,
        'precision': precision,
        'recall': recall,
        'scores': scores
    }

def cross_validate_performance(X, y, n_splits=5, random_state=42):
    """Perform cross-validation for all methods."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    methods = {
        'Surprise': evaluate_surprise_detection,
        'IsolationForest': evaluate_isolation_forest,
        'OneClassSVM': evaluate_one_class_svm
    }
    
    results = {name: {'auc_scores': [], 'ap_scores': []} for name in methods}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info(f"\nFold {fold + 1}:")
        for name, method in methods.items():
            metrics = method(X_train_scaled, X_test_scaled, y_test)
            results[name]['auc_scores'].append(metrics['auc_score'])
            results[name]['ap_scores'].append(metrics['avg_precision'])
            
            logging.info(f"{name:13} AUC-ROC = {metrics['auc_score']:.3f}, "
                        f"AP = {metrics['avg_precision']:.3f}")
    
    # Compute summary statistics
    summary = {}
    for name in methods:
        summary[name] = {
            'mean_auc': np.mean(results[name]['auc_scores']),
            'std_auc': np.std(results[name]['auc_scores']),
            'mean_ap': np.mean(results[name]['ap_scores']),
            'std_ap': np.std(results[name]['ap_scores'])
        }
    
    return summary

def main():
    # Set random seed for reproducibility
    np.random.seed(55)
    
    # Load MNIST dataset
    logging.info("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    
    # Create anomaly detection dataset with random sampling
    logging.info("Creating anomaly detection dataset...")
    X_anomaly, y_anomaly = create_anomaly_dataset(X, y, normal_digit=1, n_samples=10000)
    
    # Perform cross-validation
    logging.info("\nPerforming cross-validation...")
    cv_results = cross_validate_performance(X_anomaly, y_anomaly)
    
    logging.info("\nFinal Results:")
    for method, metrics in cv_results.items():
        logging.info(f"\n{method}:")
        logging.info(f"Mean AUC-ROC: {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
        logging.info(f"Mean AP: {metrics['mean_ap']:.3f} ± {metrics['std_ap']:.3f}")

if __name__ == "__main__":
    main()