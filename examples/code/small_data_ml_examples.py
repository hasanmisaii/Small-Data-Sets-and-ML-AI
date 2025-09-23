"""
Small Data Machine Learning Examples
====================================

This module demonstrates various techniques for working with small datasets
in machine learning, including data augmentation, regularization, ensemble methods,
and cross-validation strategies.

Author: Small Data ML Research
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SmallDataMLExamples:
    """
    A comprehensive class demonstrating various machine learning techniques
    specifically designed for small datasets.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SmallDataMLExamples class.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_small_dataset(self, dataset_name='wine'):
        """
        Load a small dataset for demonstration purposes.
        
        Parameters:
        -----------
        dataset_name : str, default='wine'
            Name of the dataset to load ('wine', 'breast_cancer', or 'synthetic')
            
        Returns:
        --------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        target_names : list
            Names of target classes
        """
        if dataset_name == 'wine':
            data = load_wine()
            X, y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        elif dataset_name == 'synthetic':
            X, y = make_classification(
                n_samples=200, n_features=20, n_informative=10,
                n_redundant=10, n_clusters_per_class=1,
                random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            target_names = ['class_0', 'class_1']
            
        else:
            raise ValueError("Dataset name must be 'wine', 'breast_cancer', or 'synthetic'")
            
        print(f"Loaded {dataset_name} dataset:")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Classes: {len(np.unique(y))}")
        print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, feature_names, target_names
    
    def create_very_small_dataset(self, X, y, n_samples_per_class=10):
        """
        Create a very small dataset by sampling few examples per class.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        n_samples_per_class : int, default=10
            Number of samples to keep per class
            
        Returns:
        --------
        X_small : array-like
            Small feature matrix
        y_small : array-like
            Small target vector
        """
        X_small = []
        y_small = []
        
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            selected_indices = np.random.choice(
                class_indices, 
                size=min(n_samples_per_class, len(class_indices)), 
                replace=False
            )
            X_small.extend(X[selected_indices])
            y_small.extend(y[selected_indices])
        
        X_small = np.array(X_small)
        y_small = np.array(y_small)
        
        print(f"Created very small dataset:")
        print(f"  - Samples: {X_small.shape[0]}")
        print(f"  - Class distribution: {dict(zip(*np.unique(y_small, return_counts=True)))}")
        
        return X_small, y_small
    
    def demonstrate_data_augmentation_smote(self, X, y):
        """
        Demonstrate data augmentation using SMOTE for small datasets.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
            
        Returns:
        --------
        X_resampled : array-like
            Augmented feature matrix
        y_resampled : array-like
            Augmented target vector
        """
        print("\n" + "="*50)
        print("DATA AUGMENTATION WITH SMOTE")
        print("="*50)
        
        # Original class distribution
        original_dist = dict(zip(*np.unique(y, return_counts=True)))
        print(f"Original class distribution: {original_dist}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # New class distribution
        new_dist = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"After SMOTE class distribution: {new_dist}")
        print(f"Dataset size increased from {len(y)} to {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def compare_regularization_techniques(self, X, y):
        """
        Compare different regularization techniques for small datasets.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        """
        print("\n" + "="*50)
        print("REGULARIZATION TECHNIQUES COMPARISON")
        print("="*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different regularization strengths
        regularization_params = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        # Logistic Regression with L1 and L2 regularization
        l1_scores = []
        l2_scores = []
        
        for C in regularization_params:
            # L1 regularization (Lasso)
            l1_model = LogisticRegression(
                C=C, penalty='l1', solver='liblinear', 
                random_state=self.random_state, max_iter=1000
            )
            l1_model.fit(X_train_scaled, y_train)
            l1_score = l1_model.score(X_test_scaled, y_test)
            l1_scores.append(l1_score)
            
            # L2 regularization (Ridge)
            l2_model = LogisticRegression(
                C=C, penalty='l2', solver='liblinear',
                random_state=self.random_state, max_iter=1000
            )
            l2_model.fit(X_train_scaled, y_train)
            l2_score = l2_model.score(X_test_scaled, y_test)
            l2_scores.append(l2_score)
        
        # Plot regularization comparison
        plt.figure(figsize=(10, 6))
        plt.plot(regularization_params, l1_scores, 'o-', label='L1 (Lasso)', linewidth=2)
        plt.plot(regularization_params, l2_scores, 's-', label='L2 (Ridge)', linewidth=2)
        plt.xlabel('Regularization Strength (C)')
        plt.ylabel('Test Accuracy')
        plt.title('Regularization Techniques Comparison')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Find best parameters
        best_l1_idx = np.argmax(l1_scores)
        best_l2_idx = np.argmax(l2_scores)
        
        print(f"Best L1 regularization: C={regularization_params[best_l1_idx]}, "
              f"Accuracy={l1_scores[best_l1_idx]:.3f}")
        print(f"Best L2 regularization: C={regularization_params[best_l2_idx]}, "
              f"Accuracy={l2_scores[best_l2_idx]:.3f}")
    
    def ensemble_methods_comparison(self, X, y):
        """
        Compare different ensemble methods for small datasets.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        """
        print("\n" + "="*50)
        print("ENSEMBLE METHODS COMPARISON")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Define ensemble models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=5  # Limit depth to prevent overfitting
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=3, learning_rate=0.1
            ),
            'SVM': SVC(
                random_state=self.random_state, probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        }
        
        # Evaluate models using cross-validation
        cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_scores[name] = scores
            print(f"{name}: CV Accuracy = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Visualize cross-validation scores
        plt.figure(figsize=(12, 6))
        
        # Box plot of CV scores
        plt.subplot(1, 2, 1)
        box_data = [cv_scores[name] for name in models.keys()]
        plt.boxplot(box_data, labels=models.keys())
        plt.title('Cross-Validation Accuracy Distribution')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Mean CV scores
        plt.subplot(1, 2, 2)
        means = [cv_scores[name].mean() for name in models.keys()]
        stds = [cv_scores[name].std() for name in models.keys()]
        plt.bar(range(len(models)), means, yerr=stds, capsize=5)
        plt.xlabel('Models')
        plt.ylabel('Mean CV Accuracy')
        plt.title('Mean Cross-Validation Accuracy')
        plt.xticks(range(len(models)), models.keys(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return cv_scores
    
    def learning_curve_analysis(self, X, y, model_name='Random Forest'):
        """
        Analyze learning curves to understand data requirements.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        model_name : str
            Name of the model to analyze
        """
        print("\n" + "="*50)
        print(f"LEARNING CURVE ANALYSIS - {model_name}")
        print("="*50)
        
        # Select model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, max_depth=5
            )
        elif model_name == 'Logistic Regression':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=self.random_state))
            ])
        else:
            raise ValueError("Model name must be 'Random Forest' or 'Logistic Regression'")
        
        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation accuracy')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Analyze the gap between training and validation scores
        final_gap = train_mean[-1] - val_mean[-1]
        print(f"Final training-validation gap: {final_gap:.3f}")
        
        if final_gap > 0.1:
            print("⚠️  High bias detected - consider more complex model or more features")
        elif final_gap < 0.05:
            print("✅ Good bias-variance balance")
        else:
            print("⚠️  Some overfitting detected - consider regularization")
    
    def feature_importance_analysis(self, X, y, feature_names):
        """
        Analyze feature importance for small datasets.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        """
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Train Random Forest for feature importance
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state
        )
        rf_model.fit(X, y)
        
        # Get feature importances
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature ranking
        print("Feature ranking:")
        for i in range(min(10, len(feature_names))):  # Top 10 features
            print(f"{i+1:2d}. {feature_names[indices[i]][:30]:30s} ({importances[indices[i]]:.3f})")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        
        # Top features plot
        n_top = min(15, len(feature_names))
        plt.subplot(2, 1, 1)
        plt.bar(range(n_top), importances[indices[:n_top]])
        plt.title('Feature Importances (Random Forest)')
        plt.ylabel('Importance')
        plt.xticks(range(n_top), [feature_names[i][:20] for i in indices[:n_top]], 
                   rotation=45, ha='right')
        
        # Cumulative importance
        plt.subplot(2, 1, 2)
        cumulative_importance = np.cumsum(importances[indices])
        plt.plot(range(len(cumulative_importance)), cumulative_importance, 'o-')
        plt.axhline(y=0.8, color='r', linestyle='--', 
                   label='80% of total importance')
        plt.axhline(y=0.9, color='orange', linestyle='--', 
                   label='90% of total importance')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Recommend feature selection
        features_80 = np.where(cumulative_importance >= 0.8)[0][0] + 1
        features_90 = np.where(cumulative_importance >= 0.9)[0][0] + 1
        
        print(f"Recommendation:")
        print(f"  - {features_80} features capture 80% of importance")
        print(f"  - {features_90} features capture 90% of importance")
        print(f"  - Consider using top {features_80}-{features_90} features for small datasets")
    
    def cross_validation_strategies(self, X, y):
        """
        Demonstrate different cross-validation strategies for small datasets.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        """
        print("\n" + "="*50)
        print("CROSS-VALIDATION STRATEGIES FOR SMALL DATA")
        print("="*50)
        
        # Model to evaluate
        model = RandomForestClassifier(
            n_estimators=50, random_state=self.random_state, max_depth=5
        )
        
        # Different CV strategies
        cv_strategies = {
            '5-Fold CV': StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            '10-Fold CV': StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state),
            'Leave-One-Out': None  # Will use custom implementation
        }
        
        results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            if cv_name == 'Leave-One-Out':
                # Custom LOOCV implementation for small datasets
                from sklearn.model_selection import LeaveOneOut
                cv_strategy = LeaveOneOut()
                
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
            results[cv_name] = scores
            
            print(f"{cv_name}:")
            print(f"  Mean Accuracy: {scores.mean():.3f}")
            print(f"  Std Deviation: {scores.std():.3f}")
            print(f"  Number of folds: {len(scores)}")
            print()
        
        # Visualize CV results
        plt.figure(figsize=(12, 6))
        
        # Box plot comparison
        plt.subplot(1, 2, 1)
        cv_data = [results[name] for name in ['5-Fold CV', '10-Fold CV']]
        cv_labels = ['5-Fold CV', '10-Fold CV']
        plt.boxplot(cv_data, labels=cv_labels)
        plt.title('Cross-Validation Strategy Comparison')
        plt.ylabel('Accuracy')
        
        # Detailed comparison
        plt.subplot(1, 2, 2)
        means = [results[name].mean() for name in cv_strategies.keys()]
        stds = [results[name].std() for name in cv_strategies.keys()]
        x_pos = range(len(cv_strategies))
        
        plt.bar(x_pos, means, yerr=stds, capsize=5)
        plt.xlabel('CV Strategy')
        plt.ylabel('Mean Accuracy')
        plt.title('Cross-Validation Results')
        plt.xticks(x_pos, cv_strategies.keys(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def bootstrap_confidence_intervals(self, X, y, n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals for model performance.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        n_bootstrap : int
            Number of bootstrap samples
        """
        print("\n" + "="*50)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=50, random_state=self.random_state, max_depth=5
        )
        
        bootstrap_scores = []
        n_samples = len(X)
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Out-of-bag samples for testing
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            if len(oob_indices) > 0:
                X_oob = X[oob_indices]
                y_oob = y[oob_indices]
                
                # Train and evaluate
                model.fit(X_bootstrap, y_bootstrap)
                score = model.score(X_oob, y_oob)
                bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_ci = np.percentile(bootstrap_scores, (alpha/2) * 100)
        upper_ci = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        mean_score = np.mean(bootstrap_scores)
        
        print(f"Bootstrap Results ({n_bootstrap} iterations):")
        print(f"  Mean Accuracy: {mean_score:.3f}")
        print(f"  95% Confidence Interval: [{lower_ci:.3f}, {upper_ci:.3f}]")
        print(f"  Standard Error: {np.std(bootstrap_scores):.3f}")
        
        # Plot bootstrap distribution
        plt.figure(figsize=(10, 6))
        plt.hist(bootstrap_scores, bins=50, alpha=0.7, density=True)
        plt.axvline(mean_score, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_score:.3f}')
        plt.axvline(lower_ci, color='orange', linestyle='--', 
                   label=f'95% CI: [{lower_ci:.3f}, {upper_ci:.3f}]')
        plt.axvline(upper_ci, color='orange', linestyle='--')
        plt.xlabel('Accuracy')
        plt.ylabel('Density')
        plt.title('Bootstrap Distribution of Model Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return bootstrap_scores, (lower_ci, upper_ci)


def main():
    """
    Main function to run all small data ML examples.
    """
    print("SMALL DATA MACHINE LEARNING EXAMPLES")
    print("="*60)
    
    # Initialize the examples class
    ml_examples = SmallDataMLExamples(random_state=42)
    
    # Load dataset
    X, y, feature_names, target_names = ml_examples.load_small_dataset('wine')
    
    # Create an even smaller dataset to simulate extreme small data scenarios
    X_small, y_small = ml_examples.create_very_small_dataset(X, y, n_samples_per_class=15)
    
    # Run all demonstrations
    print("\n🔍 Running Small Data ML Analysis...")
    
    # 1. Data Augmentation
    X_augmented, y_augmented = ml_examples.demonstrate_data_augmentation_smote(X_small, y_small)
    
    # 2. Regularization Comparison
    ml_examples.compare_regularization_techniques(X_small, y_small)
    
    # 3. Ensemble Methods
    cv_scores = ml_examples.ensemble_methods_comparison(X_small, y_small)
    
    # 4. Learning Curve Analysis
    ml_examples.learning_curve_analysis(X_small, y_small, 'Random Forest')
    
    # 5. Feature Importance
    ml_examples.feature_importance_analysis(X_small, y_small, feature_names)
    
    # 6. Cross-Validation Strategies
    cv_results = ml_examples.cross_validation_strategies(X_small, y_small)
    
    # 7. Bootstrap Confidence Intervals
    bootstrap_scores, ci = ml_examples.bootstrap_confidence_intervals(X_small, y_small)
    
    print("\n✅ All demonstrations completed successfully!")
    print("\nKey Takeaways for Small Data ML:")
    print("1. Use appropriate cross-validation (LOOCV for very small datasets)")
    print("2. Apply regularization to prevent overfitting")
    print("3. Consider data augmentation techniques like SMOTE")
    print("4. Ensemble methods can improve robustness")
    print("5. Feature selection is crucial with limited data")
    print("6. Bootstrap confidence intervals provide uncertainty estimates")


if __name__ == "__main__":
    main()