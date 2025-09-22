"""
Medical Diagnosis Case Study: Small Data Machine Learning
=========================================================

This case study demonstrates machine learning techniques for medical diagnosis
using a small dataset. We simulate a scenario where we have limited patient
data for rare disease diagnosis.

Author: Small Data ML Research
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class MedicalDiagnosisMLCase:
    """
    A comprehensive case study for medical diagnosis using small datasets.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the medical diagnosis case study.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_medical_dataset(self):
        """
        Load and prepare the medical dataset (using breast cancer as example).
        
        Returns:
        --------
        X : array-like
            Feature matrix (patient measurements)
        y : array-like
            Target vector (diagnosis: 0=malignant, 1=benign)
        feature_names : list
            Names of medical features
        """
        # Load breast cancer dataset as proxy for medical data
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        # Create a more realistic medical scenario by:
        # 1. Reducing dataset size to simulate rare disease
        # 2. Creating class imbalance (fewer positive cases)
        
        # First, let's make it more imbalanced (fewer malignant cases)
        malignant_indices = np.where(y == 0)[0]  # malignant (positive class)
        benign_indices = np.where(y == 1)[0]     # benign (negative class)
        
        # Keep fewer malignant cases to simulate rare disease
        selected_malignant = np.random.choice(malignant_indices, size=30, replace=False)
        selected_benign = np.random.choice(benign_indices, size=80, replace=False)
        
        selected_indices = np.concatenate([selected_malignant, selected_benign])
        np.random.shuffle(selected_indices)
        
        X_small = X[selected_indices]
        y_small = y[selected_indices]
        
        print("Medical Dataset Summary:")
        print(f"  - Total patients: {len(y_small)}")
        print(f"  - Malignant cases: {np.sum(y_small == 0)} ({np.mean(y_small == 0)*100:.1f}%)")
        print(f"  - Benign cases: {np.sum(y_small == 1)} ({np.mean(y_small == 1)*100:.1f}%)")
        print(f"  - Number of features: {X_small.shape[1]}")
        print(f"  - Feature types: {', '.join(feature_names[:5])}...")
        
        return X_small, y_small, feature_names
    
    def exploratory_data_analysis(self, X, y, feature_names):
        """
        Perform exploratory data analysis on the medical dataset.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame(X, columns=feature_names)
        df['diagnosis'] = y
        
        # Basic statistics
        print("Dataset Statistics:")
        print(df.describe())
        
        # Class distribution
        plt.figure(figsize=(15, 10))
        
        # 1. Class distribution
        plt.subplot(2, 3, 1)
        class_counts = pd.Series(y).value_counts()
        plt.pie(class_counts.values, labels=['Malignant', 'Benign'], autopct='%1.1f%%')
        plt.title('Class Distribution')
        
        # 2. Feature correlation heatmap
        plt.subplot(2, 3, 2)
        # Select a subset of features for visualization
        important_features = feature_names[:10]
        corr_matrix = df[important_features].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        
        # 3. Box plots for key features
        plt.subplot(2, 3, 3)
        key_features = ['mean radius', 'mean texture', 'mean area']
        df_melted = df[key_features + ['diagnosis']].melt(
            id_vars='diagnosis', var_name='feature', value_name='value'
        )
        sns.boxplot(data=df_melted, x='feature', y='value', hue='diagnosis')
        plt.xticks(rotation=45)
        plt.title('Key Features by Diagnosis')
        
        # 4. Feature importance using mutual information
        plt.subplot(2, 3, 4)
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_indices = np.argsort(mi_scores)[::-1][:15]
        
        plt.barh(range(15), mi_scores[mi_indices])
        plt.yticks(range(15), [feature_names[i][:20] for i in mi_indices])
        plt.xlabel('Mutual Information Score')
        plt.title('Top 15 Features by Mutual Information')
        
        # 5. Principal Component Analysis
        plt.subplot(2, 3, 5)
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Visualization')
        plt.colorbar(scatter)
        
        # 6. Feature distributions
        plt.subplot(2, 3, 6)
        # Plot distribution of a highly discriminative feature
        best_feature_idx = np.argmax(mi_scores)
        best_feature = feature_names[best_feature_idx]
        
        malignant_values = X[y == 0, best_feature_idx]
        benign_values = X[y == 1, best_feature_idx]
        
        plt.hist(malignant_values, alpha=0.7, label='Malignant', bins=20)
        plt.hist(benign_values, alpha=0.7, label='Benign', bins=20)
        plt.xlabel(best_feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {best_feature[:20]}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print(f"\nMost discriminative feature: {best_feature}")
        print(f"Mutual information score: {mi_scores[best_feature_idx]:.3f}")
        
        # T-test for the best feature
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(malignant_values, benign_values)
        print(f"T-test p-value: {p_value:.2e}")
        
        return df
    
    def feature_selection_analysis(self, X, y, feature_names):
        """
        Analyze and select the most important features for the medical model.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
            
        Returns:
        --------
        X_selected : array-like
            Selected features
        selected_features : list
            Names of selected features
        """
        print("\n" + "="*60)
        print("FEATURE SELECTION ANALYSIS")
        print("="*60)
        
        # Different feature selection methods
        selectors = {
            'Univariate (f_classif)': SelectKBest(score_func=f_classif, k=10),
            'RFE (Random Forest)': RFE(
                RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                n_features_to_select=10
            )
        }
        
        feature_selection_results = {}
        
        for method_name, selector in selectors.items():
            # Fit selector
            X_selected = selector.fit_transform(X, y)
            
            # Get selected features
            if hasattr(selector, 'get_support'):
                mask = selector.get_support()
                selected_features = [feature_names[i] for i in range(len(mask)) if mask[i]]
            else:
                selected_features = feature_names[:10]  # fallback
            
            feature_selection_results[method_name] = {
                'X_selected': X_selected,
                'selected_features': selected_features,
                'selector': selector
            }
            
            print(f"\n{method_name}:")
            print(f"  Selected {len(selected_features)} features:")
            for i, feature in enumerate(selected_features[:5]):
                print(f"    {i+1}. {feature}")
            if len(selected_features) > 5:
                print(f"    ... and {len(selected_features)-5} more")
        
        # Compare feature selection methods using cross-validation
        cv_scores = {}
        base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Full feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring='roc_auc')
        cv_scores['All Features'] = scores
        
        # Selected feature sets
        for method_name, results in feature_selection_results.items():
            X_selected = results['X_selected']
            X_selected_scaled = StandardScaler().fit_transform(X_selected)
            scores = cross_val_score(base_model, X_selected_scaled, y, cv=cv, scoring='roc_auc')
            cv_scores[method_name] = scores
        
        # Visualize feature selection results
        plt.figure(figsize=(12, 8))
        
        # Cross-validation scores comparison
        plt.subplot(2, 2, 1)
        methods = list(cv_scores.keys())
        means = [cv_scores[method].mean() for method in methods]
        stds = [cv_scores[method].std() for method in methods]
        
        plt.bar(range(len(methods)), means, yerr=stds, capsize=5)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.ylabel('ROC AUC Score')
        plt.title('Feature Selection Method Comparison')
        plt.grid(True, alpha=0.3)
        
        # Feature importance from Random Forest
        plt.subplot(2, 2, 2)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        plt.barh(range(15), importances[indices])
        plt.yticks(range(15), [feature_names[i][:25] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        
        # Feature selection overlap
        plt.subplot(2, 2, 3)
        # Venn diagram-like visualization for feature overlap
        univariate_features = set(feature_selection_results['Univariate (f_classif)']['selected_features'])
        rfe_features = set(feature_selection_results['RFE (Random Forest)']['selected_features'])
        
        overlap = len(univariate_features & rfe_features)
        univariate_only = len(univariate_features - rfe_features)
        rfe_only = len(rfe_features - univariate_features)
        
        categories = ['Univariate Only', 'RFE Only', 'Both Methods']
        values = [univariate_only, rfe_only, overlap]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        plt.pie(values, labels=categories, colors=colors, autopct='%1.0f')
        plt.title('Feature Selection Method Overlap')
        
        # Learning curves with different feature sets
        plt.subplot(2, 2, 4)
        for method_name, results in feature_selection_results.items():
            X_selected = results['X_selected']
            train_sizes, train_scores, val_scores = learning_curve(
                base_model, X_selected, y, 
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=3, random_state=self.random_state
            )
            
            train_mean = np.mean(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            
            plt.plot(train_sizes, val_mean, 'o-', label=method_name)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Validation Score')
        plt.title('Learning Curves by Feature Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print results summary
        print(f"\nFeature Selection Results Summary:")
        for method_name, scores in cv_scores.items():
            print(f"  {method_name}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Return the best feature selection method
        best_method = max(cv_scores.keys(), key=lambda k: cv_scores[k].mean())
        print(f"\nBest method: {best_method}")
        
        if best_method in feature_selection_results:
            return (feature_selection_results[best_method]['X_selected'], 
                   feature_selection_results[best_method]['selected_features'])
        else:
            return X, feature_names
    
    def handle_class_imbalance(self, X, y):
        """
        Demonstrate techniques for handling class imbalance in medical data.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
            
        Returns:
        --------
        resampled_datasets : dict
            Dictionary containing different resampled datasets
        """
        print("\n" + "="*60)
        print("HANDLING CLASS IMBALANCE")
        print("="*60)
        
        print(f"Original class distribution:")
        print(f"  Malignant (0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
        print(f"  Benign (1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
        
        # Different resampling techniques
        resamplers = {
            'Original': None,
            'SMOTE': SMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'SMOTETomek': SMOTETomek(random_state=self.random_state)
        }
        
        resampled_datasets = {}
        
        for method_name, resampler in resamplers.items():
            if resampler is None:
                X_resampled, y_resampled = X, y
            else:
                X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            resampled_datasets[method_name] = (X_resampled, y_resampled)
            
            print(f"\n{method_name}:")
            print(f"  Total samples: {len(y_resampled)}")
            print(f"  Malignant: {np.sum(y_resampled == 0)} ({np.mean(y_resampled == 0)*100:.1f}%)")
            print(f"  Benign: {np.sum(y_resampled == 1)} ({np.mean(y_resampled == 1)*100:.1f}%)")
        
        # Evaluate impact of resampling on model performance
        model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        results = {metric: {} for metric in metrics}
        
        for method_name, (X_res, y_res) in resampled_datasets.items():
            for metric in metrics:
                scores = cross_val_score(model, X_res, y_res, cv=cv, scoring=metric)
                results[metric][method_name] = scores
        
        # Visualize resampling results
        plt.figure(figsize=(15, 10))
        
        # Class distribution comparison
        plt.subplot(2, 3, 1)
        methods = list(resampled_datasets.keys())
        malignant_counts = [np.sum(resampled_datasets[method][1] == 0) for method in methods]
        benign_counts = [np.sum(resampled_datasets[method][1] == 1) for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, malignant_counts, width, label='Malignant', alpha=0.8)
        plt.bar(x + width/2, benign_counts, width, label='Benign', alpha=0.8)
        plt.xlabel('Resampling Method')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution After Resampling')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        
        # Performance metrics comparison
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'roc_auc']):
            plt.subplot(2, 3, i + 2)
            
            method_names = list(results[metric].keys())
            means = [results[metric][method].mean() for method in method_names]
            stds = [results[metric][method].std() for method in method_names]
            
            plt.bar(range(len(method_names)), means, yerr=stds, capsize=5)
            plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\nResampling Performance Results:")
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for method_name in results[metric].keys():
                scores = results[metric][method_name]
                print(f"  {method_name}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        return resampled_datasets
    
    def comprehensive_model_comparison(self, X, y):
        """
        Compare different machine learning models for medical diagnosis.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        # Define models with appropriate hyperparameters for small datasets
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=self.random_state
            ),
            'SVM (RBF)': SVC(
                probability=True, random_state=self.random_state
            ),
            'Naive Bayes': GaussianNB()
        }
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Metrics to evaluate
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Store results
        cv_results = {metric: {} for metric in scoring_metrics}
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Evaluate on each metric
            for metric in scoring_metrics:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
                cv_results[metric][model_name] = scores
                print(f"  {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Visualize model comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(scoring_metrics):
            ax = axes[i]
            
            # Box plot for this metric
            model_names = list(cv_results[metric].keys())
            scores_data = [cv_results[metric][model] for model in model_names]
            
            bp = ax.boxplot(scores_data, labels=model_names, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Remove the last subplot if not needed
        if len(scoring_metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.show()
        
        # Model ranking
        print(f"\nModel Ranking by ROC AUC:")
        roc_auc_means = {model: cv_results['roc_auc'][model].mean() 
                        for model in cv_results['roc_auc'].keys()}
        
        ranked_models = sorted(roc_auc_means.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model, score) in enumerate(ranked_models):
            print(f"  {i+1}. {model}: {score:.3f}")
        
        # Detailed analysis of best model
        best_model_name = ranked_models[0][0]
        best_model = models[best_model_name]
        
        print(f"\nDetailed Analysis of Best Model: {best_model_name}")
        
        # Train on full dataset and get predictions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', best_model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 4))
        
        # Confusion matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return cv_results, best_model_name, pipeline
    
    def model_interpretability(self, model_pipeline, X, y, feature_names):
        """
        Analyze model interpretability for medical diagnosis.
        
        Parameters:
        -----------
        model_pipeline : sklearn.Pipeline
            Trained model pipeline
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        """
        print("\n" + "="*60)
        print("MODEL INTERPRETABILITY FOR MEDICAL DIAGNOSIS")
        print("="*60)
        
        # Extract the classifier from pipeline
        classifier = model_pipeline.named_steps['classifier']
        
        # Feature importance (if available)
        if hasattr(classifier, 'feature_importances_'):
            print("Feature Importance Analysis:")
            
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Print top 10 features
            print("\nTop 10 Most Important Features:")
            for i in range(min(10, len(feature_names))):
                feature_idx = indices[i]
                print(f"  {i+1:2d}. {feature_names[feature_idx][:40]:40s} "
                      f"({importances[feature_idx]:.3f})")
            
            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            
            # Feature importance plot
            plt.subplot(2, 1, 1)
            n_features = min(15, len(feature_names))
            plt.barh(range(n_features), importances[indices[:n_features]])
            plt.yticks(range(n_features), 
                      [feature_names[indices[i]][:30] for i in range(n_features)])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance for Medical Diagnosis')
            plt.gca().invert_yaxis()
            
            # Cumulative importance
            plt.subplot(2, 1, 2)
            cumulative_importance = np.cumsum(importances[indices])
            plt.plot(range(len(cumulative_importance)), cumulative_importance, 'o-')
            plt.axhline(y=0.8, color='red', linestyle='--', label='80% importance')
            plt.axhline(y=0.9, color='orange', linestyle='--', label='90% importance')
            plt.xlabel('Number of Features')
            plt.ylabel('Cumulative Importance')
            plt.title('Cumulative Feature Importance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Medical interpretation
            print("\nMedical Interpretation of Top Features:")
            top_features = [feature_names[indices[i]] for i in range(5)]
            
            feature_interpretations = {
                'mean area': 'Larger tumor area often indicates malignancy',
                'mean radius': 'Larger tumor radius suggests malignant growth',
                'mean perimeter': 'Irregular perimeter may indicate malignancy',
                'worst area': 'Worst-case area measurement is highly predictive',
                'worst radius': 'Maximum radius measurement shows tumor extent',
                'worst perimeter': 'Maximum perimeter indicates tumor boundary irregularity',
                'mean texture': 'Texture variation can indicate malignant tissue',
                'worst texture': 'Extreme texture values suggest abnormal tissue'
            }
            
            for feature in top_features:
                if any(key in feature.lower() for key in feature_interpretations.keys()):
                    for key, interpretation in feature_interpretations.items():
                        if key in feature.lower():
                            print(f"  • {feature}: {interpretation}")
                            break
                else:
                    print(f"  • {feature}: Significant predictor of malignancy")
        
        # Model calibration analysis
        print("\nModel Calibration Analysis:")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Get probability predictions
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
        
        # Calibration plot
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(12, 4))
        
        # Calibration curve
        plt.subplot(1, 3, 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=5
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Prediction confidence distribution
        plt.subplot(1, 3, 2)
        plt.hist(y_pred_proba[y_test == 0], bins=10, alpha=0.7, 
                label='Malignant (True)', density=True)
        plt.hist(y_pred_proba[y_test == 1], bins=10, alpha=0.7, 
                label='Benign (True)', density=True)
        plt.xlabel('Predicted Probability of Benign')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Decision threshold analysis
        plt.subplot(1, 3, 3)
        thresholds = np.linspace(0, 1, 100)
        sensitivity_scores = []
        specificity_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
        
        plt.plot(thresholds, sensitivity_scores, label='Sensitivity (Recall)')
        plt.plot(thresholds, specificity_scores, label='Specificity')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Sensitivity vs Specificity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Recommend optimal threshold for medical diagnosis
        # In medical diagnosis, we often want high sensitivity (catch all malignant cases)
        optimal_sensitivity_idx = np.argmax(np.array(sensitivity_scores) >= 0.9)
        optimal_threshold = thresholds[optimal_sensitivity_idx] if optimal_sensitivity_idx > 0 else 0.5
        
        print(f"\nRecommended Decision Threshold: {optimal_threshold:.3f}")
        print(f"  (Optimized for high sensitivity in medical diagnosis)")
        
        # Clinical decision support
        print(f"\nClinical Decision Support Guidelines:")
        print(f"  • Probability < 0.3: Likely benign, routine follow-up")
        print(f"  • Probability 0.3-0.7: Uncertain, consider additional tests")
        print(f"  • Probability > 0.7: Likely malignant, urgent further evaluation")
        
        return optimal_threshold


def main():
    """
    Run the complete medical diagnosis case study.
    """
    print("MEDICAL DIAGNOSIS CASE STUDY: SMALL DATA ML")
    print("="*70)
    
    # Initialize case study
    case_study = MedicalDiagnosisMLCase(random_state=42)
    
    # Load and explore data
    X, y, feature_names = case_study.load_medical_dataset()
    df = case_study.exploratory_data_analysis(X, y, feature_names)
    
    # Feature selection
    X_selected, selected_features = case_study.feature_selection_analysis(X, y, feature_names)
    
    # Handle class imbalance
    resampled_datasets = case_study.handle_class_imbalance(X_selected, y)
    
    # Use the best resampling method (SMOTE typically works well)
    X_final, y_final = resampled_datasets['SMOTE']
    
    # Comprehensive model comparison
    cv_results, best_model_name, best_pipeline = case_study.comprehensive_model_comparison(X_final, y_final)
    
    # Model interpretability
    optimal_threshold = case_study.model_interpretability(best_pipeline, X_final, y_final, selected_features)
    
    print("\n" + "="*70)
    print("CASE STUDY SUMMARY")
    print("="*70)
    print(f"✅ Successfully built medical diagnosis model using small dataset")
    print(f"📊 Dataset: {len(y)} patients with {len(feature_names)} features")
    print(f"🔍 Selected {len(selected_features)} most important features")
    print(f"⚖️ Handled class imbalance using SMOTE")
    print(f"🏆 Best model: {best_model_name}")
    print(f"🎯 Optimal decision threshold: {optimal_threshold:.3f}")
    print(f"\n🏥 This model could assist clinicians in:")
    print(f"   • Early detection of malignant cases")
    print(f"   • Risk stratification of patients")
    print(f"   • Prioritizing cases for further investigation")
    print(f"   • Reducing diagnostic uncertainty")


if __name__ == "__main__":
    main()