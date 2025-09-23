"""
Degradation Modeling Case Study: Small Data Machine Learning for Reliability Engineering
=======================================================================================

This case study demonstrates machine learning techniques for degradation modeling
using small datasets, inspired by real-world reliability engineering applications.
We simulate scenarios common in industrial settings where component degradation
data is limited but critical for predictive maintenance.

This implementation addresses the type of problems discussed in reliability engineering
literature, including small sample reliability demonstration tests with degradation data.
The techniques shown here are applicable to real-world scenarios such as:
- Accelerated testing of electronic components
- Mechanical wear modeling with limited test data
- Battery degradation analysis in automotive applications
- Infrastructure health monitoring with sparse sensor data

Key focus areas:
- Working with very small datasets (typically 20-100 components)
- Physics-informed feature engineering
- Uncertainty quantification for decision-making
- Remaining Useful Life (RUL) prediction
- Time series analysis with temporal dependencies

Author: Small Data ML Research
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    GridSearchCV, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DegradationModelingCase:
    """
    A comprehensive case study for degradation modeling using small datasets.
    Focuses on reliability engineering and predictive maintenance applications.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the degradation modeling case study.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_degradation_data(self, n_components=50, degradation_type='linear'):
        """
        Generate synthetic degradation data mimicking real industrial components.
        
        This simulates common degradation patterns found in mechanical components,
        electronics, and other industrial systems where data is typically limited.
        
        Parameters:
        -----------
        n_components : int
            Number of components to simulate (small sample size)
        degradation_type : str
            Type of degradation pattern ('linear', 'exponential', 'mixed')
            
        Returns:
        --------
        data : DataFrame
            Degradation data with features and targets
        """
        print(f"\n{'='*60}")
        print("GENERATING DEGRADATION DATA")
        print('='*60)
        
        # Time points (typical for accelerated testing)
        time_points = np.linspace(0, 1000, 20)  # 20 time points over 1000 hours
        
        # Component characteristics (features)
        np.random.seed(self.random_state)
        
        # Operating conditions
        temperature = np.random.normal(75, 10, n_components)  # Celsius
        load_factor = np.random.uniform(0.5, 1.5, n_components)  # Load multiplier
        humidity = np.random.uniform(30, 80, n_components)  # Percentage
        material_quality = np.random.choice([1, 2, 3], n_components, p=[0.2, 0.6, 0.2])
        
        # Generate degradation paths
        degradation_data = []
        
        for i in range(n_components):
            component_id = f"COMP_{i+1:03d}"
            
            # Individual component parameters based on operating conditions
            base_rate = 0.001 * (1 + 0.1 * (temperature[i] - 75) / 10)  # Temperature effect
            base_rate *= load_factor[i]  # Load effect
            base_rate *= (1 + 0.05 * (humidity[i] - 55) / 25)  # Humidity effect
            base_rate *= (2 - material_quality[i] * 0.3)  # Material quality effect
            
            # Add random variation
            individual_rate = base_rate * np.random.lognormal(0, 0.3)
            
            for j, t in enumerate(time_points):
                # Different degradation models
                if degradation_type == 'linear':
                    # Linear degradation with noise
                    degradation = individual_rate * t + np.random.normal(0, 0.02)
                elif degradation_type == 'exponential':
                    # Exponential degradation (common in electronics)
                    degradation = 0.01 * (np.exp(individual_rate * t) - 1) + np.random.normal(0, 0.01)
                elif degradation_type == 'mixed':
                    # Mixed pattern (some linear, some exponential)
                    if i % 2 == 0:
                        degradation = individual_rate * t + np.random.normal(0, 0.02)
                    else:
                        degradation = 0.01 * (np.exp(individual_rate * t) - 1) + np.random.normal(0, 0.01)
                
                # Ensure degradation is non-negative and realistic
                degradation = max(0, min(degradation, 1.0))
                
                degradation_data.append({
                    'component_id': component_id,
                    'time': t,
                    'degradation': degradation,
                    'temperature': temperature[i],
                    'load_factor': load_factor[i],
                    'humidity': humidity[i],
                    'material_quality': material_quality[i],
                    'age_at_start': np.random.uniform(0, 100),  # Previous operating hours
                    'maintenance_history': np.random.poisson(2)  # Number of previous maintenances
                })
        
        df = pd.DataFrame(degradation_data)
        
        print(f"Generated degradation data:")
        print(f"  - Components: {n_components}")
        print(f"  - Total observations: {len(df)}")
        print(f"  - Time points per component: {len(time_points)}")
        print(f"  - Degradation type: {degradation_type}")
        print(f"  - Features: {df.columns.tolist()}")
        
        return df
    
    def exploratory_degradation_analysis(self, df):
        """
        Perform exploratory data analysis on degradation data.
        """
        print(f"\n{'='*60}")
        print("EXPLORATORY DEGRADATION ANALYSIS")
        print('='*60)
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(df.describe())
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Degradation Data Exploratory Analysis', fontsize=16)
        
        # 1. Degradation trajectories by component
        ax1 = axes[0, 0]
        components_to_plot = df['component_id'].unique()[:10]  # Plot first 10 components
        for comp in components_to_plot:
            comp_data = df[df['component_id'] == comp]
            ax1.plot(comp_data['time'], comp_data['degradation'], alpha=0.7, linewidth=1)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Degradation Level')
        ax1.set_title('Degradation Trajectories (Sample of Components)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of degradation levels
        ax2 = axes[0, 1]
        ax2.hist(df['degradation'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Degradation Level')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Degradation Levels')
        ax2.grid(True, alpha=0.3)
        
        # 3. Operating conditions correlation
        ax3 = axes[0, 2]
        operating_features = ['temperature', 'load_factor', 'humidity']
        corr_matrix = df[operating_features + ['degradation']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Operating Conditions Correlation')
        
        # 4. Degradation vs Temperature
        ax4 = axes[1, 0]
        scatter = ax4.scatter(df['temperature'], df['degradation'], 
                            c=df['time'], cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Degradation Level')
        ax4.set_title('Degradation vs Temperature (colored by time)')
        plt.colorbar(scatter, ax=ax4, label='Time')
        
        # 5. Material quality impact
        ax5 = axes[1, 1]
        for quality in sorted(df['material_quality'].unique()):
            quality_data = df[df['material_quality'] == quality]
            ax5.scatter(quality_data['time'], quality_data['degradation'], 
                       label=f'Quality {quality}', alpha=0.6)
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Degradation Level')
        ax5.set_title('Degradation by Material Quality')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Degradation rate distribution
        ax6 = axes[1, 2]
        # Calculate degradation rates for each component
        degradation_rates = []
        for comp in df['component_id'].unique():
            comp_data = df[df['component_id'] == comp].sort_values('time')
            if len(comp_data) > 1:
                rate = (comp_data['degradation'].iloc[-1] - comp_data['degradation'].iloc[0]) / \
                       (comp_data['time'].iloc[-1] - comp_data['time'].iloc[0])
                degradation_rates.append(rate)
        
        ax6.hist(degradation_rates, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax6.set_xlabel('Degradation Rate (per hour)')
        ax6.set_ylabel('Number of Components')
        ax6.set_title('Distribution of Degradation Rates')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print(f"\nKey Findings:")
        print(f"  - Average degradation level: {df['degradation'].mean():.4f}")
        print(f"  - Degradation range: {df['degradation'].min():.4f} to {df['degradation'].max():.4f}")
        print(f"  - Number of unique components: {df['component_id'].nunique()}")
        print(f"  - Time range: {df['time'].min():.0f} to {df['time'].max():.0f} hours")
        print(f"  - Average degradation rate: {np.mean(degradation_rates):.6f} per hour")
        
        return df
    
    def prepare_regression_dataset(self, df):
        """
        Prepare the degradation data for regression modeling.
        
        This creates features suitable for predicting degradation levels
        and remaining useful life.
        """
        print(f"\n{'='*60}")
        print("PREPARING REGRESSION DATASET")
        print('='*60)
        
        # Create features and target
        feature_columns = ['time', 'temperature', 'load_factor', 'humidity', 
                          'material_quality', 'age_at_start', 'maintenance_history']
        
        # Add derived features
        df_features = df.copy()
        df_features['temp_time_interaction'] = df_features['temperature'] * df_features['time']
        df_features['load_time_interaction'] = df_features['load_factor'] * df_features['time']
        df_features['cumulative_stress'] = (df_features['temperature'] - 75) * df_features['load_factor'] * df_features['time']
        
        # Polynomial features for time (degradation often follows polynomial patterns)
        df_features['time_squared'] = df_features['time'] ** 2
        df_features['time_cubed'] = df_features['time'] ** 3
        
        # Logarithmic time (for exponential degradation)
        df_features['log_time'] = np.log1p(df_features['time'])
        
        updated_features = feature_columns + ['temp_time_interaction', 'load_time_interaction', 
                                            'cumulative_stress', 'time_squared', 'time_cubed', 'log_time']
        
        X = df_features[updated_features]
        y = df_features['degradation']
        
        print(f"Regression dataset prepared:")
        print(f"  - Number of samples: {len(X)}")
        print(f"  - Number of features: {len(updated_features)}")
        print(f"  - Feature names: {updated_features}")
        print(f"  - Target variable: degradation level")
        
        return X, y, updated_features
    
    def small_data_regression_comparison(self, X, y, feature_names):
        """
        Compare regression models suitable for small degradation datasets.
        """
        print(f"\n{'='*60}")
        print("SMALL DATA REGRESSION MODEL COMPARISON")
        print('='*60)
        
        # Define models suitable for small datasets
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            'Bayesian Ridge': BayesianRidge(),
            'Random Forest': RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=self.random_state
            ),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'Gaussian Process': GaussianProcessRegressor(
                kernel=RBF() + WhiteKernel(), random_state=self.random_state
            )
        }
        
        # Use time series split for temporal data
        cv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            # Cross-validation scores
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(-cv_scores)
            
            # Additional metrics
            r2_scores = cross_val_score(pipeline, X, y, cv=cv, 
                                      scoring='r2', n_jobs=-1)
            mae_scores = cross_val_score(pipeline, X, y, cv=cv, 
                                       scoring='neg_mean_absolute_error', n_jobs=-1)
            mae_scores = -mae_scores
            
            results[name] = {
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'mae_mean': mae_scores.mean(),
                'mae_std': mae_scores.std()
            }
            
            print(f"  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
            print(f"  R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
            print(f"  MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models_list = list(results.keys())
        rmse_means = [results[m]['rmse_mean'] for m in models_list]
        rmse_stds = [results[m]['rmse_std'] for m in models_list]
        r2_means = [results[m]['r2_mean'] for m in models_list]
        r2_stds = [results[m]['r2_std'] for m in models_list]
        mae_means = [results[m]['mae_mean'] for m in models_list]
        mae_stds = [results[m]['mae_std'] for m in models_list]
        
        # RMSE comparison
        axes[0].errorbar(range(len(models_list)), rmse_means, yerr=rmse_stds, 
                        fmt='o', capsize=5, capthick=2)
        axes[0].set_xticks(range(len(models_list)))
        axes[0].set_xticklabels(models_list, rotation=45, ha='right')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Root Mean Square Error Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].errorbar(range(len(models_list)), r2_means, yerr=r2_stds, 
                        fmt='s', capsize=5, capthick=2, color='green')
        axes[1].set_xticks(range(len(models_list)))
        axes[1].set_xticklabels(models_list, rotation=45, ha='right')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('R² Score Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[2].errorbar(range(len(models_list)), mae_means, yerr=mae_stds, 
                        fmt='^', capsize=5, capthick=2, color='red')
        axes[2].set_xticks(range(len(models_list)))
        axes[2].set_xticklabels(models_list, rotation=45, ha='right')
        axes[2].set_ylabel('MAE')
        axes[2].set_title('Mean Absolute Error Comparison')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Rank models by RMSE
        ranked_models = sorted(results.items(), key=lambda x: x[1]['rmse_mean'])
        
        print(f"\nModel Ranking (by RMSE):")
        for i, (model, metrics) in enumerate(ranked_models):
            print(f"  {i+1}. {model}: RMSE={metrics['rmse_mean']:.4f}, R²={metrics['r2_mean']:.4f}")
        
        return results
    
    def remaining_useful_life_prediction(self, df, failure_threshold=0.8):
        """
        Predict Remaining Useful Life (RUL) for components.
        
        This is a key application in reliability engineering and predictive maintenance.
        """
        print(f"\n{'='*60}")
        print("REMAINING USEFUL LIFE (RUL) PREDICTION")
        print('='*60)
        
        print(f"Failure threshold: {failure_threshold} degradation level")
        
        # Prepare RUL dataset
        rul_data = []
        
        for comp in df['component_id'].unique():
            comp_data = df[df['component_id'] == comp].sort_values('time')
            
            for i, row in comp_data.iterrows():
                current_degradation = row['degradation']
                current_time = row['time']
                
                # Find time when component reaches failure threshold
                future_data = comp_data[comp_data['time'] > current_time]
                failure_times = future_data[future_data['degradation'] >= failure_threshold]
                
                if len(failure_times) > 0:
                    failure_time = failure_times['time'].iloc[0]
                    rul = failure_time - current_time
                else:
                    # If failure threshold not reached in data, estimate
                    max_time = comp_data['time'].max()
                    max_degradation = comp_data['degradation'].max()
                    
                    if max_degradation < failure_threshold:
                        # Extrapolate using linear trend
                        degradation_rate = (max_degradation - comp_data['degradation'].iloc[0]) / \
                                         (max_time - comp_data['time'].iloc[0])
                        if degradation_rate > 0:
                            remaining_degradation = failure_threshold - current_degradation
                            rul = remaining_degradation / degradation_rate
                        else:
                            rul = np.inf  # No degradation observed
                    else:
                        rul = max_time - current_time
                
                # Only include realistic RUL values
                if 0 < rul < 2000:  # Between 0 and 2000 hours
                    rul_data.append({
                        'component_id': row['component_id'],
                        'current_time': current_time,
                        'current_degradation': current_degradation,
                        'temperature': row['temperature'],
                        'load_factor': row['load_factor'],
                        'humidity': row['humidity'],
                        'material_quality': row['material_quality'],
                        'age_at_start': row['age_at_start'],
                        'maintenance_history': row['maintenance_history'],
                        'rul': rul
                    })
        
        rul_df = pd.DataFrame(rul_data)
        
        if len(rul_df) == 0:
            print("No RUL data could be generated with the current failure threshold.")
            return None
        
        print(f"RUL dataset created:")
        print(f"  - Number of observations: {len(rul_df)}")
        print(f"  - RUL range: {rul_df['rul'].min():.1f} to {rul_df['rul'].max():.1f} hours")
        print(f"  - Average RUL: {rul_df['rul'].mean():.1f} hours")
        
        # Prepare features for RUL prediction
        feature_cols = ['current_time', 'current_degradation', 'temperature', 
                       'load_factor', 'humidity', 'material_quality', 
                       'age_at_start', 'maintenance_history']
        
        # Add derived features
        rul_df['degradation_rate'] = rul_df['current_degradation'] / (rul_df['current_time'] + 1)
        rul_df['stress_factor'] = rul_df['temperature'] * rul_df['load_factor']
        
        feature_cols.extend(['degradation_rate', 'stress_factor'])
        
        X_rul = rul_df[feature_cols]
        y_rul = rul_df['rul']
        
        # Train RUL prediction model
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=self.random_state
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Cross-validation for RUL prediction
        cv_scores = cross_val_score(pipeline, X_rul, y_rul, cv=5, 
                                  scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        print(f"\nRUL Prediction Performance:")
        print(f"  - RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} hours")
        
        # Fit model for feature importance
        pipeline.fit(X_rul, y_rul)
        feature_importance = pipeline.named_steps['regressor'].feature_importances_
        
        # Visualize results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RUL distribution
        axes[0].hist(y_rul, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Remaining Useful Life (hours)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Remaining Useful Life')
        axes[0].grid(True, alpha=0.3)
        
        # Feature importance for RUL prediction
        sorted_idx = np.argsort(feature_importance)
        axes[1].barh(range(len(feature_cols)), feature_importance[sorted_idx])
        axes[1].set_yticks(range(len(feature_cols)))
        axes[1].set_yticklabels([feature_cols[i] for i in sorted_idx])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Feature Importance for RUL Prediction')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return rul_df, pipeline
    
    def uncertainty_quantification_analysis(self, X, y):
        """
        Demonstrate uncertainty quantification for degradation modeling.
        
        Critical for reliability engineering where decision-making requires
        understanding prediction confidence.
        """
        print(f"\n{'='*60}")
        print("UNCERTAINTY QUANTIFICATION ANALYSIS")
        print('='*60)
        
        # Bayesian Ridge Regression for uncertainty quantification
        bayesian_model = BayesianRidge()
        
        # Gaussian Process for uncertainty quantification
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp_model = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\nTraining uncertainty quantification models...")
        
        # Bayesian Ridge
        bayesian_model.fit(X_train_scaled, y_train)
        y_pred_bayesian, y_std_bayesian = bayesian_model.predict(X_test_scaled, return_std=True)
        
        # Gaussian Process
        gp_model.fit(X_train_scaled, y_train)
        y_pred_gp, y_std_gp = gp_model.predict(X_test_scaled, return_std=True)
        
        # Bootstrap ensemble for uncertainty estimation
        n_bootstrap = 100
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
            X_boot = X_train_scaled[indices]
            y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
            
            # Train model on bootstrap sample
            boot_model = Ridge(alpha=1.0, random_state=i)
            boot_model.fit(X_boot, y_boot)
            boot_pred = boot_model.predict(X_test_scaled)
            bootstrap_predictions.append(boot_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        y_pred_bootstrap = bootstrap_predictions.mean(axis=0)
        y_std_bootstrap = bootstrap_predictions.std(axis=0)
        
        # Calculate metrics
        rmse_bayesian = np.sqrt(mean_squared_error(y_test, y_pred_bayesian))
        rmse_gp = np.sqrt(mean_squared_error(y_test, y_pred_gp))
        rmse_bootstrap = np.sqrt(mean_squared_error(y_test, y_pred_bootstrap))
        
        print(f"\nUncertainty Quantification Results:")
        print(f"  Bayesian Ridge - RMSE: {rmse_bayesian:.4f}")
        print(f"  Gaussian Process - RMSE: {rmse_gp:.4f}")
        print(f"  Bootstrap Ensemble - RMSE: {rmse_bootstrap:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Sort by actual values for better visualization
        sort_idx = np.argsort(y_test.values if hasattr(y_test, 'values') else y_test)
        
        # Bayesian Ridge
        axes[0].fill_between(range(len(y_test)), 
                            (y_pred_bayesian - 1.96 * y_std_bayesian)[sort_idx],
                            (y_pred_bayesian + 1.96 * y_std_bayesian)[sort_idx],
                            alpha=0.3, color='blue', label='95% Confidence Interval')
        axes[0].plot(y_test.iloc[sort_idx] if hasattr(y_test, 'iloc') else y_test[sort_idx], 
                    'o', color='red', markersize=4, label='Actual')
        axes[0].plot(y_pred_bayesian[sort_idx], '-', color='blue', linewidth=2, label='Predicted')
        axes[0].set_xlabel('Sample Index (sorted)')
        axes[0].set_ylabel('Degradation Level')
        axes[0].set_title('Bayesian Ridge Regression')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gaussian Process
        axes[1].fill_between(range(len(y_test)), 
                            (y_pred_gp - 1.96 * y_std_gp)[sort_idx],
                            (y_pred_gp + 1.96 * y_std_gp)[sort_idx],
                            alpha=0.3, color='green', label='95% Confidence Interval')
        axes[1].plot(y_test.iloc[sort_idx] if hasattr(y_test, 'iloc') else y_test[sort_idx], 
                    'o', color='red', markersize=4, label='Actual')
        axes[1].plot(y_pred_gp[sort_idx], '-', color='green', linewidth=2, label='Predicted')
        axes[1].set_xlabel('Sample Index (sorted)')
        axes[1].set_ylabel('Degradation Level')
        axes[1].set_title('Gaussian Process Regression')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Bootstrap Ensemble
        axes[2].fill_between(range(len(y_test)), 
                            (y_pred_bootstrap - 1.96 * y_std_bootstrap)[sort_idx],
                            (y_pred_bootstrap + 1.96 * y_std_bootstrap)[sort_idx],
                            alpha=0.3, color='orange', label='95% Confidence Interval')
        axes[2].plot(y_test.iloc[sort_idx] if hasattr(y_test, 'iloc') else y_test[sort_idx], 
                    'o', color='red', markersize=4, label='Actual')
        axes[2].plot(y_pred_bootstrap[sort_idx], '-', color='orange', linewidth=2, label='Predicted')
        axes[2].set_xlabel('Sample Index (sorted)')
        axes[2].set_ylabel('Degradation Level')
        axes[2].set_title('Bootstrap Ensemble')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'bayesian': (y_pred_bayesian, y_std_bayesian),
            'gp': (y_pred_gp, y_std_gp),
            'bootstrap': (y_pred_bootstrap, y_std_bootstrap)
        }

def main():
    """
    Run the complete degradation modeling case study.
    """
    print("DEGRADATION MODELING CASE STUDY")
    print("="*60)
    print("Demonstrating Small Data ML Techniques for Reliability Engineering")
    print()
    print("This case study covers:")
    print("   • Degradation data generation and analysis")
    print("   • Small data regression techniques")
    print("   • Remaining Useful Life (RUL) prediction")
    print("   • Uncertainty quantification for reliability decisions")
    print("   • Real-world applications in predictive maintenance")
    
    # Initialize case study
    case_study = DegradationModelingCase(random_state=42)
    
    # Generate degradation data
    degradation_df = case_study.generate_degradation_data(
        n_components=50, degradation_type='mixed'
    )
    
    # Exploratory analysis
    case_study.exploratory_degradation_analysis(degradation_df)
    
    # Prepare regression dataset
    X, y, feature_names = case_study.prepare_regression_dataset(degradation_df)
    
    # Compare regression models
    model_results = case_study.small_data_regression_comparison(X, y, feature_names)
    
    # Remaining Useful Life prediction
    rul_df, rul_model = case_study.remaining_useful_life_prediction(degradation_df)
    
    # Uncertainty quantification
    uncertainty_results = case_study.uncertainty_quantification_analysis(X, y)
    
    print("\n" + "="*60)
    print("DEGRADATION MODELING CASE STUDY COMPLETED")
    print("="*60)
    print("\nKey Takeaways for Small Data Degradation Modeling:")
    print("1. Use physics-informed features (temperature, load, stress interactions)")
    print("2. Gaussian Processes provide natural uncertainty quantification")
    print("3. Bayesian methods incorporate prior knowledge effectively")
    print("4. Time series cross-validation respects temporal dependencies")
    print("5. Bootstrap ensembles provide robust uncertainty estimates")
    print("6. Feature engineering is crucial for small degradation datasets")
    print("7. RUL prediction enables proactive maintenance strategies")
    
    print(f"\nRecommended approach for this dataset size:")
    print(f"  - Primary model: Gaussian Process or Bayesian Ridge")
    print(f"  - Uncertainty method: Bootstrap ensemble")
    print(f"  - Validation: Time series cross-validation")
    print(f"  - Key features: Stress interactions and degradation rate")

if __name__ == "__main__":
    main()