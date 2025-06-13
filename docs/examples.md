# Examples

This document provides complete, real-world examples of using AsymmeTree for various imbalanced classification tasks.

## Fraud Detection

Credit card fraud detection with transaction data.

### Dataset Setup

```python
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample fraud detection dataset
np.random.seed(42)
n_samples = 50000

# Generate realistic transaction features
data = {
    'transaction_amount': np.random.lognormal(3, 1.5, n_samples),
    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'atm'], n_samples),
    'hour_of_day': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'days_since_last_transaction': np.random.exponential(2, n_samples),
    'account_age_days': np.random.normal(365*2, 365, n_samples),
    'customer_risk_score': np.random.beta(2, 5, n_samples),
    'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

X = pd.DataFrame(data)
X['account_age_days'] = X['account_age_days'].clip(lower=30)

# Create fraud labels (1% fraud rate)
fraud_probability = (
    0.001 +  # Base rate
    0.01 * (X['transaction_amount'] > 1000) +  # Large amounts
    0.005 * (X['hour_of_day'].isin([2, 3, 4])) +  # Late night
    0.008 * (X['customer_risk_score'] > 0.8) +  # High risk customers
    0.003 * X['is_weekend']  # Weekend transactions
)

y = np.random.binomial(1, fraud_probability)

print(f"Dataset shape: {X.shape}")
print(f"Fraud rate: {y.mean():.2%}")
print(f"Total fraudulent transactions: {y.sum()}")
```

### Model Training

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Configure for fraud detection
fraud_tree = AsymmeTree(
    max_depth=6,
    sorted_by='f_score',  # Optimize for F-score
    node_min_recall=0.02,  # Capture at least 2% of fraud
    leaf_min_precision=0.05,  # 5% precision threshold
    verbose=True
)

# Train with custom metrics
def avg_fraud_amount(data):
    return data['transaction_amount'].mean()

def peak_hour_pct(data):
    return (data['hour_of_day'].isin([2, 3, 4])).mean()

extra_data = pd.DataFrame({
    'transaction_amount': X_train['transaction_amount'],
    'hour_of_day': X_train['hour_of_day']
}, index=X_train.index)

fraud_tree.fit(
    X_train, y_train,
    cat_features=['merchant_category'],
    gt_only_features=['transaction_amount', 'account_age_days'],
    pinned_features=['customer_risk_score'],  # Always consider risk score
    extra_metrics={
        'avg_amount': avg_fraud_amount,
        'peak_hour_rate': peak_hour_pct
    },
    extra_metrics_data=extra_data,
    auto=True
)
```

### Evaluation and Deployment

```python
# Evaluate performance
predictions = fraud_tree.predict(X_test)
fraud_tree.performance()

# Print interpretable rules
print("Fraud Detection Rules:")
fraud_tree.print(show_metrics=True)

# Export for production database
sql_rules = fraud_tree.to_sql()
print("\nSQL Rules for Production:")
print(sql_rules)

# Save model
fraud_tree.save('fraud_detection_model.json')
```

## Medical Diagnosis

Rare disease prediction from patient symptoms and lab results.

### Dataset and Model

```python
# Medical dataset simulation
np.random.seed(123)
n_patients = 20000

medical_data = {
    'age': np.random.normal(55, 15, n_patients).clip(18, 90),
    'bmi': np.random.normal(26, 4, n_patients).clip(15, 45),
    'blood_pressure_systolic': np.random.normal(125, 20, n_patients),
    'cholesterol_level': np.random.normal(200, 40, n_patients),
    'blood_sugar': np.random.normal(95, 15, n_patients),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'smoking_status': np.random.choice(['never', 'former', 'current'], n_patients, p=[0.5, 0.3, 0.2]),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'exercise_hours_per_week': np.random.exponential(3, n_patients)
}

X_med = pd.DataFrame(medical_data)

# Rare disease (0.5% prevalence)
disease_risk = (
    0.001 +  # Base rate
    0.01 * (X_med['age'] > 65) +
    0.008 * (X_med['blood_pressure_systolic'] > 140) +
    0.005 * (X_med['cholesterol_level'] > 240) +
    0.003 * (X_med['smoking_status'] == 'current') +
    0.004 * X_med['family_history']
)

y_med = np.random.binomial(1, disease_risk)

print(f"Disease prevalence: {y_med.mean():.2%}")

# Medical-specific configuration
medical_tree = AsymmeTree(
    max_depth=4,  # Keep rules simple for medical interpretation
    sorted_by='f_score',
    node_min_recall=0.03,  # Don't miss too many cases
    leaf_min_precision=0.10,  # Balance false positives
    show_metrics=True,
    verbose=True
)

# Train with medical metrics
def avg_patient_age(data):
    return data['age'].mean()

def high_risk_pct(data):
    return ((data['age'] > 65) | (data['blood_pressure_systolic'] > 140)).mean()

med_extra_data = pd.DataFrame({
    'age': X_med['age'],
    'blood_pressure_systolic': X_med['blood_pressure_systolic']
}, index=X_med.index)

medical_tree.fit(
    X_med, y_med,
    cat_features=['gender', 'smoking_status'],
    extra_metrics={
        'avg_age': avg_patient_age,
        'high_risk_rate': high_risk_pct
    },
    extra_metrics_data=med_extra_data,
    auto=True
)

# Medical decision rules
print("Medical Decision Tree Rules:")
medical_tree.print(show_metrics=True)
```

## Quality Control

Manufacturing defect detection from sensor data.

### Manufacturing Example

```python
# Manufacturing quality control dataset
np.random.seed(456)
n_products = 30000

qc_data = {
    'temperature': np.random.normal(75, 5, n_products),
    'pressure': np.random.normal(100, 10, n_products),
    'humidity': np.random.normal(45, 8, n_products),
    'vibration_level': np.random.exponential(2, n_products),
    'machine_id': np.random.choice([f'M{i:03d}' for i in range(10)], n_products),
    'shift': np.random.choice(['A', 'B', 'C'], n_products),
    'operator_experience': np.random.uniform(0.5, 10, n_products),
    'material_batch': np.random.choice([f'B{i:04d}' for i in range(50)], n_products),
    'production_speed': np.random.normal(100, 15, n_products)
}

X_qc = pd.DataFrame(qc_data)

# Defect rate (2% with specific patterns)
defect_prob = (
    0.005 +  # Base defect rate
    0.02 * (X_qc['temperature'] > 85) +  # High temperature
    0.015 * (X_qc['pressure'] > 120) +  # High pressure
    0.01 * (X_qc['vibration_level'] > 5) +  # High vibration
    0.008 * (X_qc['shift'] == 'C') +  # Night shift issues
    0.005 * (X_qc['operator_experience'] < 2)  # Inexperienced operators
)

y_qc = np.random.binomial(1, defect_prob)

print(f"Defect rate: {y_qc.mean():.2%}")

# Quality control tree
qc_tree = AsymmeTree(
    max_depth=5,
    sorted_by='igr',  # Information gain ratio for quality control
    node_min_recall=0.03,
    leaf_min_precision=0.08,
    num_bin=20,  # Fine-grained numerical analysis
    ignore_null=False,  # Include nulls as separate category
    verbose=True
)

# Train with production metrics
def avg_production_speed(data):
    return data['production_speed'].mean()

def night_shift_rate(data):
    return (data['shift'] == 'C').mean()

qc_extra_data = pd.DataFrame({
    'production_speed': X_qc['production_speed'],
    'shift': X_qc['shift']
}, index=X_qc.index)

qc_tree.fit(
    X_qc, y_qc,
    cat_features=['machine_id', 'shift', 'material_batch'],
    extra_metrics={
        'avg_speed': avg_production_speed,
        'night_shift_pct': night_shift_rate
    },
    extra_metrics_data=qc_extra_data,
    auto=True
)

print("Quality Control Rules:")
qc_tree.print(show_metrics=True)
```

## Interactive Marketing Campaign

Customer conversion prediction with domain expert input.

### Interactive Campaign Optimization

```python
# Marketing campaign dataset
np.random.seed(789)
n_customers = 25000

campaign_data = {
    'age': np.random.normal(42, 12, n_customers).clip(18, 80),
    'income': np.random.lognormal(10.5, 0.8, n_customers),
    'previous_purchases': np.random.poisson(3, n_customers),
    'days_since_last_purchase': np.random.exponential(30, n_customers),
    'email_opens_last_month': np.random.poisson(2, n_customers),
    'website_visits_last_week': np.random.poisson(1.5, n_customers),
    'customer_segment': np.random.choice(['premium', 'standard', 'basic'], n_customers, p=[0.2, 0.5, 0.3]),
    'geographic_region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
    'preferred_channel': np.random.choice(['email', 'phone', 'social'], n_customers, p=[0.5, 0.3, 0.2])
}

X_campaign = pd.DataFrame(campaign_data)

# Conversion probability (3% base rate)
conversion_prob = (
    0.01 +  # Base conversion rate
    0.05 * (X_campaign['customer_segment'] == 'premium') +
    0.02 * (X_campaign['previous_purchases'] > 5) +
    0.015 * (X_campaign['email_opens_last_month'] > 3) +
    0.01 * (X_campaign['income'] > 75000) +
    0.005 * (X_campaign['website_visits_last_week'] > 2)
)

y_campaign = np.random.binomial(1, conversion_prob)

print(f"Conversion rate: {y_campaign.mean():.2%}")

# Interactive campaign tree
campaign_tree = AsymmeTree(
    max_depth=4,
    sorted_by='f_score',
    node_min_recall=0.05,
    leaf_min_precision=0.12,
    verbose=True
)

# Revenue-based custom metrics
def avg_customer_value(data):
    return data['income'].mean() * 0.1  # Assume 10% income as customer value

def premium_customer_rate(data):
    return (data['customer_segment'] == 'premium').mean()

campaign_extra_data = pd.DataFrame({
    'income': X_campaign['income'],
    'customer_segment': X_campaign['customer_segment']
}, index=X_campaign.index)

print("Starting Interactive Campaign Tree Building...")
print("This will guide you through building a conversion prediction tree.")
print("Consider business constraints when making decisions.\n")

# Interactive training
campaign_tree.fit(
    X_campaign, y_campaign,
    cat_features=['customer_segment', 'geographic_region', 'preferred_channel'],
    pinned_features=['previous_purchases', 'email_opens_last_month'],
    extra_metrics={
        'avg_customer_value': avg_customer_value,
        'premium_rate': premium_customer_rate
    },
    extra_metrics_data=campaign_extra_data,
    auto=False  # Interactive mode
)

# Results
print("\nFinal Campaign Targeting Rules:")
campaign_tree.print(show_metrics=True)

# Business deployment
sql_targeting = campaign_tree.to_sql()
print(f"\nSQL for Campaign Targeting:\n{sql_targeting}")
```

## Cross-Validation Example

Proper validation for imbalanced datasets.

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

def cross_validate_asymmetree(X, y, config, n_splits=5):
    """
    Cross-validate AsymmeTree with proper stratification for imbalanced data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'trees': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train tree
        tree = AsymmeTree(**config)
        tree.fit(X_train, y_train, cat_features=['merchant_category'], auto=True)
        
        # Predict and evaluate
        y_pred = tree.predict(X_val)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['trees'].append(tree)
        
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Summary statistics
    print(f"\nCross-Validation Results:")
    print(f"Precision: {np.mean(results['precision']):.3f} ± {np.std(results['precision']):.3f}")
    print(f"Recall: {np.mean(results['recall']):.3f} ± {np.std(results['recall']):.3f}")
    print(f"F1-Score: {np.mean(results['f1_score']):.3f} ± {np.std(results['f1_score']):.3f}")
    
    return results

# Example usage with fraud detection data
config = {
    'max_depth': 5,
    'sorted_by': 'f_score',
    'node_min_recall': 0.02,
    'leaf_min_precision': 0.05
}

cv_results = cross_validate_asymmetree(X, y, config, n_splits=5)

# Select best tree
best_idx = np.argmax(cv_results['f1_score'])
best_tree = cv_results['trees'][best_idx]

print(f"\nBest Tree (Fold {best_idx + 1}):")
best_tree.print(show_metrics=True)
```

## Hyperparameter Tuning

Grid search for optimal parameters.

### Parameter Optimization

```python
from itertools import product

def grid_search_asymmetree(X, y, param_grid, cv_folds=3, scoring='f1'):
    """
    Grid search for AsymmeTree hyperparameters.
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = -1
    best_params = None
    best_tree = None
    results = []
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Cross-validate this parameter combination
        cv_results = cross_validate_asymmetree(X, y, params, cv_folds)
        
        if scoring == 'f1':
            score = np.mean(cv_results['f1_score'])
        elif scoring == 'precision':
            score = np.mean(cv_results['precision'])
        elif scoring == 'recall':
            score = np.mean(cv_results['recall'])
        
        results.append({
            'params': params,
            'score': score,
            'std': np.std(cv_results[scoring.replace('f1', 'f1_score')]),
            'cv_results': cv_results
        })
        
        if score > best_score:
            best_score = score
            best_params = params
            best_tree = cv_results['trees'][np.argmax(cv_results[scoring.replace('f1', 'f1_score')])]
        
        print(f"  Score: {score:.3f}\n")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_tree': best_tree,
        'all_results': results
    }

# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'node_min_recall': [0.01, 0.02, 0.05],
    'leaf_min_precision': [0.05, 0.10, 0.15],
    'sorted_by': ['f_score', 'igr']
}

# Run grid search (this will take some time)
print("Starting grid search for optimal parameters...")
search_results = grid_search_asymmetree(X, y, param_grid, cv_folds=3, scoring='f1')

print(f"Best parameters: {search_results['best_params']}")
print(f"Best F1-score: {search_results['best_score']:.3f}")

# Train final model with best parameters
final_tree = AsymmeTree(**search_results['best_params'])
final_tree.fit(X, y, auto=True)

print("\nFinal Optimized Tree:")
final_tree.print(show_metrics=True)
```

These examples demonstrate AsymmeTree's versatility across different domains and use cases. Each example can be adapted to your specific dataset and business requirements. 