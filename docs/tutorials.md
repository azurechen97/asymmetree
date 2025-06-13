# Tutorials

Step-by-step tutorials for mastering AsymmeTree in different scenarios.

## Tutorial 1: Fraud Detection from Scratch

Learn to build a complete fraud detection system using AsymmeTree.

### Step 1: Understanding the Problem

Fraud detection is a classic imbalanced classification problem where:
- Fraudulent transactions are rare (< 1% typically)
- False positives are costly (blocking legitimate transactions)
- False negatives are very costly (allowing fraud)
- Business rules need to be interpretable and auditable

### Step 2: Data Preparation

```python
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree
import matplotlib.pyplot as plt

# Load sample transaction data
np.random.seed(42)
n_transactions = 100000

# Create realistic transaction features
data = {
    'amount': np.random.lognormal(mean=3.5, sigma=1.2, size=n_transactions),
    'merchant_category': np.random.choice([
        'grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'
    ], size=n_transactions, p=[0.3, 0.15, 0.2, 0.15, 0.15, 0.05]),
    'hour': np.random.randint(0, 24, n_transactions),
    'day_of_week': np.random.randint(0, 7, n_transactions),
    'is_weekend': lambda x: np.where(x >= 5, 1, 0),
    'time_since_last': np.random.exponential(24, n_transactions),  # hours
    'account_age': np.random.normal(365*3, 365, n_transactions),   # days
    'previous_declines': np.random.poisson(0.1, n_transactions),
    'international': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
}

df = pd.DataFrame(data)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['account_age'] = df['account_age'].clip(lower=30)

# Create fraud labels based on realistic patterns
fraud_prob = (
    0.001 +  # Base fraud rate
    0.02 * (df['amount'] > 500) +  # Large transactions
    0.01 * (df['hour'].isin([2, 3, 4, 23])) +  # Unusual hours
    0.005 * df['international'] +  # International transactions
    0.003 * df['is_weekend'] +  # Weekend transactions
    0.01 * (df['previous_declines'] > 0) +  # History of declines
    0.005 * (df['time_since_last'] < 1)  # Rapid successive transactions
)

df['is_fraud'] = np.random.binomial(1, fraud_prob)

print(f"Dataset size: {len(df):,}")
print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
print(f"Total fraud cases: {df['is_fraud'].sum():,}")
```

### Step 3: Exploratory Data Analysis

```python
# Analyze fraud patterns
print("Fraud Analysis:")
print("-" * 40)

# Amount analysis
fraud_amounts = df[df['is_fraud'] == 1]['amount']
normal_amounts = df[df['is_fraud'] == 0]['amount']

print(f"Avg fraud amount: ${fraud_amounts.mean():.2f}")
print(f"Avg normal amount: ${normal_amounts.mean():.2f}")

# Time patterns
fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean'])
print(f"\nHour with highest fraud rate: {fraud_by_hour['mean'].idxmax()}")

# Merchant category patterns
fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean'])
print(f"Merchant category with highest fraud rate: {fraud_by_merchant['mean'].idxmax()}")
```

### Step 4: Basic Model Training

```python
from sklearn.model_selection import train_test_split

# Prepare features and target
features = ['amount', 'merchant_category', 'hour', 'day_of_week', 
           'is_weekend', 'time_since_last', 'account_age', 
           'previous_declines', 'international']

X = df[features]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} transactions")
print(f"Test set: {X_test.shape[0]:,} transactions")
print(f"Training fraud rate: {y_train.mean():.3%}")

# Initialize AsymmeTree for fraud detection
fraud_detector = AsymmeTree(
    max_depth=5,                    # Reasonable depth for interpretability
    sorted_by='f_score',           # Optimize for F-score
    node_min_recall=0.02,          # Capture at least 2% of fraud
    leaf_min_precision=0.05,       # 5% minimum precision for fraud prediction
    verbose=True                   # Show training progress
)

# Train the model
print("\nTraining fraud detection model...")
fraud_detector.fit(
    X_train, y_train,
    cat_features=['merchant_category'],  # Specify categorical features
    gt_only_features=['amount', 'account_age'],  # These should only use >= splits
    auto=True
)

print("\nTraining completed!")
```

### Step 5: Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions
y_pred = fraud_detector.predict(X_test)

# Performance metrics
print("Fraud Detection Performance:")
print("=" * 50)
fraud_detector.performance()

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives: {cm[1,1]:,}")

# Business impact analysis
false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1])

print(f"\nBusiness Impact:")
print(f"False Positive Rate: {false_positive_rate:.2%} (blocked legitimate transactions)")
print(f"False Negative Rate: {false_negative_rate:.2%} (missed fraud)")
```

### Step 6: Rule Interpretation

```python
# Display the decision tree rules
print("\nFraud Detection Rules:")
print("=" * 50)
fraud_detector.print(show_metrics=True)

# Export SQL rules for database deployment
sql_rules = fraud_detector.to_sql()
print(f"\nSQL Rules for Production Deployment:")
print(f"SELECT transaction_id, amount, merchant_category")
print(f"FROM transactions")
print(f"WHERE {sql_rules};")

# Save the model
fraud_detector.save('fraud_detection_model.json')
print("\nModel saved as 'fraud_detection_model.json'")
```

### Step 7: Production Deployment

```python
# Example of loading and using the model in production
production_model = AsymmeTree()
production_model.load('fraud_detection_model.json')

# Score new transactions
new_transactions = X_test.head(10)  # Sample new transactions
fraud_scores = production_model.predict(new_transactions)

print("Production Scoring Example:")
for i, (idx, row) in enumerate(new_transactions.iterrows()):
    score = fraud_scores[i]
    print(f"Transaction {i+1}: ${row['amount']:.2f} at {row['merchant_category']} -> {'FRAUD' if score else 'NORMAL'}")
```

## Tutorial 2: Interactive Medical Diagnosis

Build a medical diagnosis system using interactive mode to incorporate domain expertise.

### Step 1: Medical Dataset Setup

```python
# Simulate medical diagnosis dataset
np.random.seed(123)
n_patients = 15000

# Patient demographics and vital signs
medical_data = {
    'age': np.random.normal(55, 15, n_patients).clip(18, 90),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'bmi': np.random.normal(26, 4, n_patients).clip(15, 45),
    'systolic_bp': np.random.normal(125, 20, n_patients),
    'diastolic_bp': np.random.normal(80, 12, n_patients),
    'heart_rate': np.random.normal(72, 12, n_patients),
    'cholesterol': np.random.normal(200, 40, n_patients),
    'blood_sugar': np.random.normal(95, 15, n_patients),
    'smoking': np.random.choice(['never', 'former', 'current'], n_patients, p=[0.5, 0.3, 0.2]),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
    'exercise_hours': np.random.exponential(3, n_patients),
    'symptoms_duration': np.random.exponential(14, n_patients)  # days
}

medical_df = pd.DataFrame(medical_data)

# Create disease probability based on medical knowledge
# Simulating cardiovascular disease diagnosis
disease_risk = (
    0.005 +  # Base rate
    0.02 * (medical_df['age'] > 65) +  # Age factor
    0.015 * (medical_df['systolic_bp'] > 140) +  # Hypertension
    0.01 * (medical_df['cholesterol'] > 240) +  # High cholesterol
    0.008 * (medical_df['bmi'] > 30) +  # Obesity
    0.005 * (medical_df['smoking'] == 'current') +  # Current smoking
    0.003 * medical_df['family_history'] +  # Family history
    0.004 * (medical_df['blood_sugar'] > 125)  # Diabetes indicator
)

medical_df['has_disease'] = np.random.binomial(1, disease_risk)

print(f"Medical dataset: {len(medical_df):,} patients")
print(f"Disease prevalence: {medical_df['has_disease'].mean():.2%}")
```

### Step 2: Interactive Model Building

```python
# Prepare for interactive training
X_med = medical_df.drop('has_disease', axis=1)
y_med = medical_df['has_disease']

# Medical-specific configuration
medical_tree = AsymmeTree(
    max_depth=4,  # Keep medical rules simple
    sorted_by='f_score',
    node_min_recall=0.05,  # Don't miss too many cases
    leaf_min_precision=0.08,  # Balance false positives
    verbose=True,
    show_metrics=True
)

print("Interactive Medical Diagnosis Model")
print("=" * 50)
print("You'll be guided through building a diagnosis tree.")
print("Consider medical best practices when making decisions.")
print("\nAt each step, you can:")
print("- Choose a feature number to split on")
print("- Type 'c' to continue automatically")
print("- Type 'q' to quit")
print("- Type 's' to skip current node")

# Start interactive training
medical_tree.fit(X_med, y_med, cat_features=['gender', 'smoking'], auto=False)
```

### Step 3: Medical Rule Validation

```python
# After interactive training, validate the medical rules
print("\nMedical Decision Rules:")
print("=" * 50)
medical_tree.print(show_metrics=True)

# Evaluate performance
predictions = medical_tree.predict(X_med)
medical_tree.performance()

# Medical-specific evaluation
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# For medical diagnosis, we want to understand the precision-recall tradeoff
precision, recall, thresholds = precision_recall_curve(y_med, predictions)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Medical Diagnosis: Precision-Recall Analysis')
plt.grid(True)
plt.show()

print(f"\nMedical Performance Summary:")
print(f"Sensitivity (Recall): {recall[-2]:.2%}")  # Last point before threshold=1
print(f"Positive Predictive Value (Precision): {precision[-2]:.2%}")
```

### Step 4: Clinical Rule Export

```python
# Export rules for clinical decision support system
clinical_rules = medical_tree.to_sql()
print("\nClinical Decision Rules (SQL format):")
print("=" * 60)
print(f"SELECT patient_id, age, gender, systolic_bp, cholesterol")
print(f"FROM patient_records")
print(f"WHERE {clinical_rules}")
print(f"ORDER BY age DESC;")

# Save medical model
medical_tree.save('medical_diagnosis_model.json')
print("\nMedical model saved for clinical deployment.")
```

## Tutorial 3: Quality Control Optimization

Optimize manufacturing quality control using AsymmeTree.

### Step 1: Manufacturing Data Simulation

```python
# Simulate manufacturing sensor data
np.random.seed(456)
n_products = 50000

manufacturing_data = {
    'temperature': np.random.normal(75, 5, n_products),
    'pressure': np.random.normal(100, 10, n_products),
    'humidity': np.random.normal(45, 8, n_products),
    'vibration': np.random.exponential(2, n_products),
    'machine_id': np.random.choice([f'M{i:03d}' for i in range(20)], n_products),
    'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], n_products, p=[0.4, 0.4, 0.2]),
    'operator_id': np.random.choice([f'OP{i:02d}' for i in range(50)], n_products),
    'material_batch': np.random.choice([f'B{i:04d}' for i in range(100)], n_products),
    'production_speed': np.random.normal(100, 15, n_products),
    'ambient_temp': np.random.normal(22, 3, n_products)
}

qc_df = pd.DataFrame(manufacturing_data)

# Create defect probability based on manufacturing knowledge
defect_prob = (
    0.01 +  # Base defect rate
    0.03 * (qc_df['temperature'] > 85) +  # Overheating
    0.025 * (qc_df['pressure'] > 120) +  # High pressure
    0.02 * (qc_df['vibration'] > 5) +  # Excessive vibration
    0.015 * (qc_df['shift'] == 'Night') +  # Night shift challenges
    0.01 * (qc_df['production_speed'] > 120) +  # High speed issues
    0.008 * (qc_df['humidity'] > 60)  # High humidity
)

qc_df['is_defective'] = np.random.binomial(1, defect_prob)

print(f"Manufacturing dataset: {len(qc_df):,} products")
print(f"Defect rate: {qc_df['is_defective'].mean():.2%}")
```

### Step 2: Quality Control Model

```python
# Prepare quality control features
X_qc = qc_df.drop('is_defective', axis=1)
y_qc = qc_df['is_defective']

# Quality control specific configuration
qc_tree = AsymmeTree(
    max_depth=6,  # More detailed rules for manufacturing
    sorted_by='igr',  # Information gain ratio works well for QC
    node_min_recall=0.03,
    leaf_min_precision=0.08,
    num_bin=30,  # Fine-grained analysis of sensor readings
    ignore_null=False,  # Missing sensor readings are important
    verbose=True
)

# Add custom metrics for manufacturing context
def avg_production_speed(data):
    return data['production_speed'].mean()

def night_shift_percentage(data):
    return (data['shift'] == 'Night').mean() * 100

def high_temp_percentage(data):
    return (data['temperature'] > 80).mean() * 100

# Extra data for custom metrics
qc_extra_data = pd.DataFrame({
    'production_speed': X_qc['production_speed'],
    'shift': X_qc['shift'],
    'temperature': X_qc['temperature']
}, index=X_qc.index)

# Train quality control model
qc_tree.fit(
    X_qc, y_qc,
    cat_features=['machine_id', 'shift', 'operator_id', 'material_batch'],
    extra_metrics={
        'avg_speed': avg_production_speed,
        'night_shift_pct': night_shift_percentage,
        'high_temp_pct': high_temp_percentage
    },
    extra_metrics_data=qc_extra_data,
    auto=True
)

print("\nQuality Control Rules:")
qc_tree.print(show_metrics=True)
```

### Step 3: Manufacturing Insights

```python
# Analyze quality control insights
qc_predictions = qc_tree.predict(X_qc)
qc_tree.performance()

# Export rules for manufacturing system
manufacturing_rules = qc_tree.to_sql()
print("\nManufacturing Quality Control Rules:")
print("=" * 60)
print("-- Use this query to flag potentially defective products")
print(f"SELECT product_id, machine_id, shift, temperature, pressure")
print(f"FROM production_data")
print(f"WHERE {manufacturing_rules}")
print(f"ORDER BY temperature DESC, pressure DESC;")

# Save quality control model
qc_tree.save('quality_control_model.json')
```

### Step 4: Real-time Quality Monitoring

```python
# Simulate real-time quality monitoring
def quality_monitor(new_batch_data, model_path='quality_control_model.json'):
    """
    Monitor new production batch for quality issues
    """
    # Load trained model
    qc_model = AsymmeTree()
    qc_model.load(model_path)
    
    # Predict quality for new batch
    quality_predictions = qc_model.predict(new_batch_data)
    
    # Generate alerts
    defect_indices = np.where(quality_predictions == 1)[0]
    
    print(f"Quality Monitoring Report")
    print(f"=" * 40)
    print(f"Batch size: {len(new_batch_data)}")
    print(f"Predicted defects: {len(defect_indices)}")
    print(f"Defect rate: {len(defect_indices)/len(new_batch_data):.2%}")
    
    if len(defect_indices) > 0:
        print(f"\nDefective Products Alert:")
        for idx in defect_indices[:5]:  # Show first 5
            row = new_batch_data.iloc[idx]
            print(f"Product {idx}: Machine {row['machine_id']}, "
                  f"Temp: {row['temperature']:.1f}°C, "
                  f"Pressure: {row['pressure']:.1f}")
    
    return quality_predictions

# Test with new batch
new_batch = X_qc.sample(1000).reset_index(drop=True)
quality_monitor(new_batch)
```

## Tutorial 4: Model Validation and Hyperparameter Tuning

Learn proper validation techniques for imbalanced datasets.

### Step 1: Cross-Validation Setup

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def asymmetree_cross_validation(X, y, param_combinations, cv_folds=5):
    """
    Comprehensive cross-validation for AsymmeTree
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter set {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        fold_results = {
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
                    # Train model with current parameters
        tree = AsymmeTree(**params)
        tree.fit(X_train, y_train, auto=True)
            
            # Evaluate on validation fold
            y_pred = tree.predict(X_val)
            
            # Calculate metrics (handle edge cases)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            fold_results['precision'].append(precision)
            fold_results['recall'].append(recall)
            fold_results['f1_score'].append(f1)
        
        # Calculate mean and std for this parameter set
        result = {
            'params': params,
            'precision_mean': np.mean(fold_results['precision']),
            'precision_std': np.std(fold_results['precision']),
            'recall_mean': np.mean(fold_results['recall']),
            'recall_std': np.std(fold_results['recall']),
            'f1_mean': np.mean(fold_results['f1_score']),
            'f1_std': np.std(fold_results['f1_score'])
        }
        
        results.append(result)
        
        print(f"F1-Score: {result['f1_mean']:.3f} ± {result['f1_std']:.3f}")
    
    return results

# Define parameter grid for tuning
param_grid = [
    {'max_depth': 3, 'node_min_recall': 0.01, 'leaf_min_precision': 0.05, 'sorted_by': 'f_score'},
    {'max_depth': 4, 'node_min_recall': 0.01, 'leaf_min_precision': 0.05, 'sorted_by': 'f_score'},
    {'max_depth': 5, 'node_min_recall': 0.01, 'leaf_min_precision': 0.05, 'sorted_by': 'f_score'},
    {'max_depth': 4, 'node_min_recall': 0.02, 'leaf_min_precision': 0.05, 'sorted_by': 'f_score'},
    {'max_depth': 4, 'node_min_recall': 0.01, 'leaf_min_precision': 0.10, 'sorted_by': 'f_score'},
    {'max_depth': 4, 'node_min_recall': 0.01, 'leaf_min_precision': 0.05, 'sorted_by': 'igr'},
]

# Run cross-validation (using fraud data from Tutorial 1)
cv_results = asymmetree_cross_validation(X_train, y_train, param_grid, cv_folds=5)
```

### Step 2: Results Analysis

```python
# Analyze cross-validation results
results_df = pd.DataFrame(cv_results)

# Sort by F1-score
results_df = results_df.sort_values('f1_mean', ascending=False)

print("Cross-Validation Results (ranked by F1-score):")
print("=" * 80)

for i, row in results_df.iterrows():
    print(f"Rank {len(results_df) - i}:")
    print(f"  Parameters: {row['params']}")
    print(f"  F1-Score: {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")
    print(f"  Precision: {row['precision_mean']:.3f} ± {row['precision_std']:.3f}")
    print(f"  Recall: {row['recall_mean']:.3f} ± {row['recall_std']:.3f}")
    print()

# Best parameters
best_params = results_df.iloc[0]['params']
print(f"Best parameters: {best_params}")
```

### Step 3: Final Model Training

```python
# Train final model with best parameters
final_model = AsymmeTree(**best_params)
final_model.fit(X_train, y_train, auto=True)

# Evaluate on test set
test_predictions = final_model.predict(X_test)
final_model.performance()

print("\nFinal Model Rules:")
final_model.print(show_metrics=True)

# Save the optimized model
final_model.save('optimized_fraud_model.json')
```

These tutorials provide comprehensive, hands-on experience with AsymmeTree across different domains. Each tutorial builds progressively more advanced skills while demonstrating real-world applications. 