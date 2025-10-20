# Backpack Price Prediction - Kaggle Competition
Ranked 264th out of 3,393 participants (top 8%) with 38.66 RMSE

## Overview
This project implements a machine learning regression model to predict backpack prices based on various features such as brand, material, size, compartments, and other specifications. The solution uses CatBoost with GPU acceleration and achieves optimized performance through advanced feature engineering and cross-validation ensemble techniques.

## Dataset
- **Training Data**: `train.csv` + `training_extra.csv` (combined for more training samples)
- **Test Data**: `test.csv`
- **Sample Submission**: `sample_submission.csv`

### Features
- **Brand**: Manufacturer name (Nike, Adidas, Under Armour, Jansport, etc.)
- **Material**: Backpack material (Leather, Nylon, Canvas, etc.)
- **Size**: Size category (Small, Medium, Large)
- **Compartments**: Number of compartments
- **Weight Capacity (kg)**: Maximum weight capacity
- **Laptop Compartment**: Yes/No
- **Waterproof**: Yes/No
- **Style**: Design style (Backpack, Messenger, Tote, Duffle)
- **Color**: Backpack color
- **Price**: Target variable (to predict)

## Project Structure
```
.
├── data/
│   ├── train.csv
│   ├── training_extra.csv
│   ├── test.csv
│   └── sample_submission.csv
├── main.py                    # Main training script
├── submission.csv             # Generated predictions
└── README.md
```

## Feature Engineering
The model incorporates 30+ engineered features to capture complex pricing patterns:

### 1. Numeric Transformations
- **Size mapping**: Small→1, Medium→2, Large→3
- **Polynomial features**: Size², Weight²
- **Ratio features**: 
  - Compartments per Size
  - Weight per Compartment
  - Weight per Size
  - Compartments per Weight

### 2. Binary Indicators
- Waterproof (Yes/No → 1/0)
- Laptop Compartment (Yes/No → 1/0)
- Waterproof × Laptop interaction
- Size indicators (Is_Small, Is_Medium, Is_Large)

### 3. Material-Based Features
- `Is_Durable_Material`: Leather or Nylon
- `Is_Lightweight_Material`: Canvas or Nylon
- `Luxury_Material`: Leather only

### 4. Style-Based Features
- `Professional_Style`: Messenger or Tote
- `Casual_Style`: Backpack or Duffle

### 5. Brand-Based Features
- `Is_Premium_Brand`: Nike, Under Armour, Adidas
- `Is_Budget_Brand`: Jansport

### 6. Interaction Features
- Premium × Professional
- Premium × Durable
- Size × Compartments
- Brand_Material combinations
- Brand_Style combinations
- Material_Style combinations

### 7. Advanced Encoding
- **Categorical codes with weight**: For each categorical feature, creates `feature_code` × 100 + Weight Capacity
- **Target encoding**: Mean and standard deviation of price for each categorical-weight combination using 5-fold cross-validation to prevent leakage

## Model Architecture

### CatBoost Regressor
CatBoost was chosen for its superior handling of categorical features and built-in regularization.

**Hyperparameters:**
- `learning_rate`: 0.062
- `depth`: 8
- `iterations`: 5000
- `l2_leaf_reg`: 7
- `loss_function`: RMSE
- `early_stopping_rounds`: 200
- `task_type`: GPU (for faster training)

**Categorical Features Handled:**
- Brand, Material, Size, Compartments
- Laptop Compartment, Waterproof, Style, Color
- Combined features: Brand_Material, Brand_Style, Material_Style
- Weight Capacity (categorical version)

## Training Strategy

### 5-Fold Cross-Validation
- Splits data into 5 folds with shuffling (random_state=42)
- Each fold trains independently with early stopping on validation set
- Uses best model from each fold based on validation RMSE
- Final prediction is the ensemble average of all 5 fold predictions

### Advantages
- Reduces overfitting through validation-based early stopping
- Provides robust performance estimates
- Ensemble averaging smooths predictions and reduces variance
- GPU acceleration significantly speeds up training time

## Requirements

```bash
pip install pandas numpy scikit-learn catboost
```

**Note**: GPU support requires CUDA-compatible GPU and proper CatBoost GPU installation.

## Usage

### 1. Prepare Data
Place your data files in the `./data/` directory:
- `train.csv`
- `training_extra.csv`
- `test.csv`
- `sample_submission.csv`

### 2. Run Training
```bash
python main.py
```

### 3. Output
- Console displays fold-by-fold RMSE scores and cross-validation results
- `submission.csv` is generated with predictions for the test set

## Results

The model outputs:
- Individual fold RMSE scores
- Cross-validated RMSE mean ± standard deviation
- Final ensemble predictions averaged across all folds

Example output:
```
Cross-validated RMSE: X.XXXX +/- X.XXXX
```

## Key Techniques

1. **Feature Engineering**: 30+ features capturing non-linear relationships
2. **Target Encoding**: Cross-validated to prevent information leakage
3. **Ensemble Learning**: 5-fold predictions averaged for robustness
4. **GPU Acceleration**: Faster training with CatBoost GPU support
5. **Early Stopping**: Prevents overfitting by monitoring validation loss