import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

###########################################
# 1. Data Loading
###########################################
print("---------- Loading Data ----------")
df_train = pd.read_csv('./data/train.csv')
df_train_ex = pd.read_csv('./data/training_extra.csv')
df_test  = pd.read_csv('./data/test.csv')
df_sub = pd.read_csv('./data/sample_submission.csv')

# Combine training sets.
df_train = pd.concat([df_train_ex, df_train], axis=0).reset_index(drop=True)
print("Combined training shape:", df_train.shape)

# Remove the 'id' column.
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)

###########################################
# 2. Base Feature Engineering
###########################################
def feature_engineering(df):
    # Map Size to numeric
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    
    # Ratio features
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
    
    # Map binary features
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    
    # Material-based features
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    
    # Style-based features
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    
    # Brand-based features
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    
    # Size indicators
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)
    
    # Polynomial features
    df['Size_Squared'] = df['Size_Num'] ** 2
    df['Weight_Squared'] = df['Weight Capacity (kg)'] ** 2

    # Interaction features
    df['Premium_Professional'] = df['Is_Premium_Brand'] * df['Professional_Style']
    df['Premium_Durable'] = df['Is_Premium_Brand'] * df['Is_Durable_Material']
    df['Size_x_Compartments'] = df['Size_Num'] * df['Compartments']
    df['Weight_per_Size'] = df['Weight Capacity (kg)'] / df['Size_Num']
    df['Compartments_per_Weight'] = df['Compartments'] / (df['Weight Capacity (kg)'] + 1e-5)

    # Combined categorical features
    df['Brand_Material'] = df['Brand'].astype(str) + "_" + df['Material'].astype(str)
    df['Brand_Style'] = df['Brand'].astype(str) + "_" + df['Style'].astype(str)
    df['Material_Style'] = df['Material'].astype(str) + "_" + df['Style'].astype(str)
    
    return df

print("\n---------- Feature Engineering ----------")
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

###########################################
# 3. Missing Values & Categorical Conversion
###########################################
print("\n---------- Handling Missing Values & Categoricals ----------")
# Original categorical columns.
cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color']
# Combined categorical columns.
combined_cat_cols = ['Brand_Material', 'Brand_Style', 'Material_Style']
print("Categorical columns:", cat_cols + combined_cat_cols)
for col in cat_cols + combined_cat_cols:
    df_train[col] = df_train[col].fillna('None').astype('string').astype('category')
    df_test[col] = df_test[col].fillna('None').astype('string').astype('category')
print("Categorical columns converted.")
# Ensure numerical column is filled.
median_weight = df_train['Weight Capacity (kg)'].median()
df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')
df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(median_weight).astype('float64')
print("Numerical column filled.")
# Create a categorical version of weight capacity.
df_train['Weight Capacity (kg) categorical'] = df_train['Weight Capacity (kg)'].astype('string').astype('category')
df_test['Weight Capacity (kg) categorical'] = df_test['Weight Capacity (kg)'].astype('string').astype('category')
print("Weight Capacity categorical column created.")
###########################################
# 4. Additional Encoding: “_wc” & Target Encoding
###########################################
# For each original categorical feature, create a numeric code and then combine with weight.
print("\n---------- Additional Encoding ----------")
for col in cat_cols:
    df_train[col + "_code"] = df_train[col].cat.codes
    df_test[col + "_code"] = df_test[col].cat.codes
    df_train[col + "_wc"] = df_train[col + "_code"] * 100 + df_train['Weight Capacity (kg)']
    df_test[col + "_wc"] = df_test[col + "_code"] * 100 + df_test['Weight Capacity (kg)']

# Define a helper function to add target-encoded features using groupby aggregations.
def add_target_encoding(train_df, test_df, feature, target, stats=["mean", "std"], n_splits=5):
    te_mean = f"TE_{feature}_mean"
    te_std = f"TE_{feature}_std"
    train_df[te_mean] = np.nan
    train_df[te_std] = np.nan
    kf_te = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf_te.split(train_df):
        X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        agg = X_tr.groupby(feature)[target].agg(stats)
        X_val = X_val.merge(agg, left_on=feature, right_index=True, how='left')
        train_df.loc[val_idx, te_mean] = X_val['mean'].values
        train_df.loc[val_idx, te_std] = X_val['std'].values
        
    # For test set, compute using full training data.
    agg_full = train_df.groupby(feature)[target].agg(stats)
    test_df = test_df.merge(agg_full, left_on=feature, right_index=True, how='left')
    test_df.rename(columns={'mean': te_mean, 'std': te_std}, inplace=True)
    return train_df, test_df

# Apply target encoding on each new “_wc” feature.
wc_features = [col + "_wc" for col in cat_cols]
for feat in wc_features:
    df_train, df_test = add_target_encoding(df_train, df_test, feat, target='Price', stats=["mean", "std"], n_splits=5)

###########################################
# 5. Preparing Data for Regression
###########################################
print("\n---------- Preparing Data ----------")
y = df_train['Price']
# Drop target column from training features.
X = df_train.drop(columns=['Price'])
X_test = df_test.copy()

print("Training data shape:", X.shape)
print("Test data shape:", X_test.shape)

# List of categorical features for CatBoost.
# (Use the original and combined ones that make sense)
cat_features = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 
                'Style', 'Color', 'Brand_Material', 'Brand_Style', 'Material_Style', 
                'Weight Capacity (kg) categorical']

###########################################
# 6. CatBoost Parameters (Using GPU)
###########################################
catboost_params = {
    "learning_rate": 0.062,
    "depth": 8,
    "iterations": 5000,
    "l2_leaf_reg": 7,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "early_stopping_rounds": 200,
    "verbose": 100,
    "task_type": "GPU"   # Enable GPU training
}

###########################################
# 7. Cross-Validated CatBoost Regression with GPU
###########################################
print("\n---------- Training CatBoost with 5-Fold CV ----------")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []
test_preds = []

# Prepare a CatBoost Pool for the test set.
test_pool = Pool(X_test, cat_features=cat_features)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nTraining fold {fold}/{n_splits}...")
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create CatBoost Pools.
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # Initialize and train the model.
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Validation predictions & RMSE.
    val_pred = model.predict(valid_pool)
    rmse_score = np.sqrt(mean_squared_error(y_val, val_pred))
    cv_scores.append(rmse_score)
    print(f"Fold {fold} RMSE: {rmse_score:.4f}")
    
    # Predict on the test set.
    fold_test_pred = model.predict(test_pool)
    test_preds.append(fold_test_pred)

print("\n---------- CV Results ----------")
print(f"Cross-validated RMSE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

###########################################
# 8. Final Predictions & Submission
###########################################
# Average predictions across folds.
final_test_preds = np.mean(test_preds, axis=0)
df_sub['Price'] = final_test_preds
df_sub.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created.")