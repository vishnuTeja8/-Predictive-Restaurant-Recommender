
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import os

from preprocess import load_all_data, prepare_orders, get_vendor_popularity, get_customer_stats, create_features

def main():
    print("------------------------------------------------")
    print(" STARTING MODEL TRAINING")
    print("------------------------------------------------")
    
    # 1. Load Data
    train_cust, train_loc, train_ord, vendors, _, _ = load_all_data()
    
    # 2. Preprocess
    train_ord = prepare_orders(train_ord)
    vendors = get_vendor_popularity(train_ord, vendors)
    train_cust = get_customer_stats(train_ord, train_cust)
    
    # 3. Create Training Data (Positive and Negative Samples)
    print("Creating training pairs...")
    
    # Positive samples: Customer actually ordered from Vendor
    # We need to map location number. Just assuming 'location_number' or similar exists.
    # In the raw data it might be called LOCATION_NUMBER.
    loc_col = 'LOCATION_NUMBER' if 'LOCATION_NUMBER' in train_ord.columns else 'location_number'
    
    positives = train_ord[['customer_id', loc_col, 'vendor_id', 'order_date']].copy()
    positives = positives.rename(columns={loc_col: 'location_number'})
    positives['target'] = 1
    
    # Negative samples: Customer did NOT order from Vendor
    # For every positive, we'll pick 4 random vendors they didn't order from.
    print("Sampling negatives...")
    all_vendors = vendors['id'].unique()
    negatives = []
    
    # Group by customer to make it faster
    cust_history = train_ord.groupby('customer_id')['vendor_id'].unique().to_dict()
    
    # We will sample negatives based on the positives we already have
    for idx, row in positives.iterrows():
        cid = row['customer_id']
        known_vendors = cust_history.get(cid, [])
        
        # Pick random vendors not in known list
        # Simple loop to find 4 negatives
        count = 0
        while count < 4:
            v = random.choice(all_vendors)
            if v not in known_vendors:
                negatives.append({
                    'customer_id': cid,
                    'location_number': row['location_number'],
                    'vendor_id': v,
                    'order_date': pd.NaT, # no order date for negative
                    'target': 0
                })
                count += 1
                
    negative_df = pd.DataFrame(negatives)
    
    # Combine
    train_df = pd.concat([positives, negative_df], ignore_index=True)
    print(f"Total training samples: {len(train_df)}")
    
    # 4. Feature Extraction
    train_df, features = create_features(train_df, train_loc, vendors, train_cust, train_ord)
    
    print(f"Features used: {features}")
    
    # 5. Train Model
    X = train_df[features]
    y = train_df['target']
    
    # Split 80/20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training LightGBM model...")
    # Simple parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100
    )
    
    # Check score
    val_preds = model.predict(X_val)
    score = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC Score: {score:.4f}")
    
    # 6. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save_model('models/lgb_model.txt')
    print("Model saved to models/lgb_model.txt")
    print("Done!")

if __name__ == "__main__":
    main()
