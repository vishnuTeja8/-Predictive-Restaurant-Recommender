
import pandas as pd
import lightgbm as lgb
import os

from preprocess import load_all_data, prepare_orders, get_vendor_popularity, get_customer_stats, create_features

def main():
    print("------------------------------------------------")
    print(" STARTING PREDICTION")
    print("------------------------------------------------")
    
    # 1. Load Model
    if not os.path.exists('models/lgb_model.txt'):
        print("Error: Model not found. Run train.py first.")
        return
        
    print("Loading model...")
    model = lgb.Booster(model_file='models/lgb_model.txt')
    
    # 2. Load Data
    train_cust, train_loc, train_ord, vendors, test_cust, test_loc = load_all_data()
    
    # Precompute same stats as training
    train_ord = prepare_orders(train_ord)
    vendors = get_vendor_popularity(train_ord, vendors)
    train_cust = get_customer_stats(train_ord, train_cust)
    
    # 3. Create Test Pairs (Cartesian Product)
    # We need to predict for every Customer-Location in test against ALL Vendors
    print("Creating test pairs (this might take a moment)...")
    
    # Unique customer-locations in test
    test_keys = test_loc[['customer_id', 'location_number']].drop_duplicates()
    test_keys['key'] = 1
    
    # All vendors
    vendor_keys = vendors[['id']].rename(columns={'id': 'vendor_id'})
    vendor_keys['key'] = 1
    
    # Cross join
    test_df = test_keys.merge(vendor_keys, on='key').drop('key', axis=1)
    
    # Add dummy order date for feature ref (using max date or just None is handled in feature gen)
    test_df['order_date'] = pd.NaT
    
    print(f"Total pairs to predict: {len(test_df)}")
    
    # 4. Feature Extraction & Prediction
    test_df, features = create_features(test_df, test_loc, vendors, train_cust, train_ord)
    
    print("Predicting...")
    X_test = test_df[features]
    preds = model.predict(X_test)
    test_df['probability'] = preds
    
    # 5. Select Top 3 Vendors
    print("Selecting top 3 vendors per customer...")
    
    # Rank by probability
    test_df['rank'] = test_df.groupby(['customer_id', 'location_number'])['probability'].rank(method='first', ascending=False)
    
    # Keep top 3
    test_df['target'] = test_df['rank'].apply(lambda x: 1 if x <= 3 else 0)
    
    # 6. Create Submission File
    # Format: CID X LOC_NUM X VENDOR
    submission = pd.DataFrame()
    submission['CID X LOC_NUM X VENDOR'] = (
        test_df['customer_id'].astype(str) + " X " + 
        test_df['location_number'].astype(str) + " X " + 
        test_df['vendor_id'].astype(str)
    )
    submission['target'] = test_df['target']
    
    submission.to_csv('submission_final.csv', index=False)
    print("Saved to submission_final.csv")

if __name__ == "__main__":
    main()
