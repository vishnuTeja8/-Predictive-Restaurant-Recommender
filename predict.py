import pandas as pd
import lightgbm as lgb
import os

from preprocess import load_all_data, prepare_orders, get_vendor_popularity, get_customer_stats, create_features

def main():
    print("------------------------------------------------")
    print(" STARTING PREDICTION")
    print("------------------------------------------------")
    
    if not os.path.exists('models/lgb_model.txt'):
        print("Error: Model not found. Run train.py first.")
        return
        
    print("Loading model...")
    model = lgb.Booster(model_file='models/lgb_model.txt')
    
    train_cust, train_loc, train_ord, vendors, test_cust, test_loc = load_all_data()
    
    train_ord = prepare_orders(train_ord)
    vendors = get_vendor_popularity(train_ord, vendors)
    train_cust = get_customer_stats(train_ord, train_cust)
    
    # Create cartesian product: every customer-location paired with every vendor
    print("Creating test pairs (this might take a moment)...")
    
    test_keys = test_loc[['customer_id', 'location_number']].drop_duplicates()
    test_keys['key'] = 1
    
    vendor_keys = vendors[['id']].rename(columns={'id': 'vendor_id'})
    vendor_keys['key'] = 1
    
    test_df = test_keys.merge(vendor_keys, on='key').drop('key', axis=1)
    
    test_df['order_date'] = pd.NaT
    
    print(f"Total pairs to predict: {len(test_df)}")
    
    test_df, features = create_features(test_df, test_loc, vendors, train_cust, train_ord)
    
    print("Predicting...")
    X_test = test_df[features]
    preds = model.predict(X_test)
    test_df['probability'] = preds
    
    print("Selecting top 3 vendors per customer...")
    
    test_df['rank'] = test_df.groupby(['customer_id', 'location_number'])['probability'].rank(method='first', ascending=False)
    
    test_df['target'] = test_df['rank'].apply(lambda x: 1 if x <= 3 else 0)
    
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
