
import os
import pandas as pd
import numpy as np
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# Simple string paths
DATA_TRAIN = "data/Train"
DATA_TEST = "data/Test"

# Filenames
FILES = {
    'train_customers': 'train_customers.csv',
    'train_locations': 'train_locations.csv',
    'train_orders': 'orders.csv',
    'vendors': 'vendors.csv',
    'test_customers': 'test_customers.csv',
    'test_locations': 'test_locations.csv'
}


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
def load_file(filename_key):
    """
    Finds and loads a CSV file from data folders.
    """
    fname = FILES[filename_key]
    
    # Try looking in both Train and Test folders
    search_paths = [DATA_TRAIN, DATA_TEST]
    
    for folder in search_paths:
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            print(f"Loading {fname}...")
            return pd.read_csv(full_path)
    
    # If not found by exact name, try searching for the file in directories
    for folder in search_paths:
        if os.path.exists(folder):
            for existing_file in os.listdir(folder):
                if fname in existing_file:
                    print(f"Loading {existing_file}...")
                    return pd.read_csv(os.path.join(folder, existing_file))
                    
    print(f"Error: Could not find {fname}")
    return None

def load_all_data():
    """
    Loads all datasets at once.
    """
    train_cust = load_file('train_customers')
    train_loc = load_file('train_locations')
    train_ord = load_file('train_orders')
    vendors = load_file('vendors')
    test_cust = load_file('test_customers')
    test_loc = load_file('test_locations')
    
    return train_cust, train_loc, train_ord, vendors, test_cust, test_loc


# --------------------------------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------------------------------

def prepare_orders(orders_df):
    """
    Cleans order data and creates a proper date column.
    """
    # handle different column names for dates
    if 'created_at' in orders_df.columns:
        orders_df['order_date'] = pd.to_datetime(orders_df['created_at'])
    elif 'delivery_date' in orders_df.columns:
        orders_df['order_date'] = pd.to_datetime(orders_df['delivery_date'])
    else:
        orders_df['order_date'] = pd.NaT
    return orders_df

def get_vendor_popularity(orders_df, vendors_df):
    """
    Calculates how many orders each vendor has received.
    """
    # Count orders per vendor
    pop = orders_df.groupby('vendor_id').size().reset_index(name='vendor_order_count')
    
    # Add this info to the vendors table
    vendors_df = vendors_df.merge(pop, left_on='id', right_on='vendor_id', how='left')
    vendors_df['vendor_order_count'] = vendors_df['vendor_order_count'].fillna(0)
    
    return vendors_df

def get_customer_stats(orders_df, customers_df):
    """
    Calculates summary statistics for each customer.
    """
    # Aggregate order info
    stats = orders_df.groupby('customer_id').agg({
        'order_id': 'count',
        'grand_total': 'mean',
        'order_date': 'max'
    }).reset_index()
    
    stats.columns = ['customer_id', 'total_orders', 'avg_spend', 'last_order_date']
    
    # Merge back to customers
    customers_df = customers_df.merge(stats, on='customer_id', how='left')
    return customers_df

def create_features(pairs_df, locations_df, vendors_df, customers_df, orders_df):
    """
    Creates the final set of features for the model.
    """
    print("Generating features...")
    
    # 1. Add Location Co-ordinates
    # Clean up column names first (remove spaces)
    locations_df = locations_df.rename(columns=lambda x: x.strip())
    
    # We primarily need customer location info
    cust_locs = locations_df[['customer_id', 'location_number', 'latitude', 'longitude']]
    
    pairs_df = pairs_df.merge(cust_locs, on=['customer_id', 'location_number'], how='left')
    
    # 2. Add Vendor Info
    pairs_df = pairs_df.merge(
        vendors_df[['id', 'latitude', 'longitude', 'vendor_order_count', 'vendor_rating']], 
        left_on='vendor_id', right_on='id', how='left'
    )
    
    # 3. Add Customer Stats
    pairs_df = pairs_df.merge(customers_df[['customer_id', 'total_orders', 'avg_spend', 'last_order_date']], on='customer_id', how='left')
    
    # 4. Calculate Distance (Simple Euclidean)
    # Distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    pairs_df['distance'] = np.sqrt(
        (pairs_df['latitude_x'] - pairs_df['latitude_y'])**2 + 
        (pairs_df['longitude_x'] - pairs_df['longitude_y'])**2
    )
    
    # 5. Has the customer ordered from this vendor before?
    # Create a lookup set of (customer, vendor) from history
    past_orders = set(zip(orders_df['customer_id'], orders_df['vendor_id']))
    
    def has_ordered(row):
        return 1 if (row['customer_id'], row['vendor_id']) in past_orders else 0
        
    pairs_df['ordered_before'] = pairs_df.apply(has_ordered, axis=1)
    
    # 6. Recency (Days since last order)
    pairs_df['days_since_last'] = (pd.to_datetime(pairs_df['order_date']) - pd.to_datetime(pairs_df['last_order_date'])).dt.days
    
    # Fill missing values
    pairs_df = pairs_df.fillna(0)
    
    # Select only numeric features for the model
    feature_cols = [
        'vendor_order_count', 
        'vendor_rating', 
        'ordered_before', 
        'total_orders', 
        'avg_spend', 
        'distance', 
        'days_since_last'
    ]
    
    return pairs_df, feature_cols
