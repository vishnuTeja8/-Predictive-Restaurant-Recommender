# 🍽️ Predictive Restaurant Recommender

## The Business Problem

Imagine you're running a food delivery platform with thousands of restaurants and millions of hungry customers. Every day, customers open your app wondering, *"What should I eat today?"* 

**The Challenge:** With so many options, customers feel overwhelmed. They scroll endlessly, get frustrated, and sometimes just close the app. Meanwhile, restaurants struggle to reach the right customers who would love their food.

**The Opportunity:** What if we could predict which restaurants each customer is most likely to order from? By showing personalized recommendations right when they open the app, we can:
- **Reduce decision fatigue** for customers
- **Increase order conversion rates** 
- **Help restaurants reach their ideal customers**
- **Improve overall platform engagement**

## The Solution

This project builds a machine learning system that analyzes customer behavior patterns to predict restaurant preferences. By studying past orders, location proximity, spending habits, and vendor popularity, we create personalized "Top 3" restaurant recommendations for each customer.

### How It Works

1. **Understanding Customer Behavior**: We analyze historical order data to understand what drives customer choices
2. **Smart Feature Engineering**: We extract meaningful signals like:
   - Geographic proximity between customer and restaurant
   - Customer loyalty (have they ordered from this vendor before?)
   - Spending patterns and order frequency
   - Restaurant popularity and ratings
3. **Predictive Modeling**: Using LightGBM, we train a model to predict the likelihood of a customer ordering from each restaurant
4. **Personalized Recommendations**: For each customer, we rank all available restaurants and recommend the top 3 most likely matches

## Project Structure

```
📁 Predictive-Restaurant-Recommender/
├── 📄 preprocess.py          # Data loading and feature engineering
├── 📄 train.py               # Model training pipeline
├── 📄 predict.py             # Prediction and recommendation generation
├── 📄 requirements.txt       # Python dependencies
├── 📁 data/
│   ├── 📁 Train/
│   │   ├── train_customers.csv
│   │   ├── train_locations.csv
│   │   ├── orders.csv
│   │   └── vendors.csv
│   └── 📁 Test/
│       ├── test_customers.csv
│       └── test_locations.csv
└── 📁 models/
    └── lgb_model.txt         # Trained model (generated after training)
```

## How to Run This Project

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `lightgbm` - Gradient boosting model

### Step 2: Train the Model
```bash
python train.py
```

**What happens:**
- Loads training data (customers, locations, orders, vendors)
- Creates positive samples (actual orders) and negative samples (non-orders)
- Engineers features from the data
- Trains a LightGBM binary classifier
- Saves the trained model to `models/lgb_model.txt`
- Displays validation AUC score

**Expected output:**
```
------------------------------------------------
 STARTING MODEL TRAINING
------------------------------------------------
Loading train_customers.csv...
Loading train_locations.csv...
...
Validation AUC Score: 0.XXXX
Model saved to models/lgb_model.txt
Done!
```

### Step 3: Generate Predictions
```bash
python predict.py
```

**What happens:**
- Loads the trained model
- Creates test pairs (every customer × every vendor)
- Generates predictions for all pairs
- Ranks vendors by probability for each customer
- Selects top 3 recommendations per customer
- Saves results to `submission_final.csv`

**Expected output:**
```
------------------------------------------------
 STARTING PREDICTION
------------------------------------------------
Loading model...
Creating test pairs (this might take a moment)...
Total pairs to predict: XXXXX
Predicting...
Selecting top 3 vendors per customer...
Saved to submission_final.csv
```

## Key Features

| Feature | Description | Impact |
|---------|-------------|--------|
| **Distance** | Geographic distance between customer and vendor | Closer restaurants are more likely to be chosen |
| **Ordered Before** | Has customer ordered from this vendor previously? | Strong signal of preference and satisfaction |
| **Vendor Popularity** | Total number of orders the vendor has received | Popular vendors attract more customers |
| **Vendor Rating** | Customer satisfaction rating | Higher ratings increase order likelihood |
| **Customer Stats** | Total orders and average spending | Identifies customer engagement level |
| **Recency** | Days since customer's last order | Recent customers are more likely to order again |

## Model Performance

The model uses **LightGBM** (Light Gradient Boosting Machine), which is:
- ✅ Fast to train and predict
- ✅ Handles large datasets efficiently
- ✅ Works well with mixed feature types
- ✅ Resistant to overfitting

Performance is measured using **AUC-ROC** (Area Under the Curve), which evaluates how well the model distinguishes between restaurants a customer will order from vs. won't order from.

## Output Format

The final `submission_final.csv` contains predictions in the format:

```
CID X LOC_NUM X VENDOR,target
123 X 1 X 456,1
123 X 1 X 789,1
123 X 1 X 012,1
123 X 1 X 345,0
...
```

Where `target=1` indicates the top 3 recommended vendors for each customer-location pair.

---

**Built with using Python, LightGBM, and data-driven insights**
