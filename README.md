# Restaurant Recommender System
**Assignment**

This project predicts which restaurants a customer is likely to order from. It uses machine learning (LightGBM) to analyze past orders, location, and vendor details.

## Project Files
- **preprocess.py**: Contains all the code to load data and create features.
- **train.py**: Trains the model and saves it.
- **predict.py**: Uses the trained model to create the final predictions file.
- **data/**: Contains the dataset csv files.
- **models/**: Where the trained model is saved.

## How to Run

### 1. Install Libraries
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Train Model
Run the training script to build the model:
```bash
python train.py
```
This will create a file `models/lgb_model.txt`.

### 3. Generate Predictions
Run the prediction script to create the submission file:
```bash
python predict.py
```
This will create `submission_final.csv`.

## Methodology
1. **Data Loading**: We load info about customers, their location, and past orders.
2. **Feature Engineering**: We create simple features like:
   - How popular a vendor is (total orders).
   - Distance between customer and restaurant.
   - If the customer ordered from there before.
   - Average amount the customer spends.
3. **Modeling**: We use LightGBM for binary classification (Will order / Won't order).
4. **Prediction**: For each test customer, we rank all vendors and pick the top 3.
