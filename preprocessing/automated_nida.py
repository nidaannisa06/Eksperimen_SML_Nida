import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_housing_data(input_path, output_folder):
    """Preprocess Bangalore housing data and save train/test splits"""
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Successfully loaded data from {input_path}")

        # Convert and clean features
        def convert_sqft(x):
            try:
                if isinstance(x, str) and '-' in x:
                    low, high = map(float, x.split('-'))
                    return (low + high) / 2
                return float(x)
            except:
                return np.nan

        df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
        df['bedrooms'] = df['size'].str.extract(r'(\d+)').astype(float)

        # Handle missing values
        num_cols = ['total_sqft', 'bath', 'balcony', 'bedrooms']
        cat_cols = ['area_type', 'availability', 'location', 'size', 'society']

        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Feature engineering
        df['price_per_sqft'] = df['price'] / df['total_sqft']

        # Remove outliers
        df = df[(df['price_per_sqft'] >= df['price_per_sqft'].quantile(0.01)) & 
                (df['price_per_sqft'] <= df['price_per_sqft'].quantile(0.99))]

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=['area_type', 'availability', 'location', 'size', 'society'])

        # Split into features and target
        X = df.drop('price', axis=1)
        y = df['price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Save processed data
        X_train.to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)

        print(f"Successfully saved processed data to {output_folder}")
        return True

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess Bangalore housing data')
    parser.add_argument('--input', type=str, default='../housing_raw/Bengaluru_House_Data.csv',
                      help='Path to raw data CSV file')
    parser.add_argument('--output', type=str, default='../preprocessing/housing_preprocessed',
                      help='Folder to save processed data')
    
    args = parser.parse_args()
    preprocess_housing_data(args.input, args.output)