from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def random_forest():

    # Path of the file to read
    iowa_file_path = 'train.csv'

    home_data = pd.read_csv(iowa_file_path)
    row, col = home_data.shape
    # Create target object and call it y
    y = home_data.SalePrice
    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
                'TotalBsmtSF', 'GrLivArea', 'KitchenAbvGr', 'GarageArea', 'WoodDeckSF']
    X = home_data[features]

    # Split into validation and training data
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

    #preprocessing data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
    rf_model.fit(X_train, y_train)
    val_pred = rf_model.predict(X_val)

    # Calculate the mean absolute error of your Random Forest model on the validation data
    rf_val_mae = mean_absolute_error(val_pred, y_val)
    # Calculate roc auc

    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

def main():
    random_forest()

if __name__ == '__main__':
    main()