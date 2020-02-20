from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import timeit

"""
For loop returns a smaller errors than GridSearchCV
"""

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
    features = X_train.shape[1]

    start = timeit.timeit()

    # Define the model. 
    parameters = {'n_estimators': [100, 200, 300, 400, 500, 600],
                'max_features': [features-5, features-4, features-3, features-2, features-1, features],
    }


    #train model
    print("Training model...")
    rf_model = RandomForestRegressor(random_state=1)
    grid_obj = GridSearchCV(rf_model, parameters)
    grid_fit = grid_obj.fit(X_train, y_train)
    
    # Get the estimator
    best_model = grid_fit.best_estimator_
    val_pred = best_model.predict(X_val)

    # Calculate the mean absolute error of your Random Forest model on the validation data
    rf_val_mae = mean_absolute_error(val_pred, y_val)
    # Calculate roc auc

    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
    print("best_model.get_params", best_model.get_params)
    end = timeit.timeit()
    print("grid time: %f"% (end-start))


    #for loop 
    start = timeit.timeit()
    n_estimators = [100, 200, 300, 400, 500, 600]
    max_features = [features-5, features-4, features-3, features-2, features-1, features]
    min_error = 1000000
     
    for n_estimator in n_estimators:
        for max_feature in max_features:
            model = RandomForestRegressor(n_estimators=n_estimator, 
                                            max_features=max_feature,
                                            random_state=1)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            rf_val_mae = mean_absolute_error(val_pred, y_val)
            #print(f"Validation MAE for Random Forest Model: {rf_val_mae}")
            #print("model.get_params", model.get_params)
            if rf_val_mae < min_error:
                min_error = rf_val_mae
                fit_model = model
    
    print("Validation MAE for Random Forest Model: {}".format(min_error))
    print("fit_model.get_params", fit_model.get_params)
    
    end = timeit.timeit()
    print("for time: %f" % (end-start))


def main():
    random_forest()

if __name__ == '__main__':
    main()