import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def random_forest_tree():
    #import data
    dataset = pd.read_csv('bill_authentication.csv')
    #prepare data for training
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Feature Scaling
    sc = StandardScaler()
    #fit the mean and standard deviation and then transform
    X_train = sc.fit_transform(X_train)
    #keep the consistency and the same scaling
    X_test = sc.transform(X_test)

    #train the alforithm
    classifier = RandomForestClassifier(n_estimators=200, 
                                bootstrap = True,
                                random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    rf_prob = classifier.predict_proba(X_test)[:, 1]
    y_accuracy = roc_auc_score(y_test, rf_prob)

    #evaluate the algorithm
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f'Model ROC Accuracy: {y_accuracy}')

 
    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=200, 
                                bootstrap = True,)
    # Fit on training data
    model.fit(X_train, y_train)
    # Actual class predictions
    rf_predictions = model.predict(X_test)
    # Probabilities for each class
    rf_probs = model.predict_proba(X_test)[:, 1]
    # Calculate roc auc
    roc_value = roc_auc_score(y_test, rf_probs)
    print(f'Model ROC Accuracy: {roc_value}')
    print(metrics.classification_report(rf_predictions, y_test))
    print(rf_predictions)


def main():
    random_forest_tree()

if __name__ == '__main__':
    main()