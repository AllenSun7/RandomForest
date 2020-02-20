from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def random_forest():
    
    digits = load_digits()
    print(digits.keys())
    print(np.size(digits))
    
    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                        digits.target,
                                                        random_state=0)
    
    #Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Train model
    model = RandomForestClassifier(n_estimators=1000,
                                    bootstrap=True,)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_pred, y_test))

    #plot confusion tree
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()



def main():
    random_forest()

if __name__ == '__main__':
    main()