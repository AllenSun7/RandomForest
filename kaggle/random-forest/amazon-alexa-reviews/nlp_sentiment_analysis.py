import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import spacy
from collections import Counter

def nlp_sentiment_analysis():
    df_review = pd.read_csv('amazon_alexa.tsv', sep='\t')
    print(df_review.head())
    print(df_review.describe())

    #Making a new column to detect how long the text messages are:
    df_review['length'] = df_review['verified_reviews'].apply(len)
    print(df_review.head())
    df_review['length'].plot(bins=50, kind='hist')
    plt.show()
    #Now let's focus back on the idea of trying to see 
    #if review length is a distinguishing feature between positive and negative review:
    df_review.hist(column='length', by='feedback', bins=50,figsize=(10,4))
    plt.show()

def randome_forest(X, y):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Feature Scaling
    sc =StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #modelling
    print("Training model...")
    clf = RandomForestClassifier()
    params = {
                    'bootstrap': [True],
                    'max_depth': [80, 100, 'auto'],
                    'min_samples_split': [8, 12],
                    'n_estimators': [100, 300]
    }
    grid = GridSearchCV(estimator=clf,
                        param_grid=params)
    grid.fit(X_train, y_train)
    #get best model
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    #results
    print("Training Accuracy :", best_model.score(X_train, y_train))
    print("Testing Accuracy :", best_model.score(X_test, y_test))
     #plot confusion tree
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    #f1 = f1_score(y_test, y_pred)
    #print("F1-Score: ", f1)

def dataset_visualization():
 # Importing the dataset
    print("preparing data...")
    # Load English tokenizer, tagger, parser, NER and word vectors
    #nlp = spacy.load("en_core_web_sm")
    dataset = pd.read_csv('amazon_alexa.tsv', delimiter = '\t', quoting = 3)
    print(dataset.head())
    print(dataset.describe())
    row, _ = dataset.shape
    corpus=[]
    # Load English tokenizer, tagger, parser, NER and word vectors
    #nlp = spacy.load("en_core_web_sm")
    for i in range(0, row):
        review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i] )
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        #natural language process
        #doc = nlp(review)
        #review = [chunk.text for chunk in doc.noun_chunks]
        #review = ' '.join(review)
        corpus.append(review)
    

    # creating the Bag of words Model
    vectorizer = CountVectorizer(max_features=1500)
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 0].values
    
    #run random forest
    #randome_forest(X, y)

    # most common words in negative reviews
    common_words(corpus)

def common_words(corpus):
    words = []
    for i in range(0,len(corpus)):
        words = words + (re.findall(r'\w+', corpus[i]))
    # words cantain all the words in the dataset
    words_counts = Counter(words)
    print(words_counts)
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_wordList = []
    most_common_CountList = []
    for x, y in most_common_words:
        most_common_wordList.append(x)
        most_common_CountList.append(y)
    plt.figure(figsize=(20,18))
    plot = sns.barplot(np.arange(20), most_common_CountList[0:20])
    plt.ylabel('Word Count',fontsize=20)
    plt.xticks(np.arange(20), most_common_wordList[0:20], fontsize=20, rotation=40)
    plt.title('Most Common Word used in Negative Review.', fontsize=20)
    plt.show()

def main():
    #nlp_sentiment_analysis()
    #randome_forest()
    dataset_visualization()

if __name__ == "__main__":
    main()