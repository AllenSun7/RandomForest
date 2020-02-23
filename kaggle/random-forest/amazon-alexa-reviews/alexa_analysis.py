# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up matplotlib style 
plt.style.use('ggplot')

# Libraries for wordcloud making and image importing
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# And libraries for data transformation
import datetime
from string import punctuation

#words counter
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
from nltk.tokenize import word_tokenize 

#model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from itertools import compress
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import shap


def data_visualization(data):
    #distribution of reviews length
    reviews_length_distribution(data)
    #rating pie
    rating_pie(data)
    #distribution of counts variation
    counts_variation(data)
    #word cloud
    word_cloud(data)
    #reviews trend based on date
    reviews_trend_date(data)
    #box plot
    box_plot(data)

def reviews_length_distribution(data):
    #Take a look at the distribution of the length
    plt.figure()
    data['review_length'].hist(bins=20)
    plt.title('Distribution of review length')
    plt.show()

def rating_pie(data):
    #rating pie
    rating = data['rating'].value_counts()
    label_rating = rating.index
    size_rating = rating.values
    df = pd.DataFrame(size_rating, index=label_rating, columns=[''])
    #plot the pie
    df.plot(kind='pie', subplots=True, figsize=(8, 8))
    plt.title('Ratings Pie Chat')
    plt.show()

def counts_variation(data): 
    #counts of each variation
    plt.figure()
    sns.set(rc={'figure.figsize':(10,6)})
    sns.countplot(data.variation,
                order = data['variation'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Counts of each variation')
    plt.show()

def word_cloud(data):
    #word cloud of reviews
    A = np.array(Image.open('amazon_logo.png'))
    np.random.seed(321)
    sns.set(rc={'figure.figsize':(14,8)})
    reviews = ' '.join(data['verified_reviews'].tolist())
    wordcloud = WordCloud(mask=A,background_color="white").generate(reviews)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title('Reviews', size=20)
    plt.show()

    #group ratings based on 5, 1, 1-4
    data.rating.value_counts()
    data5 = data[data.rating == 5]
    data_not_5 = data[data.rating != 5]
    data1 = data[data.rating == 1]
    sns.set(rc={'figure.figsize':(14,8)})
    reviews = ' '.join(data5['verified_reviews'].tolist())
    
    #word cloud of reviews rating 5
    wordcloud = WordCloud(mask=A,background_color="white").generate(reviews)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title('Reviews of rating 5', size=20)
    plt.show()

    #several key words in rating 5
    pd.options.display.max_colwidth = 200
    data5[data5['verified_reviews'].str.contains('prime')]['verified_reviews'][:3]
    data5[data5['verified_reviews'].str.contains('time')]['verified_reviews'][:3]
    data5[data5['verified_reviews'].str.contains('easy')]['verified_reviews'][:3]

    #Reviews of rating 1
    sns.set(rc={'figure.figsize':(14,8)})
    reviews = ' '.join(data1['verified_reviews'].tolist())
    wordcloud = WordCloud(mask=A,background_color="white").generate(reviews)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title('Reviews of rating 1', size=20)
    plt.show()

    #several key words in rating 1
    data1[data1['verified_reviews'].str.contains('useless')]['verified_reviews'][:3]

    #Reviews if negative reviews(rating 1-4)
    reviews = ' '.join(data_not_5['verified_reviews'].tolist())
    wordcloud = WordCloud(mask=A,background_color="white").generate(reviews)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title('Reviews of negative reviews (rating 1-4)', size=20)
    plt.show()

def reviews_trend_date(data):
    #data trend on dates
    # It's weird that there is a peak in the end of July.
    print(data.head())
    data_date = data.groupby('date').count()
    data_date.rating.plot()
    plt.show()

def box_plot(data):
    #box figure
    plt.figure()
    sns.boxplot(data.variation, data.rating)
    plt.xticks(rotation = 90)
    plt.show()
    
    #review length
    sns.boxplot('rating','review_length',data=data)
    plt.show()
    #log review length of rating
    sns.boxplot('rating','log_review_length',data=data)
    plt.show()
    #log review length of variation
    sns.boxplot('variation','log_review_length',data=data)
    plt.xticks(rotation = 90)
    plt.show()

def data_preprocess(data):
    counts, vocab_to_int = count_common_words(data)
    print('Unique words: ', len((vocab_to_int)))
    print(counts.most_common(20))

    #visualize 20 most common words after natual language processing
    #common_words(data)
    
    #group ratings based on 5, 1, 1-4
    data.rating.value_counts()
    data5 = data[data.rating == 5]
    data_not_5 = data[data.rating != 5]
    data1 = data[data.rating == 1]

    print(data1['rating'].value_counts())
    counts1, vocab_to_int1 = count_common_words(data1)
    print(counts1.most_common(20))
    
    #natual language process
    #data_nlp(data)
    #set rating 5 as positive, the rest as negative
    data['positive'] = 0
    data.loc[data['rating'] ==5, 'positive'] = 1
    #print(word_tokenize(data.verified_reviews[0]))
    stop_words = set(stopwords.words('english')) 
    data['cleaned_reviews'] = data.verified_reviews.apply(lambda x: word_tokenize(x))
    data['cleaned_reviews'] = data.cleaned_reviews.apply(lambda x: [w for w in x if w not in stop_words])
    data['cleaned_reviews'] = data.cleaned_reviews.apply(lambda x: ' '.join(x))
    print(data.head())
    print(data.info())

    return data

def count_common_words(data):
    text = ' '.join(data['verified_reviews'].tolist())
    review_word = text.split(' ')
    all_reviews = ' '.join(review_word)
    words = all_reviews.split()
    # words wrong datatype
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    reviews_ints = []
    for review in review_word:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    return counts, vocab_to_int

def common_words(data):
    corpus=[]
    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")
    row, _ = data.shape
    for i in range(0, row):
        review = re.sub('[^a-zA-Z]', ' ', data['verified_reviews'][i] )
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        #natural language process
        doc = nlp(review)
        review = [chunk.text for chunk in doc.noun_chunks]
        review = ' '.join(review)
        corpus.append(review)
    common_words_visualization(corpus)

def common_words_visualization(corpus):
    words = []
    for i in range(0,len(corpus)):
        words = words + (re.findall(r'\w+', corpus[i]))
    # words contain all the words in the dataset
    words_counts = Counter(words)
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_wordList = []
    most_common_CountList = []
    for x, y in most_common_words:
        most_common_wordList.append(x)
        most_common_CountList.append(y)
    plt.figure(figsize=(20,18))
    plot = sns.barplot(np.arange(20), most_common_CountList[0:20])
    plt.ylabel('Word Count',fontsize=20)
    plt.xticks(np.arange(20), 
                most_common_wordList[0:20], 
                fontsize=20, 
                rotation=40)
    plt.title('20 Most Common Words used in Review.', fontsize=20)
    plt.show()

def data_model(data):
    y = data['positive']
    X_train, X_test, y_train, y_test = train_test_split(data["cleaned_reviews"], 
                                                        y, 
                                                        test_size=0.33
                                                        ,random_state=53)
    #Step 1
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words="english")
    # Transform the training data using only the 'text' column values: count_train 
    count_train = count_vectorizer.fit_transform(X_train)
    y_train = np.asarray(y_train.values)
    ch2 = SelectKBest(chi2, k = 300)
    X_new = ch2.fit_transform(count_train, y_train)
    # Transform the test data using only the 'text' column values: count_test 
    count_test = count_vectorizer.transform(X_test)
    X_test_new = ch2.transform(X=count_test)

    #Step 2
    # Initialize a TfidfVectorizer object: tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    # Transform the training data: tfidf_train 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    # Transform the test data: tfidf_test 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    #Step 3
    # Create the CountVectorizer DataFrame: count_df
    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
    # Create the TfidfVectorizer DataFrame: tfidf_df
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
    # Print the head of count_df
    print(count_df.head())
    # Print the head of tfidf_df
    print(tfidf_df.head())
    # Calculate the difference in columns: difference
    difference = set(count_df.columns) - set(tfidf_df.columns)
    print(difference)
    # Check whether the DataFrames are equal
    print(count_df.equals(tfidf_df))

    #Step 3
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_model(X_new, X_test_new, y_train, y_test)

    #Step 4
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_model(tfidf_train, tfidf_test, y_train, y_test)

    #Step 5
    #random forest classifier
    train, test = text_reprocessing(X_train, y_train, X_test)
    rf_model(train, test, y_train, y_test)

    #Step 6
    rf_model(X_new, X_test_new, y_train, y_test)

    #Step 7
    rf_model(tfidf_train, tfidf_test, y_train, y_test)

    #Model Tune
    print("Training model...")
    best_clf = model_tuning(X_new, X_test_new, y_train, y_test)
    print(best_clf)
    
    #features visualation
    features = features_visualization(count_vectorizer, best_clf, ch2)

    #weights and features
    X_test_new = X_test_new.toarray()
    perm = PermutationImportance(best_clf, random_state=1).fit(X_test_new, y_test)
    eli5.show_weights(perm, feature_names = features)

    #features contribution 
    best_clf.fit(count_train,y_train)
    eli5.show_prediction(best_clf, doc=X_train[20], vec=count_vectorizer)
    
    #decision tree
    tree_model = DecisionTreeClassifier(random_state=0, 
                                        max_depth=5, 
                                        min_samples_split=5)
    tree_model.fit(X_new, y_train)
    tree_graph = tree.export_graphviz(tree_model, feature_names=features)
    graphviz.Source(tree_graph)

    df = pd.DataFrame(X_test_new, columns = features)
    print(df.head())

    #plot feature love
    plot_pdp(tree_model, df, features, 'love')
    #plot feature time
    plot_pdp(tree_model, df, features, 'time')
    #plot feature use
    best_clf.fit(X_new, y_train)
    plot_pdp(best_clf, df, features, 'use')

    row_to_show = 5
    # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction = df.iloc[row_to_show]  
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    best_clf.predict_proba(data_for_prediction_array)
    #shap explainer
    try: 
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(best_clf, check_additivity=False)

        # Calculate Shap values
        shap_values = explainer.shap_values(data_for_prediction)
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], 
                        shap_values[1], 
                        data_for_prediction)
        plt.show()

    except:
        pass
    
    try:
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(best_clf, check_additivity=False)
        # calculate shap values. This is what we will plot.
        # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
        shap_values = explainer.shap_values(df)
        # Make plot. Index of [1] is explained in text below.
        shap.summary_plot(shap_values[1], df)
        plt.show()

        shap.dependence_plot('love', 
                            shap_values[1], 
                            df, 
                            interaction_index="great")
        plt.show()
    except:
        pass

def plot_pdp(model, dataset, features, feature_to_plot):
    # Create the data that we will plot
    pdp_goals = pdp.pdp_isolate(model=model, 
                                dataset=dataset, 
                                model_features=features, 
                                feature=feature_to_plot)
    # plot it
    pdp.pdp_plot(pdp_goals, feature_to_plot)
    plt.show()

def nb_model(X_train, X_test, y_train, y_test):
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()
    # Fit the classifier to the training data
    nb_classifier.fit(X_train, y_train)
    # Create the predicted tags: pred
    pred = nb_classifier.predict(X_test)
    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print('Accuracy is:', score)
    f1 = metrics.f1_score(y_test, pred)
    print('F score is:', f1)
    sns.heatmap(metrics.confusion_matrix(pred, y_test), annot=True, fmt='2.0f')
    plt.show()

def rf_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=51)
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    # Create the predicted tags: pred
    pred = clf.predict(X_test)
    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print('Accuracy is:', score)
    f1 = metrics.f1_score(y_test, pred)
    print('F score is:', f1)
    sns.heatmap(metrics.confusion_matrix(pred, y_test), annot=True, fmt='2.0f')
    plt.show()

def model_tuning(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    scorer = metrics.make_scorer(metrics.fbeta_score, beta=0.5)
    """
    parameters = {
                    'n_estimators': [150, 180, 250], #250
                    'max_features': [120, 150], #150
                    'max_depth': [120, 135, 150], #120
                    'min_samples_split':[3, 5], #3
                    'min_samples_leaf':[1, 3, 5] #1
                    }
    """
    parameters = {
                    'n_estimators': [250, 500], #250
                    'max_features': [150], #150
                    'max_depth': [120], #120
                    'min_samples_split':[3], #3
                    'min_samples_leaf':[1] #1
                    }

    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
    grid_fit = grid_obj.fit(X_train, y_train)
    # Get the estimator
    best_clf = grid_fit.best_estimator_
    best_predictions = best_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, best_predictions)
    print('Accuracy is:',score)
    f1 = metrics.f1_score(y_test, best_predictions)
    print('F score is:',f1)
    sns.heatmap(metrics.confusion_matrix(best_predictions, y_test),annot=True,fmt='2.0f')
    plt.show()
    return best_clf

def features_visualization(count_vectorizer, best_clf, ch2):
    #fetures importance
    features = count_vectorizer.get_feature_names()
    mask = ch2.get_support()
    features = list(compress(features, mask))
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)
    sns.set(rc={'figure.figsize':(11,50)})
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    return features

def text_reprocessing(X_train, y_train, X_test):
    #Test processing
    CountVectorizer(stop_words="english")
    count_vectorizer = CountVectorizer(stop_words="english")
    count = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)
    tfidf_vectorizer = TfidfTransformer()
    tfidf = tfidf_vectorizer.fit_transform(count)
    tfidf_test = tfidf_vectorizer.transform(count_test)
    ch2 = SelectKBest(chi2, k = 300)
    train_new = ch2.fit_transform(tfidf, y_train)
    test_new = ch2.transform(tfidf_test)
    
    return train_new, test_new

def main():
    # Import data and transform tsv file
    data = pd.read_csv('amazon_alexa.tsv', delimiter='\t')
    # Data overlook
    print(data.head())
    print(data.info())
    data['verified_reviews'] = data.verified_reviews.apply(lambda x: x.lower())
    data['verified_reviews'] = data.verified_reviews.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
    data['review_length'] = data.verified_reviews.apply(lambda x: len(x))
    data['log_review_length'] = data.review_length.apply(lambda x: (np.log(x)+1))
    # Check the data again
    print(data.head())
    # Take a look at the mean, standard deviation, and maximum
    print('The mean for the length of review:',data['review_length'].mean())
    print('The standard deviation for the length of reviews:',data['review_length'].std())
    print('The maximum for the length of reviews:',data['review_length'].max())

    # Transform date to datetime data type
    data['date'] = data.date.apply(lambda x: datetime.datetime.strptime(x, '%d-%b-%y'))
    print(data.info())
    
    #data visualization
    #data_visualization(data)

    #data preprocess
    data = data_preprocess(data)

    #data model
    data_model(data)






if __name__ == "__main__":
    main()
