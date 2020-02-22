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
# Pie chat
import plotly.graph_objs as go


def data_preprocess():
    data = 1
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
    data_visualization(data)

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

def main():
    data_preprocess()

if __name__ == "__main__":
    main()
