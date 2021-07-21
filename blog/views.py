from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import random
import warnings; warnings.simplefilter('ignore')

data=pd.read_csv('result_final.csv')
data=data.rename(columns={'text':'sub-title'})
data=data.drop(['Unnamed: 0', 'Unnamed: 0.1','title_summary'],axis=1)
data.insert(0,'id',range(0,data.shape[0]))
data=data.set_index('id')
data=data.dropna()

#data wrangling
#we will remove all the stopwords those are unnecessary from the text. The words like is,a,the,etc
def remove_stopwords(news):
    news=news.split()
    news_without_stopwords=[]
    en_stops = set(stopwords.words('english'))
    for word in news:
        if word not in en_stops:
            news_without_stopwords.append(word)
    news=news_without_stopwords                      #news will be changed to string without unnceccessary words.
    return news

#Using the above created functions for getting clean data (without unwanted words and in a single format)
data['changed_summary'] = data['summary'].str.lower()
data['changed_summary'] =  data.changed_summary.apply(func = remove_stopwords)
data['changed_summary']=[" ".join(review) for review in data['changed_summary'].values]

display_data=data[['link','title','date']].copy()

for i in range(len(data)):
    data.iloc[i,4] = " ".join(eval(data.iloc[i,4]))

tfidf = TfidfVectorizer(analyzer='word',norm=None,use_idf=True,ngram_range=(1,3),smooth_idf=True)
matrix = tfidf.fit_transform(data['keywords'])

cosine_similarities = cosine_similarity(matrix, matrix)

def filter_function(x):
    if(x[1]>1000):
        return False
    return True


def recommendation(id=''):
    # get similarity values with other articles
    if (id == ''):
        return
    id=int(id)

    similarity_score = [(el[0], int(el[1]*10000)) for el in list(enumerate(cosine_similarities[id]))]
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = list(filter(filter_function, similarity_score))
    sim2 = []
    y = [j[1] for j in similarity_score]
    for i in range(len(similarity_score)):
        x = similarity_score[i][1]
        if (x not in y[i + 1:]):
            sim2.append(similarity_score[i])
    num = min(5, len(sim2))
    similarity_score = random.sample(sim2[:num+5], num)
    result=[]
    Header="News --> " + data['title'].iloc[id]
    Border=" "
    result.append(Header)
    result.append(Border)
    news = [i[0] for i in similarity_score]
    for i in range(len(news)):
        result.append("Recomendation " + str(i + 1) + " : " +
              data['title'].iloc[news[i]])
        result.append("Link -->" + data['link'].iloc[news[i]])
        result.append("score --> " + str(similarity_score[i][1]))
        result.append(" ")
    return result


# Create your views here.
def home(request):
    # parsing the DataFrame in json format.
    json_records = display_data.reset_index().to_json(orient='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data}
    return render(request, 'blog/home.html', context)
def sub_news(request):
    if request.method=="POST":
        Submitted=request.POST['Submitted']
        output=recommendation(Submitted)
        return render(request, 'blog/Selected-News.html',{'Submitted':Submitted,'display':output})
    else:
        return render(request,'blog/Selected-News.html')