#import all teh required models
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# this function cleans the data by removing any special characters and digits
def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

# remove the stop words from the custom stop words list
def get_stop_words(stop_file_path):
    """load stop words """
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

# sorting the data
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# extracting the top N vectors/ words
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

# import the file or the sample data for training the model
df = pd.read_csv("naukri_com-job_sample.csv")
print("Schema:\n\n",df.dtypes)
print(len(df))
print("Number of questions,columns=",df.shape)

df['text'] = df['jobdescription'].astype(str) + df['jobtitle']
df['text'] = df['text'].apply(lambda x:pre_process(x))
df['text']

#load a set of stop words
stopwords=get_stop_words("stopwords.txt")
#get the text column
docs=df['text'].tolist()
#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)
# print(word_count_vector)

# fitting the training data
cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
word_count_vector=cv.fit_transform(docs)

# displaying the vocabulary of words
list(cv.vocabulary_.keys())

#initialising the model
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
model = tfidf_transformer.fit(word_count_vector)
print(model)

# TODO: change the input 
# passing the test data into the model
job_role = 'Backend Developer'
desp = 'Candidate should have at least 4 years of experience in Micro services technologies\
using Python and Django and Deep understanding of a micro services architecture including\
professional experience in the design build and operations of micro services in a\
production environment Experience designing REST APIs and implementing RESTful\
web services. Understanding of web services Experience in designing data\
persistence system using both SQL and NoSQL, DBMS, MongoDB elastic search\
Good understanding of SCRUM Agile methodology Experience in the\
management of a small team of IT professionals desirable\
Technologies Stack such as Python Django/Flask Framework,\
Unix, GitHub, Jenkins, Kafka Kibana, Postman, JSON, Spark,\
AWS Deployment.Create solutions by developing, implementing,and maintaining Python\
based components and interfaces.Define site objectives by analysing user requirements,\
envisioning system features and functionality.Design and develop user interfaces to\
internet/intranet applications by setting expectations and features priorities throughout\
development life cycle; determining design methodologies and tool sets; completing\
programming using languages and software products; designing and conducting tests.\
Lead the development effort of web services, Design and develop Rest based Web services\
Clear understanding of web services and SOA related standards like REST/OAuth/JSON.\
Development using Python, Microservices, RESTful APIs and unit tests.\
Responsible for Design, Development, Code reviews (peer review), Unit testing,\
providing support to testing team, Defect fixing, Defect triaging, Root causes Analysis\
and release / deployment support.\
Identify Risks and inform the PM and others on time\
Must have experience with Python and Django/Flask framework.\
Must have experience and knowledge in developing Microservices oriented architecture-based\
services using REST APIs.Should be comfortable in using Pyunit, Logger, Postman, Swagger.\
Should have hands on experience and through knowledge in understanding of data structures\
and algorithms. Working experience with PostgreSQL or Elastic DB.\
Proficient understanding of code versioning tools such as Git, Bit Bucket.'

# get test docs into a list
feature_names=cv.get_feature_names()
 
# get the document that we want to extract keywords from
doc= job_role + desp
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
