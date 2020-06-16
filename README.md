# Keyword Extraction using TF-IDF

# What is TF-IDF?

TF IDF is a natural language processing technique useful for the extraction of important keywords within a set of documents or chapters. The acronym stands for “term frequency-inverse document frequency”

# How does TF-IDF work?

TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:

IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

# Source
Link: https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.Xuivr3UzYxS
