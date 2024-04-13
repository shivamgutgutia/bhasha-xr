from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity(string1,string2):
    strings = [string1.lower(), string2.lower()]
    print(strings)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(strings)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return cosine_similarities[0][0]