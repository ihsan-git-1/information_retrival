from sklearn.metrics.pairwise import cosine_similarity
from utility.text_processing_helper import preprocess_text

class QueryHandler:
    def __init__(self, vsm):
        self.vsm = vsm



    def search(self, query, similarity_threshold = 0.001):
        # Preprocess the query
        preprocessed_query = preprocess_text(query)

        # Transform the query into TF-IDF vector
        query_vector = self.vsm.tfidf_vectorizer.transform([preprocessed_query])
        
        similarities = cosine_similarity(query_vector, self.vsm.tfidf_vectors).flatten()

        doc =  self.vsm.documents

        document_ranking = dict(zip(doc, similarities))
        
        filtered_documents = {key: value for key, value in document_ranking.items() if value >= similarity_threshold}

        sorted_dict = sorted(filtered_documents.items(), key=lambda item: item[1], reverse=True)

        return sorted_dict
