import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from utility.text_processing_helper import preprocess_text_embeddings
import ir_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


class EmbeddingSearcher:
    # 'all-mpnet-base-v2'
    # multi-qa-mpnet-base-dot-v1
    # all-MiniLM-L6-v2
    # dell-research-harvard/lt-wikidata-comp-en
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.documents = []
        self.device = self.check_gpu_compatibility()
        self.model = SentenceTransformer(model_name)

    def check_gpu_compatibility(self):
        if torch.backends.mps.is_available():
            print("MPS backend is available.")
            return torch.device("mps")
        else:
            print("MPS backend is not available. Using CPU.")
            return torch.device("cpu")

    def add_document(self, doc_id, text):
        self.documents.append((doc_id, text))

    def build_documents_embeddings(self):
        document_texts = [preprocess_text_embeddings(doc[1])
                          for doc in self.documents if doc[1] is not None]
        self.document_embeddings = self.model.encode(document_texts)

    def search(self, query, similarity_threshold=0.001):
        query_embedding = self.model.encode(preprocess_text_embeddings(query))

        similarities = cosine_similarity(query_embedding.reshape(
            1, -1), self.document_embeddings).flatten()

        doc = self.documents

        document_ranking = dict(zip(doc, similarities))

        filtered_documents = {key: value for key, value in document_ranking.items(
        ) if value >= similarity_threshold}

        sorted_dict = sorted(filtered_documents.items(),
                             key=lambda item: item[1], reverse=True)

        return sorted_dict

    def save(self, filename="antique_embed.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename="antique_embed.pkl"):
        with open(filename, 'rb') as f:
            return pickle.load(f)
