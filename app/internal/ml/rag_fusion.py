import os
from sentence_transformers import SentenceTransformer, util
import app.internal.ml.model as model
import io


class Rag_fusion:
    def __init__(self, query="", docs=[], doc_emb=[]):
        self.query = query
        self.docs = docs
        self.doc_emb = doc_emb

    def score_docs(self, query, model_search):
        query_emb = model_search.encode(query)
        # doc_emb = model_search.encode(self.docs_embeding)

        scores = util.dot_score(query_emb, self.doc_emb)[0].cpu().tolist()

        doc_score_pairs = list(zip([self.docs], scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return doc_score_pairs

    async def generate(self, model, model_search, top_k=4):
        querys = model(f"Сгенерируй 3 уточняющих вопроса для запроса: {self.query}. Раздели их между собой '\n'").split("\n")

        scored_queries = [self.score_docs(query, model_search) for query in querys]

        # Select top_k queries based on the highest scoring document for each query
        best_queries = sorted(scored_queries, key=lambda x: x[0][1] if x else -float('inf'), reverse=True)[:top_k]  # handle empty list

        top_docs = []
        for query_scores in best_queries:
            top_docs.extend([doc for doc, score in query_scores])

        context = " ".join(top_docs)
        final_answer = model(f"Используя предоставленный контекст: {context}, ответь на запрос: {self.query}")

        return final_answer

    async def generate_with_arr(self, model, arr_data, top_k=4):
        best_queries = sorted(arr_data, key=lambda x: x[0][1] if x else -float('inf'), reverse=True)[:top_k]  # handle empty list

        top_docs = []
        for query_scores in best_queries:
            top_docs.extend([doc for doc, score in query_scores])

        context = " ".join(top_docs)
        final_answer = model(f"Используя предоставленный контекст: {context}, ответь на запрос: {self.query}")

        return final_answer


async def example(model_search, query, api):
    docs = ["Это счастливая собака прыгнула на полку",
            "Это счастливый человек читал газету",
            "Эта погода прекрастна",
            "Это замечательный день для прогулки с собаками",
            "Собака играет на площадке",
            "Человек с улыбкой на лице",
            "Погода прекрасна для отдыха на природе",
            "Счастливые мгновения с друзьями",
            "Квадробер в парке, радующий прохожих",
            "Собака лает от счастья",
            "Человек наслаждается своей жизнью",
            "Солнечный день и веселая компания",
            "Спокойствие и радость в душе"]
    docs_emb = model_search.encode(docs)

    m = Rag_fusion(query, docs, docs_emb)
    ans = m.generate(model=model.Model_API(api_key=api).generate,
                     model_search=model_search)
    return ans


if __name__ == '__main__':
    ...