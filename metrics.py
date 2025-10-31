import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_doc_ids(docs):
    ids = set()
    for d in docs:
        if hasattr(d, "metadata"):
            page = d.metadata.get("page")
            if page is not None:
                ids.add(page)
        elif isinstance(d, int):
            ids.add(d)
        else:
            ids.add(str(d))
    return ids


def extract_facts(text):
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
    money = re.findall(r'\$\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?USD', text, flags=re.IGNORECASE)
    facts = set(numbers + percentages + money)
    return facts


def compute_precision_recall(retrieved_docs, relevant_docs):
    retrieved_set = extract_doc_ids(retrieved_docs)
    relevant_set = extract_doc_ids(relevant_docs)

    if not retrieved_set or not relevant_set:
        return 0.0, 0.0

    true_positives = retrieved_set.intersection(relevant_set)
    precision = len(true_positives) / len(retrieved_set)
    recall = len(true_positives) / len(relevant_set)

    return precision, recall


def compute_mrr(retrieved_docs, relevant_docs):
    return relevant_docs
