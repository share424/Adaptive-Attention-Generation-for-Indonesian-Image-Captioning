# references: https://github.com/vrama91/cider
from typing import List, Dict, Tuple

import nltk
import numpy as np
from torchmetrics import Metric
import torch
from sklearn.feature_extraction.text import CountVectorizer


class CIDErScore(Metric):
    """Consensus-based Image Description Evaluation (CIDEr Score)"""

    def __init__(self, name: str, n: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.n = n
        self.count_preds = []
        self.count_refs = []

    def _calculate_ngram(self, sentences: List[str]) -> Dict[tuple, int]:
        removed_idx = []
        for i in range(len(sentences)):
            if len(sentences[i]) == 0:
                removed_idx.append(i)
        
        for i in reversed(removed_idx):
            sentences.pop(i)

        if len(sentences) == 0:
            return {}
        
        vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, ngram_range=(1, self.n))
        x = vectorizer.fit_transform(sentences)

        return dict(zip(vectorizer.get_feature_names_out(), x.toarray().sum(axis=0)))
    
    def _calculate_document_frequency(self):
        document_frequency = {}
        for refs in self.count_refs:
            for ref in refs:
                for ngram in ref:
                    if ngram not in document_frequency:
                        document_frequency[ngram] = 0
                    
                    document_frequency[ngram] += 1

        return document_frequency
    
    def _calculate_tfidf(
        self, 
        ngram_count: Dict[tuple, int], 
        document_frequency: Dict[tuple, int], 
        ref_length: int
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        vec = [{} for _ in range(self.n)]
        norm = [0.0 for _ in range(self.n)]
        for (ngram, term_freq) in ngram_count.items():           
            df = np.log(max(1.0, document_frequency.get(ngram, 0)))
            # we need to split by space since CountVectorizer uses space as separator
            n = len(ngram.split(" ")) - 1
            vec[n][ngram] = float(term_freq) * (ref_length - df)
            norm[n] += pow(vec[n][ngram], 2)

        norm = [np.sqrt(n) for n in norm]
        return vec, norm
    
    def _calculate_similarity(
        self, 
        tfidf_pred: List[Dict[str, float]], 
        tfidf_ref: List[Dict[str, float]], 
        norm_pred: List[float], 
        norm_ref: List[float]
    ) -> np.ndarray:
        val = np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            for ngram in tfidf_pred[n]:
                val[n] += tfidf_pred[n].get(ngram, 0) * tfidf_ref[n].get(ngram, 0)

            if (norm_pred[n] != 0) and (norm_ref[n] != 0):
                val[n] /= (norm_pred[n] * norm_ref[n]) + 1e-8 # add epsilon to avoid division by zero

        return val

    def update(self, preds: List[str], target: List[List[str]]) -> None:
        for pred, refs in zip(preds, target):
            self.count_preds.append(self._calculate_ngram([pred]))
            self.count_refs.append([self._calculate_ngram([ref]) for ref in refs])

    def compute(self) -> torch.Tensor:
        document_frequency = self._calculate_document_frequency()
        ref_length = np.log(float(len(self.count_refs)))

        scores = []
        for pred, refs in zip(self.count_preds, self.count_refs):
            vec, norm = self._calculate_tfidf(pred, document_frequency, ref_length)
            score = np.array([0.0 for _ in range(self.n)])
       
            for ref in refs:
                vec_ref, norm_ref = self._calculate_tfidf(ref, document_frequency, ref_length)
                score += self._calculate_similarity(vec, vec_ref, norm, norm_ref)

            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)

        return torch.tensor(np.mean(np.array(scores)))