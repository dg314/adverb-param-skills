from typing import List, Tuple, Callable

import gensim.downloader
from gensim.models import KeyedVectors
import numpy as np

def download_keyed_vectors() -> KeyedVectors:
    keyed_vectors = gensim.downloader.load("word2vec-google-news-300")
    print(f"\033[92mSuccessfully downloaded word2vec-google-news-300 keyed vectors.\033[0m")

    return keyed_vectors

def create_continuum_predictor(keyed_vectors: KeyedVectors, words_and_values: List[Tuple[str, float]]) -> Callable[[str], float]:
    words, values = list(zip(*words_and_values))
    vectors = np.array([keyed_vectors.get_vector(word) for word in words])

    def continuum_predictor(word: str) -> float:
        vector = keyed_vectors.get_vector(word)
        
        similarities = keyed_vectors.cosine_similarities(vector, vectors) ** 5
        
        return np.sum(np.array(values) * similarities / np.sum(similarities))
    
    return continuum_predictor

def create_word_to_magnitude_predictor(keyed_vectors: KeyedVectors) -> Callable[[str], float]:
    words_and_values = [
        ("unnoticeably", 0.1),
        ("insignificantly", 0.2),
        ("marginally", 0.2),
        ("slightly", 0.2),
        ("somewhat", 0.4),
        ("considerably", 0.7),
        ("significantly", 0.8),
        ("greatly", 0.8),
        ("extremely", 0.9),
        ("incredibly", 0.9),
        ("unbelievably", 1)
    ]
    return create_continuum_predictor(keyed_vectors, words_and_values)

def create_word_to_slower_faster_predictor(keyed_vectors: KeyedVectors) -> Callable[[str], float]:
    words_and_values = [
        ("slower", -1),
        ("gentler", -0.7),
        ("swifter", 1),
        ("speedier", 1),
        ("quicker", 1),
        ("faster", 1)
    ]
    return create_continuum_predictor(keyed_vectors, words_and_values)

def create_word_to_lower_higher_predictor(keyed_vectors: KeyedVectors) -> Callable[[str], float]:
    words_and_values = [
        ("lower", -1),
        ("beneath", -1),
        ("below", -1),
        ("underneath", -1),
        ("higher", 1),
        ("above", 1)
    ]
    return create_continuum_predictor(keyed_vectors, words_and_values)

def create_word_to_closer_farther_predictor(keyed_vectors: KeyedVectors) -> Callable[[str], float]:
    words_and_values = [
        ("closer", -1),
        ("nearer", -1),
        ("further", 1),
        ("farther", 1)
    ]
    return create_continuum_predictor(keyed_vectors, words_and_values)
