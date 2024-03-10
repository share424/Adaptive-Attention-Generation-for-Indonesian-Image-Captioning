from typing import Dict, Union, List, Optional
import json
from enum import Enum

import nltk
from pycocotools.coco import COCO
from tqdm import tqdm
import torch

from .config import TokenizerConfig


class SpecialToken(str, Enum):
    START = "<start>"
    END = "<end>"
    PAD = "<pad>"
    UNK = "<unk>"


def nltk_download_punkt():
    # check if punkt is installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def create_wordmap(annotations: List[str], min_word_freq: int = 5) -> Dict[str, int]:
    """Create word map from annotations
    
    Args:
        annotations (List[str]): list of annotations
    
    Returns:
        Dict[str, int]: word map
    """
    nltk_download_punkt()

    word_freqs = {}
    for annotation in annotations:
        coco = COCO(annotation)
        annotation_ids = coco.getAnnIds()
        for ann_id in tqdm(annotation_ids, desc=f"Processing {annotation}"):
            caption: str = coco.loadAnns(ann_id)[0]['caption']
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            for word in tokens:
                word_freqs[word] = word_freqs.get(word, 0) + 1

    word_set = set([word for word, freq in word_freqs.items() if freq >= min_word_freq])
    word_map = {word: idx + 4 for idx, word in enumerate(word_set)}
    word_map[SpecialToken.PAD] = 0
    word_map[SpecialToken.UNK] = 1
    word_map[SpecialToken.START] = 2
    word_map[SpecialToken.END] = 3

    return word_map


class Tokenizer:
    def __init__(self, wordmap: Dict[str, str], max_length: int):
        self.wordmap = wordmap
        self.reverse_wordmap = {idx: word for word, idx in wordmap.items()}
        self.max_length = max_length
        nltk_download_punkt()

    def tokenize(self, sentences: Union[str, List[str]], return_length=False, return_tensor=False) -> List[List[int]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        tokenized_sentences = []
        token_lengths = []
        for sentence in sentences:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            tokenized_sentence = [self.wordmap.get(token, self.wordmap[SpecialToken.UNK]) for token in tokens]
            
            # clip to max length - 2
            if len(tokenized_sentence) > self.max_length - 2:
                tokenized_sentence = tokenized_sentence[:self.max_length - 2]

            tokenized_sentence = [self.wordmap[SpecialToken.START]] + tokenized_sentence + [self.wordmap[SpecialToken.END]]
            token_lengths.append(len(tokenized_sentence))
            # fill with pad
            tokenized_sentence = tokenized_sentence + [self.wordmap[SpecialToken.PAD]] * (self.max_length - len(tokenized_sentence))
            
            tokenized_sentences.append(tokenized_sentence)

        if return_tensor:
            tokenized_sentences = torch.tensor(tokenized_sentences)
            token_lengths = torch.tensor(token_lengths)

        if return_length:
            return tokenized_sentence, token_lengths
        
        return tokenized_sentences
    
    def detokenize(self, tokenized_sentences: Union[List[List[int]], torch.Tensor], keep_special_token=False) -> List[str]:
        if isinstance(tokenized_sentences, torch.Tensor):
            tokenized_sentences = tokenized_sentences.tolist()
        sentences = []
        special_tokens = (self.wordmap[SpecialToken.START], self.wordmap[SpecialToken.PAD])
        
        if keep_special_token:
            special_tokens = ()
        
        for tokenized_sentence in tokenized_sentences:
            tokenized_sentence = [token for token in tokenized_sentence if token not in special_tokens]
            sentence = []
            for token in tokenized_sentence:
                if token == self.wordmap[SpecialToken.END]:
                    if keep_special_token:
                        sentence.append(self.reverse_wordmap[token])
                    break
                
                sentence.append(self.reverse_wordmap[token])

            sentences.append(" ".join(sentence))
        return sentences
    
    def get_index(self, word: str) -> int:
        return self.wordmap.get(word, self.wordmap[SpecialToken.UNK])
    
    def vocab_size(self) -> int:
        return len(self.wordmap)
    
    @staticmethod
    def from_config(config: TokenizerConfig) -> "Tokenizer":
        return Tokenizer(
            wordmap=json.load(open(config.wordmap, "r")),
            max_length=config.max_length
        )