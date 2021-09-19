import gensim
import nltk
import string
import numpy as np
import psutil
import logging

from nltk.tokenize import word_tokenize
from pathlib import Path

nltk.download('punkt')
logger = logging.getLogger(__name__)

# Download model: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
fasttext_path = Path(__file__).parents[3] / 'models/cc.en.300.bin'
tags_path = Path(__file__).parents[3] / 'data/tags.txt'

# # TODO: Remove, should use fasttext
# with open(Path(__file__).parents[3] / 'data/demo4.vocab', 'r') as f:
#     ID2WORD = [l.strip() for l in f.readlines()]
# WORD2ID = {tag:idx for (idx,tag) in enumerate(ID2WORD)}

def preprocess_text(text):
    if not text:
        return []
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens

class FeatureExtractor:
    model = None
    id2tag = None
    tag2id = None

    @classmethod
    def set_config(cls,config):
        mem = psutil.virtual_memory()
        enough_ram = mem.available >= 10 * 1024 * 1024 * 1024

        if config.use_fasttext and not enough_ram:
            logger.info('Not enough free RAM. Using simple extractor.')

        if config.use_fasttext and enough_ram:
            logger.info('Loading fasttext model...')
            cls.model =gensim.models.fasttext.load_facebook_vectors(fasttext_path)
            logger.info('Loading finished!')
        else:
            logger.info('Using no word2vec model! Only tags will be used as features.')
            with open(tags_path, 'r') as f:
                cls.id2tag = [l.strip() for l in f.readlines()]
            cls.tag2id = {tag:idx for (idx,tag) in enumerate(cls.id2tag)}

    @classmethod
    def extract(cls,element):
        if cls.model is not None:
            tag_vec = cls.model[element.tag]
        
            tokens = preprocess_text(element.text)
            vecs = [cls.model[t] for t in tokens]
            vecs.append(tag_vec)

            vec = np.mean(vecs,axis=0)

        else:
            if element.tag in cls.tag2id:
                idx = cls.tag2id[element.tag ]
            else:
                idx = len(cls.id2tag)
            vec = np.zeros(len(cls.id2tag)+1)
            vec[idx] = 1

            # # TODO: Remove, should use fasttext
            # onehot_token = np.zeros(len(ID2WORD)+1)
            # tokens = preprocess_text(element.text)
            # ids = [WORD2ID[t] if t in WORD2ID else len(WORD2ID) for t in tokens]
            # onehot_token[ids] = 1
            
            # vec = np.concatenate( (vec,onehot_token) ) 

        return vec