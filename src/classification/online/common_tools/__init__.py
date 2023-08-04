try:
    from tensorflow.contrib import learn
    import learn.preprocessing.VocabularyProcessor as VocabularyProcessor
except:
    from common_tools.text import VocabularyProcessor

__all__ = [VocabularyProcessor]
