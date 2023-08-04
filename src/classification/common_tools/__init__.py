try:
    from tensorflow.contrib import learn
    import learn.preprocessing.VocabularyProcessor as VocabularyProcessor
except:
    from src.text_classification.common_tools.text import VocabularyProcessor

__all__ = [VocabularyProcessor]
