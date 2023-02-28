import re

import en_core_web_md  # python -m spacy download en_core_web_md
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


class Preprocessor():
    """
    The Preprocessor class is a utility class that helps to preprocess text data. It provides functionalities such as
    lowercasing the text, removing text inside parenthesis or square brackets, removing mentions, removing punctuation,
    removing stopwords, cleaning the text, POS tagging and lemmatization. It uses NLTK and spaCy library for the process
    and it's configurable based on the input data given to the class constructor.
    """

    def __init__(self, config_data):

        nltk.download('stopwords')
        self.nlp = en_core_web_md.load()
        self.remove_paran_content = config_data.get('remove_paran_content', None)
        self.noisywords = config_data.get('noisywords', None)

        self.tags_to_remove = config_data.get('remove_pos', None)

        self.wnl = WordNetLemmatizer()

    def preprocess_text(self, text):
        """
        This method takes a string text as input and applies various preprocessing steps to it based on the class
        variables set in the constructor. These steps include lowercasing the text, removing text inside parenthesis or
        square brackets, removing mentions, and cleaning the text.
        The method returns the preprocessed text.

        :param text: string that needs to be pre-processed
        :return: preprocessed string
        """
        text = text.lower()
        if self.remove_paran_content:
            text = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", text)

        if self.noisywords:
            text = text.split()
            text = [w for w in text if not w in set(self.noisywords) and len(w) >= 3]
            text = " ".join(text)


        return text

    def pos_preprocessing(self, docs):
        """
        This method takes a list of strings docs and a list of POS tags tags_to_remove as input and applies POS tagging
        and lemmatization to the text, removes stopwords, and any token that is not alpha or not in tags_to_remove list.
        The method returns a list of preprocessed strings.

        :param docs: list of strings docs
        :param tags_to_remove: list of POS tag to remove from doc
        :return: list of preprocessed strings
        """
        if self.tags_to_remove is None:
            return docs
        new_docs = []
        print('POS preprocessing')
        for doc in tqdm(self.nlp.pipe(docs, n_process=4)):
            tokens = [str(token) for token in doc if
                      token.pos_ not in self.tags_to_remove and not token.is_stop and token.is_alpha]

            # Lemma

            lemmatized_words = [self.wnl.lemmatize(word) for word in tokens]
            new_docs.append(" ".join(lemmatized_words))
        return new_docs
