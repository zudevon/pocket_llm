import nltk
from nltk.tokenize import sent_tokenize
import re
import string


def extract_clean_sentences(text, remove_numbers=True):
    """
    Extract sentences from text and clean them of punctuation and optionally numbers.

    Args:
        text (str): Input text to process
        remove_numbers (bool): Whether to remove numerical digits

    Returns:
        list: List of cleaned sentences
    """
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # First split into sentences
    sentences = sent_tokenize(text)

    def clean_sentence(sentence):
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers if specified
        if remove_numbers:
            sentence = re.sub(r'\d+', '', sentence)

        # Remove extra whitespace
        sentence = ' '.join(sentence.split())

        return sentence

    # Clean each sentence
    cleaned_sentences = [clean_sentence(sent) for sent in sentences]

    # Remove any empty sentences
    cleaned_sentences = [sent for sent in cleaned_sentences if sent.strip()]

    return cleaned_sentences


def extract_sentences_spacy(text, remove_numbers=True):
    """
    Alternative method using spaCy for sentence segmentation.
    Requires: pip install spacy
    and: python -m spacy download en_core_web_sm
    """
    import spacy

    # Load English language model
    nlp = spacy.load('en_core_web_sm')

    # Process the text
    doc = nlp(text)

    # Extract and clean sentences
    sentences = []
    for sent in doc.sents:
        # Convert to string and clean
        cleaned = sent.text.translate(str.maketrans('', '', string.punctuation))
        if remove_numbers:
            cleaned = re.sub(r'\d+', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        if cleaned.strip():
            sentences.append(cleaned)

    return sentences


def clean_text_chunks(text):
    # text_chunks = text.split("\n\n")
    text_chunks = []
    # text_chunks.extend(re.split(r'(?<=[.!?]) +', text))
    text_chunks.extend(re.split(r'(?<=\W)(\w.*?)(?=\W)', text))
    # result = [x for x in text_chunks if x not in list(set(string.punctuation))]

    return list(set(text_chunks))