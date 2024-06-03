import re
import pycountry
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from textacy import preprocessing
from num2words import num2words
from utility.chat_words import chat_words
from utility.abbrevations import common_abbreviations
from utility.contractions import contractions
from utility.punct_mapping import punct,punct_mapping
from nltk.stem import WordNetLemmatizer
import spacy

stemmer = PorterStemmer()
spell = SpellChecker()
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to remove punctuation, emojis, and watermarks
def preprocess_text(text):
    text = text.lower()
    # Step 1
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.punctuation(text)
    text = preprocessing.remove.brackets(text)
    text = preprocessing.remove.accents(text)
    text = preprocessing.normalize.unicode(text)

    # Step 2
    text = handle_special_characters(text)
    text = remove_stopwords(text)
    text = remove_one_letter_words(text)

   
    # Step 3
    text = normalize_country_names(text)
    text = preprocessing.normalize.quotation_marks(text)
    text = preprocessing.normalize.hyphenated_words(text)

    text = preprocessing.replace.emojis(text)
    text = preprocessing.replace.urls(text)
    text = preprocessing.normalize.whitespace(text)
    text = replace_numbers_with_words(text)
    # text = replace_named_entities(text)

    # Step 4
    # text = chat_conversion(text)
    # text = normalize_abbrevations(text)
    # text = clean_special_chars(text,punct,punct_mapping)

    text = stem_words(text)

    return text.lower()

def preprocess_text_embeddings(text):
    text = text.lower()

    # text = clean_special_chars(text,punct,punct_mapping)
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.punctuation(text)
    text = preprocessing.remove.brackets(text)

    text = preprocessing.replace.emojis(text)
    text = preprocessing.replace.urls(text)
    text = preprocessing.remove.accents(text)
    text = normalize_country_names(text)
    text = text.lower()
    return text

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def lem_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def replace_numbers_with_words(text):
    # Regular expression to find numbers
    pattern = re.compile(r'\b\d+\b')
    
    # Function to convert match to words
    def num_to_word(match):
        num = int(match.group())
        return num2words(num)

    # Replace all numbers with their word equivalents
    processed_text = pattern.sub(num_to_word, text)
    return processed_text

def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

def normalize_country_names(text):
    upper_token = text.upper()
    country_name = None

    # Try to lookup by alpha-2 code
    country = pycountry.countries.get(alpha_2=upper_token)
    if not country:
        # Try to lookup by alpha-3 code if alpha-2 lookup fails
        country = pycountry.countries.get(alpha_3=upper_token)

    if country:
        country_name = country.name  

    return country_name if country_name else text


def normalize_abbrevations(text):
    new_text = []
    for i in text.split():
        if i.upper() in common_abbreviations:
            new_text.append(common_abbreviations[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)
    
def normalize_contractions(text):
    new_text = []
    for i in text.split():
        if i in contractions:
            new_text.append(contractions[i])
        else:
            new_text.append(i)
    return " ".join(new_text)

def replace_named_entities(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, ent.label_)
    return text

def handle_special_characters(text):
    # Remove or replace special characters
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text

def remove_one_letter_words(text):
    # Convert all caps to lowercase except for acronyms
    return " ".join([word for word in str(text).split() if len(word) > 1])
