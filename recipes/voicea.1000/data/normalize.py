#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (c) 2018-2019 Voicea
# sylvainlg@voicea.ai
# mohamedn@voicea.ai

import warnings
warnings.filterwarnings("ignore")
import argparse
import traceback
import sys
import inflect
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
import unicodedata
#nltk.download('punkt')
import re
#import ipdb
import string
import pygtrie

INFLECT = inflect.engine()
APOSTROPHE_TOKEN = 'Ap0str0phE'

GESTURES = set(['mm', 'hmm', 'um', 'umm', 'uh', 'oh', 'ah', 'mmm', 'hm', 'eh', 'ahh', 'ehh', 'uhh'])

CONTRACTIONS = {
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
    "might've": "might have",
    "there'd": "there would",
    "it'd": "it would",
    "we're": "we are",
    "they're": "they are",
    "what's": "what is",
    "here's": "here is",
    "it's": "it is",
    "i'm": "i am",
    "i'll": "i will",
    "we'll": "we will",
    "let's": "let us"
}

COMMON = {
    "okay": "ok",
    "kay": "ok",
    "na": "nah",
    "naa": "nah",
    "naaa": "nah",
    "naah": "nah",
    "kinda": "kind of",
    "wanna": "want to",
    "yup": "yes",
    "yeah": "yes",
    "yep": "yes",
}

SPLIT_WORDS = {
    "feed back": "feedback",
    "feed forward": "feed-forward",
    "other wise": "otherwise",
    "may be": "maybe",
    "look alike": "lookalike",
    "look a like": "lookalike",
    "miss match": "mismatch",
    "linked in": "linkedin",
    "drop down": "dropdown",
    "all right": "alright",
    "after wards": "afterwards"
}
SPLIT_WORDS = pygtrie.StringTrie(SPLIT_WORDS, separator=" ")


NUM_WORD_REGEX = re.compile(r"([0-9]+)([a-z]+)")
WORD_NUM_REGEX = re.compile(r"([a-z]+)([0-9]+)")
NUMBERS_REGEX = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?', re.MULTILINE)
PUNCT_REGEX = re.compile(r'[' + string.punctuation + ']+', re.MULTILINE)
CONTRACTIONS_REGEX = re.compile(r"\b'(all|am|clock|d|er|ll|m|re|s|t|tis|twas|ve|til)", re.MULTILINE)
NO_PARENS = re.compile(r'[\(\[\<].*?[\]\>\)]')
DOLLARS_REGEX = re.compile(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', re.MULTILINE)

def normalize_rank_suffix(input_str, suffix_from, suffix_to):
    output_str = input_str
    # find location of each pattern if any
    while True:
        pos = output_str.find(suffix_from)
        if (pos == -1):
            break
        
        # if the prior char was a digit
        # insert a 0. Example 31st -> 30 first
        if (pos > 0 and output_str[pos - 1].isdigit()):
            output_str = output_str.replace(suffix_from, "0 " + suffix_to, 1)
        else:
            output_str = output_str.replace(suffix_from, suffix_to, 1)            
    return output_str

def add_space_between_words_and_digits(text):
    text = NUM_WORD_REGEX.sub(r'\1 \2', text)
    return(WORD_NUM_REGEX.sub(r'\1 \2', text))

def replace_numbers_by_words(text):
    def number_to_words_helper(match):
        return(INFLECT.number_to_words(match.group()))
    return(NUMBERS_REGEX.sub(number_to_words_helper, text))
    
def remove_extra_spaces(text):
    return(re.sub('\s+', ' ', text).strip())

def normalize_unicode(text):
    # Beyoncé becomes beyonce. sorry B.
    return(unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii'))

def remove_fillers(text):
    pass
    
def remove_punctuation(text):
    return(PUNCT_REGEX.sub(' ', text))
        
def save_apostrophe_in_contractions(text):
    return(CONTRACTIONS_REGEX.sub(APOSTROPHE_TOKEN + r'\1', text))
    
def remove_extra_punc(text):
    return(text.replace("\\", ""))

def remove_annotators_comments(text):
    return(re.sub(NO_PARENS, "", text))

def replace_dollar_sign_by_word(text):
    return(DOLLARS_REGEX.sub(r'\1 dollars', text))

def to_lower_case(text):
    return(text.lower())

def to_upper_case(text):
    return(text.upper())

def tokenize_into_phrases(text):
    return(sent_tokenize(text))

def is_gesture(word):
    return (word in GESTURES)

def normalize_contractions(word):
    return CONTRACTIONS.get(word, word)

def normalize_common_words(word):
    return COMMON.get(word, word)

def combine_split_words(txt):
    words = txt.split(" ")
    i, words2 = 0, []
    while i < len(words):
        if not SPLIT_WORDS.has_subtrie(words[i]):
            words2.append(words[i])
            i += 1
            continue
        j = 0
        while (i+j < len(words)) and SPLIT_WORDS.has_subtrie(" ".join(words[i:i+j+1])):
            j += 1
        if SPLIT_WORDS.has_key(" ".join(words[i:i+j+1])):
            new_word = SPLIT_WORDS[" ".join(words[i:i+j+1])]
        elif SPLIT_WORDS.has_key(" ".join(words[i:i+j])):
            new_word = SPLIT_WORDS[" ".join(words[i:i+j])]
        else:
            words2.append(words[i])
            i += 1
            continue
        words2.append(new_word)
        i += j+1
    return " ".join(words2)

def normalize_word_by_word(txt):
    words = txt.split(" ")
    words2 = []
    for word in words:
        if is_gesture(word):
            continue
        word = normalize_contractions(word)
        word = normalize_common_words(word)
        words2.append(word)
    return " ".join(words2)


def normalize_text(text):
    #sys.stderr.write("[INFO] Text normalization for training\n")
    text = to_lower_case(text)
    text = remove_annotators_comments(text)
    text = replace_dollar_sign_by_word(text)
    text = normalize_rank_suffix(text, "1st", "first")
    text = normalize_rank_suffix(text, "2nd", "second")
    text = normalize_rank_suffix(text, "3rd", "third")
    text = add_space_between_words_and_digits(text)
    text = replace_numbers_by_words(text)
    text = save_apostrophe_in_contractions(text)
    text = remove_punctuation(text)
    text = remove_extra_punc(text)
    text = normalize_unicode(text)
    text = remove_extra_spaces(text)
    text = text.replace(APOSTROPHE_TOKEN, "'")
    text = normalize_word_by_word(text)
    return(text)

def normalize_text_eval(text):
    #sys.stderr.write("[INFO] Text normalization for wer\n")
    text = to_lower_case(text)
    text = remove_annotators_comments(text)
    text = replace_dollar_sign_by_word(text)
    text = normalize_rank_suffix(text, "1st", "first")
    text = normalize_rank_suffix(text, "2nd", "second")
    text = normalize_rank_suffix(text, "3rd", "third")
    text = add_space_between_words_and_digits(text)
    text = replace_numbers_by_words(text)
    text = save_apostrophe_in_contractions(text)
    text = remove_punctuation(text)
    text = remove_extra_punc(text)
    text = normalize_unicode(text)
    text = remove_extra_spaces(text)
    text = text.replace(APOSTROPHE_TOKEN, "'")
    text = normalize_word_by_word(text)
    return text
    
