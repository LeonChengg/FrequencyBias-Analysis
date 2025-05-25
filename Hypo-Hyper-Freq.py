import tempfile
import os

os.environ['TMPDIR'] = '~/data/dataset_freq_analyze'  # make sure this exists
tempfile.tempdir = '~/data/dataset_freq_analyze'

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import word_frequency
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import json

# Download needed resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # WordNet's multilingual data

lemmatizer = WordNetLemmatizer()

# Make sure WordNet is downloaded
nltk.download('wordnet')

def cal_Hypo_Hyper_Freq():
    verb_pairs = set()

    # Iterate through all verb synsets
    for synset in wn.all_synsets(pos=wn.VERB):
        hyponyms = synset.hyponyms()
        for hypo in hyponyms:
            # Add lemma names as pairs (can also use synsets directly)
            for hyper_lemma in synset.lemma_names():
                for hypo_lemma in hypo.lemma_names():
                    pair = (hypo_lemma, hyper_lemma)  # (hyponym, hypernym)
                    verb_pairs.add(pair)

    freq_hypo, freq_hyper = 0, 0
    for i, (hypo, hyper) in enumerate(sorted(verb_pairs)):
        print(f"{hypo} ? {hyper}")
        freq_hypo += word_frequency(hypo, "en")*1000*100
        freq_hyper += word_frequency(hyper, "en")*1000*100

    print(f"Average Frequency of Hyponymns:  {freq_hypo/len(verb_pairs)}")
    print(f"Average Frequency of Hypernymns:  {freq_hyper/len(verb_pairs)}")


# See If verb_a is the hypernyms of verb_b
def is_hypernym(verb_a, verb_b):
    # Get all synsets (senses) of A and B, restricted to verbs
    synsets_a = wn.synsets(verb_a, pos=wn.VERB)
    synsets_b = wn.synsets(verb_b, pos=wn.VERB)

    for syn_b in synsets_b:
        # Traverse all hypernyms (ancestors) of B using closure
        ancestors_b = set(syn_b.closure(lambda s: s.hypernyms()))
        for syn_a in synsets_a:
            if syn_a in ancestors_b:
                return True  # Found A in the hypernym tree of B
    return False

# Check if the word in wordnet verbs
def verb_in_wordnet(word):
    return any(syn.pos() == 'v' for syn in wn.synsets(word))

    

cal_Hypo_Hyper_Freq()