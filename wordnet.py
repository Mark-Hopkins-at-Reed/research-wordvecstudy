from nltk.corpus import wordnet as wn
import random
import pandas as pd
from data import TaggedPhraseDataset, TaggedPhraseSource

"""
If running for the first time, do the following in your Python console:

> import nltk
> nltk.download('wordnet')

"""

def synsets(word):
    return [s.name() for s in wn.synsets(word)]
    
def hypernym_chain(synset_name):
    """
    Provides a list of the hypernyms of the specified synset. This list
    begins with the specified synset, followed by its parent synset,
    grandparent synset, etc.
    
    > hypernym_chain('boat.n.01')
    ['boat.n.01', 
     'vessel.n.02', 
     'craft.n.02', 
     'vehicle.n.01', 
     'conveyance.n.03', 
     'instrumentality.n.03', 
     'artifact.n.01', 
     'whole.n.02', 
     'object.n.01', 
     'physical_entity.n.01', 
     'entity.n.01']
    
    """
    synset = wn.synset(synset_name)
    result = [synset.name()]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset.name())
    return result

def get_all_hypernym_synsets(synset_name):
    """
    Finds the set of all hypernym synsets of the specified synset.
    This set will not include the specified synset.
    
    > get_all_hypernym_synsets(synset_name):
    {'artifact.n.01',
     'conveyance.n.03',
     'craft.n.02',
     'entity.n.01',
     'instrumentality.n.03',
     'object.n.01',
     'physical_entity.n.01',
     'vehicle.n.01',
     'vessel.n.02',
     'whole.n.02'}
    
    """
    result = set()
    synset = wn.synset(synset_name)
    for y in synset.hypernyms():
        result.add(y.name())
        for z in get_all_hypernym_synsets(y.name()):
            result.add(z)
    return result



def get_all_hyponyms(synset_name):
    """
    Finds all words that are hyponyms of the given synset (i.e. they
    are lemmas of the given synset, or one of its descendant synsets).
    
    > get_all_hyponyms('noble_metal.n.01')
    {'24-karat_gold',
     'Ag',
     'Au',
     'Pt',
     'atomic_number_47',
     'atomic_number_78',
     'atomic_number_79',
     'coin_silver',
     'gold',
     'gold_dust',
     'green_gold',
     'guinea_gold',
     'platinum',
     'pure_gold',
     'silver'}
    
    """
    def get_all_hyponym_senses_from_sense(sense):
        closed = set()
        open_list = [sense]
        while len(open_list) > 0:
            synset = open_list.pop()
            if synset not in closed:
                closed.add(synset)
                for y in synset.hyponyms():
                    open_list.append(y)                
        #closed.remove(sense)
        return closed
    result = set()
    hyponyms = get_all_hyponym_senses_from_sense(wn.synset(synset_name))
    for hyponym in hyponyms:
        for lemma in hyponym.lemmas():
            result.add(lemma.name())
    return result




class Specificity:
    """
    Memoizing function that takes a synset and returns its "specificity,"
    defined as how many hyponyms it has.
    
    """
    def __init__(self):
        self.cache = dict()
        
    def __call__(self, synset_name):
        if synset_name not in self.cache:
            spec = len(get_all_hyponyms(synset_name))
            self.cache[synset_name] = spec
        return self.cache[synset_name]


specificity = Specificity()

def get_specific_synsets(min_specificity, max_specificity):
    """
    Returns the set of all synsets within the given (inclusive) 
    specificity range.
    
    """    
    result = set()
    synsets = wn.all_synsets()
    for synset in synsets:
        spec = specificity(synset.name())
        if spec >= min_specificity and spec <= max_specificity:
            result.add(synset.name())
    return result

    


def word_based(fn):
    """
    Given a function fn, that takes the name of a synset and returns a
    corresponding set, this creates a new function that takes a word and
    returns the union of fn(synset) for all synsets of that word.
    
    > get_hypernyms = word_based(get_all_hypernym_synsets)
    > get_hypernyms('wagon')
    {'artifact.n.01',
     'car.n.01',
     'container.n.01',
     'conveyance.n.03',
     'entity.n.01',
     'instrumentality.n.03',
     'motor_vehicle.n.01',
     'object.n.01',
     'physical_entity.n.01',
     'self-propelled_vehicle.n.01',
     'truck.n.01',
     'van.n.05',
     'vehicle.n.01',
     'wheeled_vehicle.n.01',
     'whole.n.02'}
    
    """
    def f(word):
        result = set()
        for x in wn.synsets(word):
            for y in fn(x.name()):
                result.add(y)
        return result
    return f


def find_lowest_common_ancestor(words):
    """
    Returns the most specific synset that is a hypernym of all of the
    given words.
    
    The return value is a tuple of the form: (specificity, synset_name)
    
    > find_lowest_common_ancestor(['communist', 'democrat', 'republican'])
    (60, 'politician.n.02')
    > find_lowest_common_ancestor(['apple', 'banana', 'orange', 'grape'])
    (297, 'edible_fruit.n.01')
    
    """
    get_hypernyms = word_based(get_all_hypernym_synsets)
    common_hypernyms = get_hypernyms(words[0])
    for word in words[1:]:
        common_hypernyms = common_hypernyms & get_hypernyms(word)
    if len(common_hypernyms) == 0:
        hyp = 'entity.n.01'
        return (specificity(hyp), hyp)
    scored_hypernyms = [(specificity(hyp), hyp) for hyp in common_hypernyms]
    sorted_hypernyms = sorted(scored_hypernyms)
    return sorted_hypernyms[0]



class WordnetDataSource(TaggedPhraseSource):
    
    def __init__(self):
        self.unknown = None
        self.min_length = None
        self.total = None

    def read(self, tags, phrase_recognizer):        
        recognized_tag_map = dict()
        unknown = 0
        total = 0
        min_length = float('inf')
        for tag in tags:
            recognized = []
            for phrase in get_all_hyponyms(tag):
                total += 1
                if phrase_recognizer(phrase):
                    recognized.append(phrase)
                else:
                    unknown += 1           
            random.shuffle(recognized)
            recognized_tag_map[tag] = recognized
            min_length = min(min_length, len(recognized))            
        result = []
        for tag in tags:
            for phrase in recognized_tag_map[tag]:
                result.append({'phrase': phrase, 'tag': tag})
        random.shuffle(result)
        self.unknown = unknown
        self.min_length = min_length
        self.total = total
        #return pd.DataFrame(result)
        return TaggedPhraseDataset(make_even(remove_duplicates(pd.DataFrame(result))))

                
def remove_duplicates(frame):
    rows_by_tag = {k: v for k, v in frame.groupby('tag')}
    phrase_sets = [set(rows_by_tag[tag]['phrase']) for tag in rows_by_tag]
    seen_before = set()
    duplicates = set()
    for s in phrase_sets:
        duplicates |= (s & seen_before)
        seen_before |= s
    result = []
    for tag in rows_by_tag:
        for phrase in rows_by_tag[tag]['phrase']:
            if phrase not in duplicates:
                result.append({'phrase': phrase, 'tag': tag})
    df = pd.DataFrame(result)
    df = df.reset_index(drop=True)
    return df
    
        
def make_even(frame):
    rows_by_tag = {k: v for k, v in frame.groupby('tag')}
    min_phrases = min([len(rows_by_tag[tag]['phrase']) for 
                       tag in rows_by_tag])
    result = []
    for tag in rows_by_tag:
        rows = rows_by_tag[tag]
        df = rows.sample(frac=min_phrases/len(rows))
        result.append(df)
    df = pd.concat(result)
    df = df.reset_index(drop=True)
    return df
    
    
    
    
    
    