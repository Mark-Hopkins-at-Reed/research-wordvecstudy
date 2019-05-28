import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import json
from collections import defaultdict

 
class TaggedPhraseDataset(Dataset):
    """
    A Torch Dataset consisting of tagged phrases. 
    Each datum is a dictionary with the following keys:
        
        - 'phrase': a string representation of a phrase
        - 'tag': a (string) tag for that phrase
    
    """
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        tag = self.data_frame['tag'][idx]
        phrase = self.data_frame['phrase'][idx]
        sample = {'phrase': phrase, 'tag': tag}
        return sample
    
    def restrict_to_tag(self, tag):
        df = self.data_frame
        return df.loc[df['tag'] == tag]


class TaggedPhraseSource:
    """
    An abstract class that allows you to build TaggedPhraseDatasets from
    various sources, e.g. a JSON file or Wordnet.
    
    """
    
    def read(self, tags, phrase_recognizer):
        """
        Finds all (phrase, tag) pairs from the given source such that
        tag is in the set `tags` and phrase_recognizer(phrase) returns
        True.
        
        - tags: a set of tags
        - phrase_recognizer: a predicate that maps a string to a boolean        
        
        """        
        raise NotImplementedError()
    

class JsonPhraseSource(TaggedPhraseSource):
    """
    Reads tagged phrases from a JSON file that stores a list of 
    data of the form: {'tag': tag, 'phrase': phrase}.
    
    """    
    def __init__(self, tag_json):
        self.tag_json = tag_json

    def read(self, tags, phrase_recognizer):
        with open(self.tag_json) as f:
            tag_map = json.load(f)
        recognized_tag_map = dict()
        unknown = 0
        total = 0
        min_length = float('inf')
        for tag in tags:
            recognized = []
            for phrase in tag_map[tag]:
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
            for phrase in recognized_tag_map[tag][:min_length]:
                result.append({'phrase': phrase, 'tag': tag})
        random.shuffle(result)
        return TaggedPhraseDataset(pd.DataFrame(result))



class DataManager:
    """
    Wrapper for a TaggedPhraseSource that indexes the tags, and splits
    the data into train, dev, and test partitions.
    
    The DataManager also provides a mechanism for converting the phrases
    into vector embeddings.
    
    """    
    def __init__(self, 
                 data_source, 
                 tags, 
                 vectorizer, 
                 phrase_recognizer=lambda p: True):
        
        self.dataset = data_source.read(tags, phrase_recognizer)           
        self.num_phrases = {tag: len(self.dataset.restrict_to_tag(tag)) for 
                            tag in tags}
        self.tags = tags
        self.vectorizer = vectorizer
        self.train, self.dev, self.test = DataManager.get_samplers(self.dataset, 0.3, 0.3)
        self._tag_indices = {tags[i]: i for i in range(len(tags))}
    
    def vectorize(self, phrase):
        """
        Converts a phrase into a vector representation.
        
        """
        return self.vectorizer(phrase)
    
    def tag(self, tag_index):
        """
        Returns the tag associated with the given index.
        
        """
        return self.tags[tag_index]
        
    def tag_index(self, tag):
        """
        Returns the index associated with the given tag.
        
        """
        return self._tag_indices[tag]

    def get_sampler(self, partition):
        """
        Returns a Torch sampler for the specified partition id.
        
        Recognized partition ids: 'train', 'dev', 'test'.
        
        """
        if partition == 'train':
            return self.train
        elif partition == 'dev':
            return self.dev
        elif partition == 'test':
            return self.test
        else:
            raise Exception('Unrecognized partition: {}'.format(partition))

    def batched_loader(self, partition, batch_size):        
        """
        Returns a Torch DataLoader for the specified partition id. You must
        specify the size of each batch to load.
        
        Recognized partition ids: 'train', 'dev', 'test'.
        
        """        
        return DataLoader(self.dataset, batch_size=batch_size,
                          sampler=self.get_sampler(partition), num_workers=2)

    
    @staticmethod
    def get_samplers(dataset, dev_percent, test_percent):
        """
        Splits a TaggedPhraseDataset in train, dev, and test partitions,
        according to the specified percentages, then returns
        torch.SubsetRandomSamplers over each partition.
        
        """
        dev_size = int(dev_percent * len(dataset))
        test_size = int(test_percent * len(dataset))
        train_ids = set(range(len(dataset)))
        dev_ids = random.sample(train_ids, dev_size)
        train_ids = train_ids - set(dev_ids)
        test_ids = random.sample(train_ids, test_size)
        train_ids = list(train_ids - set(test_ids))
        train_sampler = SubsetRandomSampler(train_ids)
        dev_sampler = SubsetRandomSampler(dev_ids)
        test_sampler = SubsetRandomSampler(test_ids)
        return train_sampler, dev_sampler, test_sampler


def merge_jsons(json1, json2, out_json):
    """
    Merges the two dictionaries encoded by two JSON files. If json1
    and json2 have the same key, then the value from json2 is kept.

    The resulting dictionary is saved to the file specified by out_json.    
    
    """    
    with open(json1) as f:
        dict1 = json.load(f)
    with open(json2) as f:
        dict2 = json.load(f)
    dict1.update(dict2)
    with open(out_json, 'w') as outhandle:
        outhandle.write(json.dumps(dict1, indent=4, sort_keys=True))
    
   
def compile_db(tags_file, db_file):
    """
    Given a JSON file (tags_file) that encodes a dictionary that maps
    strings to string lists, this 'inverts' the dictionary and writes
    the inversion to another JSON file (db_file).
    
    e.g. if tags_file encodes the dictionary:
       {'a': ['apple','banana','carrot'],
        'b': ['banana', 'bike']}
    then the resulting db_file would encode the dictionary:
        {'apple': ['a'],
         'banana': ['a', 'b'],
         'carrot': ['a'],
         'bike': ['b']
        }
    
    """    
    compiled = defaultdict(set)
    with open(tags_file) as f:
        tags = json.load(f)
    for tag in tags:
        words = tags[tag]
        for word in words:
            compiled[word].add(tag)
    result = {k: sorted(compiled[k]) for k in compiled}
    with open(db_file, 'w') as outhandle:
        outhandle.write(json.dumps(result, indent=4, sort_keys=True))
    
