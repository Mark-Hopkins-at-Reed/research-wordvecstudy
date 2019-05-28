import random
from data import DataManager
from nltk.corpus import wordnet as wn
from wordnet import specificity, get_specific_synsets, WordnetDataSource
from twolayer import SimpleClassifier, train_net, evaluate
from word2vec import GOOGLENEWS_MODEL
w2v = GOOGLENEWS_MODEL



def run_binary_classification(datasource, tag1, tag2, verbose = True):    
    """
    Trains a binary classifier to distinguish between TaggedPhraseDataSource
    phrases tagged with tag1 and phrases tagged with tag2.
    
    This returns the accuracy of the binary classifier on the test
    partition.
    
    """    
    vectorizer = lambda p: w2v.get_word_vector(p)
    phrase_recognizer = lambda p: vectorizer(p) is not None
    dmanager = DataManager(datasource, 
                           [tag1, tag2], 
                           vectorizer,
                           phrase_recognizer)
    classifier = SimpleClassifier(300,100,2)
    net = train_net(classifier, dmanager,
                    batch_size=32, n_epochs=30, learning_rate=0.001,
                    verbose=False)
    acc, misclassified = evaluate(net, dmanager, 'test')
    if verbose:        
        for tag in sorted(dmanager.tags):
            print('{} phrases are tagged with "{}".'.format(
                    dmanager.num_phrases[tag], tag))
        print('\nERRORS:')
        for (phrase, guessed, actual) in sorted(misclassified):
            print('"{}" classified as "{}"\n  actually: "{}".'.format(
                    phrase, guessed, actual))
        print("\nOverall test accuracy = {:.2f}".format(acc))
    return acc
    
def wordnet_experiment(min_spec, max_spec, num_iters):
    """
    Randomly chooses two synsets from Wordnet that have specificity within
    the given bounds, then trains a binary classifier to distinguish
    between the hyponyms of these synsets.
    
    """
    senses = [hyp.name() for hyp in get_specific_synsets(min_spec, max_spec)]
    for _ in range(num_iters):
        [tag1, tag2] = random.sample(senses, 2)
        acc = run_binary_classification(WordnetDataSource(), 
                                        tag1, tag2, verbose=False)
        print("{} vs. {}: {:.3f}".format(tag1, tag2, acc))
            
        
def wordnet_sibling_experiment(min_spec, max_spec, num_iters):
    """
    Randomly chooses a synset from Wordnet that has specificity within
    the given bounds, then trains a binary classifier to distinguish
    between two random child synsets of this synset.
    
    The children must have specificity greater than 50.
    
    """   
 
    def random_siblings(min_sp, max_sp):
        synsets = list(get_specific_synsets(min_sp, max_sp)) 
        siblings = set()
        while len(siblings) < 2:
            parent = random.choice(synsets)
            children = [hyp for hyp in wn.synset(parent).hyponyms() if 
                        specificity(hyp.name()) >= 50]
            if len(children) >= 2:
                siblings = random.sample(children, 2)
        return [s.name() for s in siblings]

    for _ in range(num_iters):
        [tag1, tag2] = random_siblings(min_spec, max_spec)
        acc = run_binary_classification(WordnetDataSource(), tag1, tag2)
        print("{} vs. {}: {:.3f}".format(tag1, tag2, acc))
            
