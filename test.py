import unittest

from wordnet import hypernym_chain, get_all_hypernym_synsets
from wordnet import get_all_hyponyms, Specificity, get_specific_synsets
from wordnet import find_lowest_common_ancestor, word_based

specificity = Specificity()

class WordnetTests(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        pass
        
    def tearDown(self):
        """Call after every test case."""
        pass
    
    def test_hypernym_chain(self):
        actual = hypernym_chain('boat.n.01')
        expected = ['boat.n.01', 
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
        assert actual == expected
        
    def test_get_all_hypernym_synsets(self):
        actual = get_all_hypernym_synsets('boat.n.01')
        expected = { 'artifact.n.01',
                     'conveyance.n.03',
                     'craft.n.02',
                     'entity.n.01',
                     'instrumentality.n.03',
                     'object.n.01',
                     'physical_entity.n.01',
                     'vehicle.n.01',
                     'vessel.n.02',
                     'whole.n.02'}
        assert actual == expected
        
    def test_get_all_hyponyms(self):
        actual = get_all_hyponyms('noble_metal.n.01')
        expected = { '24-karat_gold',
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
        assert actual == expected

    def test_specificity(self):
        assert specificity('noble_metal.n.01') == 15
        
    def test_get_specific_synsets(self):
        actual = get_specific_synsets(202, 203)
        expected = {'climber.n.01', 'fastener.n.02', 'solid.n.03'}
        assert actual == expected
        
    def test_find_lowest_common_ancestor(self):
        actual = find_lowest_common_ancestor(['communist', 
                                              'democrat', 
                                              'republican'])
        assert actual == (60, 'politician.n.02')
        actual = find_lowest_common_ancestor(['apple', 
                                              'banana', 
                                              'orange', 
                                              'grape'])        
        assert actual == (297, 'edible_fruit.n.01')
        
    def test_word_based(self):
        get_hypernyms = word_based(get_all_hypernym_synsets)
        actual = get_hypernyms('wagon')
        expected = { 'artifact.n.01',
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
        assert actual == expected

if __name__ == "__main__":
    unittest.main() # run all tests