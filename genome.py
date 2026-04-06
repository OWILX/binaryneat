from .neuralnetwork import BinaryNeuralNetwork
from .config import get_config
import numpy as np
from copy import deepcopy
import time
import pickle

class Genome:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.fitness = 0
        self.normalized_fitness = 0
        self.adjusted_fitness = 0
        self.id = callbacks['get_next_genome_id']()
        self.network = BinaryNeuralNetwork(callbacks)

    @staticmethod
    def crossover(g1, g2):
        most_fit, least_fit = (g1, g2) if g1.fitness > g2.fitness else (g2, g1)
        child = most_fit.clone()

        # dict for fast lookup of least-fit nodes
        least_fit_nodes = {n.id: n for n in least_fit.network.nodes}

        # Only inherit gates (no weights/biases exist)
        for n in child.network.nodes:
            if n.id in least_fit_nodes:
                n.gate = np.random.choice([n.gate, least_fit_nodes[n.id].gate])

        return child

    def mutate(self):
        mutation_rates = {
            'add_node': self.callbacks['config'].getfloat('MutationRates', 'add_node'),
            'modify_connection': self.callbacks['config'].getfloat('MutationRates', 'modify_connection'),
            'change_gate': self.callbacks['config'].getfloat('MutationRates', 'change_gate'),
            'swap_input_nodes': self.callbacks['config'].getfloat('MutationRates', 'swap_input_nodes')
        }

        total = sum(mutation_rates.values())
        mutation_rates = {k: v / total for k, v in mutation_rates.items()}
        mutation = np.random.choice(list(mutation_rates.keys()), p=list(mutation_rates.values()))

        mutation_functions = {
            'add_node': self.network.add_random_node,
            'modify_connection': self.network.modify_random_connection,
            'change_gate': self.network.change_random_gate,
            'swap_input_nodes': self.network.swap_random_input_nodes
        }
        
        mutation_functions[mutation]()

    def clone(self):
        copy = pickle.loads(pickle.dumps(self))
        return copy

    def activate(self, inputs):
        '''
        alias for network.feed_forward
        '''
        return self.network.feed_forward(inputs)