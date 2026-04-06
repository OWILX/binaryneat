import numpy as np
import time
from .species import Species
from .config import get_config
from .genome import Genome

class Population:
    def __init__(self):
        self.config = get_config()
        self.innovations = []
        self.genome_counter = 0
        self.species_counter = 0
        self.species = []
        self.generation = 0
        self.compatibility_threshold = self.config.getfloat('Population', 'compatibility_threshold')
        self.best_genome_seen = None
        self.callbacks = {
            'find_or_create_innovation': self.find_or_create_innovation,
            'get_next_genome_id': self.get_next_genome_id,
            'get_next_species_id': self.get_next_species_id,
            'config': self.config
        }

        self.initialize()

    @property
    def genomes(self):
        return [genome for species in self.species for genome in species.members]

    def initialize(self):
        genomes = [g for g in [Genome(self.callbacks) for _ in range(self.config.getint('Population', 'population_size'))]]
        self.speciate(genomes)

    def reproduce(self):
        elite_size = self.config.getint('Population', 'elite_size')
        min_species_size = self.config.getint('Population', 'min_species_size')
        population_size = self.config.getint('Population', 'population_size')
        survival_threshold = self.config.getfloat('Population', 'survival_threshold')

        offspring = []

        # normalize fitness scores
        min_fitness = min(g.fitness for g in self.genomes)
        max_fitness = max(g.fitness for g in self.genomes)
        if max_fitness - min_fitness == 0:
            max_fitness += 0.0001 # avoid division by zero
        for g in self.genomes:
            g.normalized_fitness = (g.fitness - min_fitness) / (max_fitness - min_fitness)

        for s in self.species:

            # bump stagnation
            if s.members[0].fitness > s.best_fitness:
                s.best_fitness = s.members[0].fitness
                s.stagnation = 0
            else:
                s.stagnation += 1

            # assign adjusted fitness scores
            s_size = len(s.members)
            for g in s.members:
                g.adjusted_fitness = max(g.normalized_fitness / s_size, 0.0001) # avoid division by zero

        # remove stagnant species
        while len(self.species) > self.config.getint('Evolution', 'min_species'):
            stagnant_species = [s for s in self.species if s.stagnation >= self.config.getint('Evolution', 'max_stagnation')]
            stagnant_species = sorted(stagnant_species, key=lambda x: x.best_fitness, reverse=True)
            if not stagnant_species:
                break
            extinct = stagnant_species.pop()
            extinct.members = sorted(extinct.members, key=lambda x: x.fitness, reverse=True)

            # preserve the elite
            elite = extinct.members[:elite_size]
            offspring.extend([g.clone() for g in elite])

            self.species.remove(extinct)
            print(f'[i] Species {extinct.id} went extinct due to stagnation')

        # perform reproduction inside of each species
        for s in self.species:
            s_offspring = []

            # copy elite as-is
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            elite = s.members[:elite_size]
            s_offspring.extend([g.clone() for g in elite])

            # kill of worst performing members
            if len(s.members) > min_species_size + len(elite):
                s.members = s.members[:int(len(s.members) * survival_threshold + 1)]

            # calculate allowed offspring
            total_fitness = sum(g.adjusted_fitness for g in self.genomes)
            s_fitness = sum(g.adjusted_fitness for g in s.members)
            allowed_offspring = (s_fitness / total_fitness) * population_size
            allowed_offspring = int(max(allowed_offspring, min_species_size))

            if len(s.members) <= 1:
                continue

            # breed the rest
            total_adjusted_fitness = sum(g.adjusted_fitness for g in s.members)
            selection_probabilities = [g.adjusted_fitness / total_adjusted_fitness for g in s.members]
            while len(s_offspring) < allowed_offspring:
                parent1 = np.random.choice(s.members, p=selection_probabilities)
                parent2 = np.random.choice(s.members, p=selection_probabilities)
                child = Genome.crossover(parent1, parent2)
                child.mutate()
                child.id = self.get_next_genome_id()
                s_offspring.append(child)
                
            offspring.extend(s_offspring)

        # add genomes if we're below the population size
        while len(offspring) < population_size:
            g = np.random.choice(offspring).clone()
            g.mutate()
            g.id = self.get_next_genome_id()
            offspring.append(g)


        # re-speciate
        self.generation += 1
        self.speciate(offspring)

    def speciate(self, genomes=None):
        unspeciated = genomes or self.genomes
        new_reps = {}
        new_members = {}

        # find a representative for each species
        for s in self.species:
            candidates = []
            for g in unspeciated:
                d = self.measure_genetic_distance(s.representative, g)
                candidates.append((d, g))
            
            _, new_rep = min(candidates, key=lambda x: x[0])
            new_reps[s.id] = new_rep
            new_members[s.id] = [new_rep]
            unspeciated.remove(new_rep)

        while unspeciated:
            g = unspeciated.pop()
            candidates = []
            
            for sid, rep in new_reps.items():
                d = self.measure_genetic_distance(rep, g)
                if d < self.compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                d, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(g)
            else:
                new_species = Species(g, self.callbacks)
                new_reps[new_species.id] = g
                new_members[new_species.id] = [g]
                self.species.append(new_species)

        for s in self.species:
            s.members = new_members[s.id]
            s.representative = new_reps[s.id]

        # adjust compatibility threshold
        if len(self.species) < self.config.getint('Evolution', 'target_species'):
            self.compatibility_threshold *= 0.97
        else:
            self.compatibility_threshold = self.config.getfloat('Population', 'compatibility_threshold')
            

    def find_or_create_innovation(self, in_nodes, out_node):
        innovation = next((i for i in self.innovations 
                          if set(n.id for n in i.in_nodes) == set(n.id for n in in_nodes) 
                          and i.out_node.id == out_node.id), None)
        if not innovation:
            innovation = Innovation(len(self.innovations), in_nodes, out_node)
            self.innovations.append(innovation)
        return innovation.innovation_number
    def get_next_genome_id(self):
        self.genome_counter += 1
        return self.genome_counter

    def get_next_species_id(self):
        self.species_counter += 1
        return self.species_counter

    @staticmethod
    def measure_genetic_distance(g1, g2):
        # constants
        c1 = c2 = 1.0
        c3 = 0.6

        # Connection genes (innovation-based)
        g1_conns = {c.innovation_number: c for c in g1.network.connections}
        g2_conns = {c.innovation_number: c for c in g2.network.connections}

        matching = g1_conns.keys() & g2_conns.keys()
        disjoint = len(g1_conns.keys() ^ g2_conns.keys())
        excess = len(g1_conns) + len(g2_conns) - 2 * len(matching)
        N = max(len(g1_conns), len(g2_conns)) or 1

        # Gate differences (G) on nodes matched by .id
        g1_nodes = {n.id: n for n in g1.network.nodes}
        g2_nodes = {n.id: n for n in g2.network.nodes}
        matching_nodes = g1_nodes.keys() & g2_nodes.keys()
        G = 0.0
        if matching_nodes:
            gate_diff = sum(1 for nid in matching_nodes if g1_nodes[nid].gate != g2_nodes[nid].gate)
            G = gate_diff / len(matching_nodes)

        return (c1 * excess + c2 * disjoint) / N + c3 * G

class Innovation:
    def __init__(self, innovation_number, in_nodes, out_node):
        self.innovation_number = innovation_number
        self.in_nodes = in_nodes
        self.out_node = out_node