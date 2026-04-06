from binaryneat.population import Population
from binaryneat.config import get_config
import numpy as np
import pickle as pkl
from tabulate import tabulate as tb
from tqdm import tqdm
import multiprocessing as mp


def save_genome(genome, filename):
    with open(filename, 'wb') as f:
        pkl.dump(genome, f)


def evaluate_genome(args):
    g, ff = args
    fitness = ff(g)
    return fitness


def evaluate_population(pop, ff):
    print('\n')
    with mp.Pool(mp.cpu_count() - 1) as p:
        try:
            fitness_scores = list(tqdm(
                p.imap(evaluate_genome, [(g, ff) for g in pop.genomes]),
                total=len(pop.genomes),
                desc='[-] Evaluating',
                leave=False
            ))
            for g, fitness in zip(pop.genomes, fitness_scores):
                g.fitness = fitness
        except KeyboardInterrupt:
            p.terminate()
            p.join()
            return  # exit early


def save_checkpoint(pop):
    """Save with temporary callback stripping (pickle fails on bound methods)."""
    cb = pop.callbacks
    pop.callbacks = None
    for g in pop.genomes:
        g.callbacks = None
        for n in g.network.nodes:
            n.callbacks = None

    try:
        with open('checkpoint.pkl', 'wb') as f:
            pkl.dump(pop, f)
    finally:
        # restore callbacks
        pop.callbacks = cb
        for g in pop.genomes:
            g.callbacks = cb
            for n in g.network.nodes:
                n.callbacks = cb

    print(f'[+] Checkpoint saved (gen. {pop.generation})')


def load_checkpoint():
    try:
        with open('checkpoint.pkl', 'rb') as f:
            pop = pkl.load(f)

        # Re-attach callbacks after unpickling
        pop.callbacks = {
            'find_or_create_innovation': pop.find_or_create_innovation,
            'get_next_genome_id': pop.get_next_genome_id,
            'get_next_species_id': pop.get_next_species_id,
            'config': pop.config
        }
        for g in pop.genomes:
            g.callbacks = pop.callbacks
            for n in g.network.nodes:
                n.callbacks = pop.callbacks

        print(f'[+] Loaded checkpoint (gen. {pop.generation})')
        return pop
    except FileNotFoundError:
        return None


def print_stats(pop):
    # sort by fitness
    for s in pop.species:
        s.members = sorted(s.members, key=lambda x: x.fitness, reverse=True)
    species = sorted(pop.species, key=lambda x: x.members[0].fitness if x.members else 0, reverse=True)

    # print general stats
    best_ever = pop.best_genome_seen.fitness if pop.best_genome_seen else 'N/A'
    print(f'\n[i] Generation: {pop.generation}')
    print(f'[i] Compatibility threshold: {round(pop.compatibility_threshold, 2)}')
    print(f'[i] Population size: {len(pop.genomes)}')
    print(f'[i] Species: {len(pop.species)}')
    print(f'[i] Average fitness: {round(np.mean([g.fitness for g in pop.genomes]), 2)}')
    print(f'[i] Best fitness: {round(max(g.fitness for g in pop.genomes), 2)} (best ever: {best_ever})')
    print('-' * 88)

    # print species
    headers = ['Species', 'Members', 'Best Fitness', 'Average Fitness', 'Stagnation', 'Best Complexity']
    data = []
    for s in species:
        if not s.members:
            data.append([s.id, 0, 0, 0, s.stagnation, '0n + 0c'])
            continue
        best_c = s.members[0]
        data.append([
            s.id,
            len(s.members),
            round(max(g.fitness for g in s.members), 2),
            round(np.mean([g.fitness for g in s.members]), 2),
            s.stagnation,
            f'{len(best_c.network.nodes)}n + {len(best_c.network.connections)}c'
        ])
    print(tb(data, headers=headers))
    print('-' * 88)


def evolve(fitness_function):
    config = get_config()

    pop = load_checkpoint() or Population()

    max_generations = config.getint('Evolution', 'max_generations') or float('inf')
    max_fitness = config.getfloat('Evolution', 'max_fitness') or float('inf')

    try:
        while True:
            # evaluate population
            evaluate_population(pop, fitness_function)

            # print stats
            print_stats(pop)

            # reproduce
            print(f'[-] Reproducing...', end='\r', flush=True)
            pop.reproduce()
            print(f'[+] Reproduced                                   ')

            # save checkpoint
            if pop.generation % 10 == 0:
                save_checkpoint(pop)

            # update best ever
            genomes = sorted(pop.genomes, key=lambda x: x.fitness, reverse=True)
            best = genomes[0]

            if pop.best_genome_seen is None or best.fitness > pop.best_genome_seen.fitness:
                new_best = best.clone()
                print(f'\n\n[+] New best genome found with fitness: {round(new_best.fitness, 2)} '
                      f'(previous was {round(pop.best_genome_seen.fitness, 2) if pop.best_genome_seen else "N/A"})')
                pop.best_genome_seen = new_best

            # termination conditions
            if best.fitness >= max_fitness or pop.generation >= max_generations:
                winner = best
                save_genome(winner, 'winner.pkl')
                if best.fitness >= max_fitness:
                    print(f'\n\n[+] Winner found with fitness: {winner.fitness}\n\n')
                else:
                    print(f'\n\n[+] Reached max generations, achieved fitness: {winner.fitness}\n\n')
                return winner

    except KeyboardInterrupt:
        print(f'\n\n[+] Best genome saved, with a fitness of '
              f'{pop.best_genome_seen.fitness if pop.best_genome_seen else "N/A"}\n')
        if pop.best_genome_seen:
            save_genome(pop.best_genome_seen, 'winner.pkl')
        return pop.best_genome_seen