from functools import reduce
from operator import mul
import random


def fitness_function(pile_0, pile_1):
    return abs(((pile_0-36)*10+(pile_1-360))**2)


def add36(lst):
    return sum(lst)


def mult360(lst):
    return reduce(mul, lst)


class Genotype:
    def __init__(self):
        self.bitstring = 0
        self.fitness = None

    def calculate_fitness(self):
        bitmask = 0
        sum36 = 0
        mul360 = 1
        for i in range(self.bitstring.bit_length()):
            cur_val = (self.bitstring >> bitmask) & 1
            if cur_val:
                mul360 *= (bitmask + 1)
            else:
                sum36 += (bitmask + 1)
        self.fitness = fitness_function(sum36, mul360)

    def set_bit(self, index, value):
        mask = 1 << index
        self.bitstring &= ~mask
        if value:
            self.bitstring |= mask
        return self.bitstring

    def get_bit(self, index):
        return self.bitstring >> index & 1

class Evolutionary_Algortithm:
    def __init__(self, evolution_function, batch_size=10):
        self.evolution_function = evolution_function
        self.batch_size = batch_size
        self.current_evolution = []
        for _ in range(batch_size):
            genotype = Genotype()
            for i in range(10):
                genotype.bitstring |= (random.randint(0, 1) << i)
            self.current_evolution.append(genotype)
        for genotype in self.current_evolution:
            genotype.calculate_fitness()
     #   for genotype in self.current_evolution:
     #       print(genotype.fitness)
        self.current_evolution = sort_genotypes(self.current_evolution)
       # for genotype in self.current_evolution:
       #         print(genotype.fitness)

    def evolve(self, amount=100):
        for _ in range(amount):
            self.current_evolution = self.evolution_function(self.current_evolution, self.batch_size)
            for genotype in self.current_evolution:
                genotype.calculate_fitness()
            self.current_evolution = sort_genotypes(self.current_evolution)

        return self.current_evolution


def mutate(batch, mutations):
    for genotype in batch:
        # randomly move a number from one side to the other
        for _ in range(mutations):
            nr = random.randint(0, 10)
            # isolate a (random) bit, flip it and put it back in place
            # not 100% tested
            genotype.bitstring |= (1 ^ ((genotype.bitstring >> nr) & 1)) << nr


def crossover(batch, mutations=1, survivor_rate=0.2):
    new_batch = []
    target_len = len(batch)
    current_batch = batch[:int(len(batch)*survivor_rate)]
    batch_len = len(current_batch)
    while len(new_batch) < target_len:

        genotype = current_batch[random.randint(0, batch_len - 1)]
        other_genotype = current_batch[random.randint(0, batch_len - 1)]
        for _ in range(mutations):
            i = random.randint(0, 10)
            genotype.set_bit(i, other_genotype.get_bit(i))
        new_batch.append(genotype)
    return new_batch




def sort_genotypes(genotypes: [Genotype]):
    result = [genotypes[0]]
    for genotype in genotypes[1:]:
        for i in range(len(result)):
            if result[i].fitness > genotype.fitness:
                result = result[:i] + [genotype] + result[i:]
                break
            if i == len(result) - 1:
                result.append(genotype)
    return result


def evolution_function_0(evolution, batch_size):

    survivors = evolution[:int(batch_size*0.2)]


    return

def main():
    ea = Evolutionary_Algortithm(crossover, 300)
    best = ea.evolve()
    for genotype in best:
        print(genotype.fitness)

if __name__ == '__main__':
    main()
