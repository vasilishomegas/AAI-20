from functools import reduce
from operator import mul
import random


def fitness_function(pile_0, pile_1):
    return -(((pile_0-36)*10+(pile_1-360))**2)


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


class Evolutionary_Algortithm:
    def __init__(self, evolution_function, batch_size=10):
        self.evolution_function = evolution_function
        self.batch_size = batch_size
        self.current_evolution = []
        for _ in range(batch_size):
            genotype = Genotype()
            for i in range(10):
                genotype.bitstring &= (random.randint(0, 1) << i)
            self.current_evolution.append(genotype)
        map(Genotype.calculate_fitness, self.current_evolution)

    def evolve(self, amount=100):
        for _ in range(amount):
            self.current_evolution = self.evolution_function(self.current_evolution, self.batch_size)
            map(Genotype.calculate_fitness, self.current_evolution)
        return self.current_evolution


def mutate(batch, mutations):
    for genotype in batch:
        # randomly move a number from one side to the other
        for _ in range(mutations):
            nr = random.randint(0, 10)
            genotype.bitstring &= (1 ^ ((genotype.bitstring >> nr) & 1)) << nr


def sort_genotypes(genotypes: [Genotype]):
    result = genotypes[0]
    for genotype in genotypes[1:]:
        for i in range(result):
            if result[i].fitness < genotype.fitness:
                result = result[:i] + [genotype] + result[i:]
                break
            if i == len(result) - 1:
                result.append(genotype)


def evolution_function_0(evolution, batch_size):

    survivors = evolution[:int(batch_size*0.2)]


    return

def main():
    ea = Evolutionary_Algortithm(evolution_function_0)
    print(ea.evolve())


if __name__ == '__main__':
    main()
