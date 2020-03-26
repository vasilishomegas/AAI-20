from functools import reduce
from operator import mul
import copy
import random


def fitness_function(pile_0, pile_1):
    return -abs(((pile_0-36)*10+(pile_1-360)))


class Genotype:
    def __init__(self):
        self.bitstring = 0
        self.fitness = None

    def calculate_fitness(self):
        sum36 = 0
        mul360 = 1
        for i in range(10):
            if (self.bitstring >> i) & 1:
                mul360 *= i
            else:
                sum36 += i
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
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.current_evolution = []
        for _ in range(batch_size):
            genotype = Genotype()
            genotype.bitstring = random.randint(0, 0b1111111111)
            self.current_evolution.append(genotype)
        for genotype in self.current_evolution:
            genotype.calculate_fitness()
        self.current_evolution = sort_genotypes(self.current_evolution)
        print(self.current_evolution[0].fitness)

    def evolve(self, amount=100):
        for _ in range(amount):
            for genotype in self.current_evolution:
                genotype.calculate_fitness()
            # Make sure those of the population with the best fitness are ranked first
            self.current_evolution = sort_genotypes(self.current_evolution)
            # The best 20%, in our case, will automatically qualify for the next generation
            next_generation = copy.deepcopy(self.current_evolution[:int(len(self.current_evolution)*0.2)])
            # We mutate the next 60%, hoping it will lead to random improvement
            mutate_generation = copy.deepcopy(mutate(self.current_evolution[int(len(self.current_evolution)*0.2):int(len(self.current_evolution)*0.8)], 4))
            # We apply crossover and mutation to the bottom 20%, with crossover being done with the best 20%
            crossover_generation = copy.deepcopy(mutate(crossover(self.current_evolution[:int(len(self.current_evolution)*0.2)], 1), 1))
            self.current_evolution = next_generation + mutate_generation + crossover_generation

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
            genotype.set_bit(nr, (1 ^ genotype.get_bit(nr)))
    return batch


def crossover(batch, mutations):
    # oneway crossover function
    for _ in range(len(batch)):
        # choose two genotype for crossover
        genotype = batch[random.randint(0, len(batch) - 1)]
        other_genotype = batch[random.randint(0, len(batch) - 1)]
        for _ in range(mutations):
            # pick a random bit
            i = random.randint(0, 10)
            # take the value of that bit from one genotype and write it to the other
            genotype.set_bit(i, other_genotype.get_bit(i))
    return batch


def sort_genotypes(genotypes: [Genotype]):
    result = [genotypes[0]]
    for genotype in genotypes[1:]:
        for i in range(len(result)):
            if result[i].fitness < genotype.fitness:
                result = result[:i] + [genotype] + result[i:]
                break
            if i == len(result) - 1:
                result.append(genotype)
    for genotype in result:
        print(genotype.fitness)
    print()
    return result


def main():
    ea = Evolutionary_Algortithm()
    print(ea.evolve()[0].fitness)


if __name__ == '__main__':
    main()
