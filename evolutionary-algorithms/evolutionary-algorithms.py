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
        self.piles = ([], [])
        self.fitness = None

    def calculate_fitness(self):
        self.fitness = fitness_function(add36(self.piles[0]), mult360(self.piles[1]))

class Evolutionary_Algortithm:
    def __init__(self, evolution_function, batch_size=10):
        self.evolution_function = evolution_function
        self.batch_size = batch_size
        self.current_evolution = []
        for _ in range(batch_size):
            genotype = Genotype()
            for i in range(1, 11):
                genotype.piles[random.randint(0, 1)].append(i)
            self.current_evolution.append(genotype)
        map(Genotype.calculate_fitness, self.current_evolution)


    def evolve(self, amount=100):
        for _ in range(amount):
            self.current_evolution = self.evolution_function(self.current_evolution, self.batch_size)
            map(Genotype.calculate_fitness, self.current_evolution)
        return self.current_evolution


def evolution_function_0(evolution, batch_size):
    evolution.sort(key=Genotype.fitness)
    survivors = evolution[:int(batch_size*0.2)]

    return

def main():
    ea = Evolutionary_Algortithm(evolution_function_0)
    print(ea.evolve())

if __name__ == '__main__':
    main()
