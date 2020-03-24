from functools import reduce

def fitness_fucntion(pile_0, pile_1):
    return -(((pile_0-36)*10+(pile_1-360))**2)

def add36(lst):
    return sum(lst)
def mult360(lst):
    return reduce((lambda x, y: x * y), lst)

