'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
import random

import numpy


class Chromosome(object):
    '''Class to represent a chromosome.'''

    def __init__(self, chromosome, len_chromosome, mutation_rate=0.01):
        self.__chromosome = chromosome
        self.__len_chromosome = len_chromosome
        self.__mask = 2 ** (self.__len_chromosome) - 1
        self.__mutation_rate = mutation_rate

    def get_chromosome(self):
        '''Gets the chromosome.'''
        return self.__chromosome

    def mutate(self):
        '''Mutates the chromosome.'''
        for i in range(self.__len_chromosome):
            if random.random() < self.__mutation_rate:
                # Mutate:
                mask = 1 << i
                self.__chromosome = self.__chromosome ^ mask

    def breed(self, partner):
        '''Breeds chromosome with a partner Chromosome.'''
        i = int(random.random() * self.__len_chromosome)
        end = 2 ** i - 1
        start = self.__mask - end
        return (partner.get_chromosome() & start) + (self.__chromosome & end)

    def __repr__(self):
        return format(self.__chromosome, '0' + str(self.__len_chromosome) +
                      'b') + '\t' + str(self.__chromosome)


class GeneticAlgorithm(object):
    '''Base class to run a genetic algorithm.'''

    def __init__(self, pop_size, args, retain=0.2, random_select=0.05,
                 mutate=0.01, verbose=False):
        self._args = args
        self._verbose = verbose
        self.__pop_size = pop_size
        self.__retain = retain
        self.__random_select = random_select
        self.__mutate = mutate
        self.__pop = []

        while len(list(numpy.unique(numpy.array(self.__pop)))) < pop_size:
            self.__pop.append(self._get_individual())

    def run(self, max_iter=1024, max_tries=1024):
        '''Runs the genetic algorithm.'''
        for _ in range(max_iter):
            result, optimised = self.__evolve(max_tries)

            if optimised:
                break

        return result, optimised

    def _fitness(self, individual):
        '''Determine the fitness of an individual.'''
        pass

    def _get_individual(self):
        '''Create a member of the population.'''
        return {key: self._get_arg(key) for key in self._args}

    def _get_arg(self, key, individual=None):
        '''Gets a random argument.'''
        args = self._args[key]

        return random.randint(args[0], args[1]) if isinstance(args, tuple) \
            else random.choice(args)

    def _procreate(self, male, female):
        '''Procreate.'''
        pos = random.randint(0, len(male))

        male_parts = {k: male[k]
                      for i, k in enumerate(male.keys())
                      if i < pos}
        female_parts = {k: female[k]
                        for i, k in enumerate(female.keys())
                        if i >= pos}

        return dict(male_parts.items() + female_parts.items())

    def __evolve(self, max_tries):
        '''Performs one round of evolution.'''
        graded = sorted([(self._fitness(x), x) for x in self.__pop])

        if graded[0][0] == 0:
            return graded[0][1], True

        if self._verbose:
            print graded[0]

        graded = [x[1] for x in graded]
        retain_length = int(self.__pop_size * self.__retain)

        # Retain best and randomly add other individuals to promote genetic
        # diversity:
        self.__pop = graded[:retain_length] + \
            [ind for ind in graded[retain_length:]
             if self.__random_select > random.random()]

        # Mutate some individuals:
        for individual in self.__pop[retain_length:]:
            if self.__mutate > random.random():
                key = random.choice(individual.keys())

                if key in self._args:
                    individual[key] = self._get_arg(key, individual)

        # Ensure uniqueness in population:
        self.__pop = list(numpy.unique(numpy.array(self.__pop)))

        try:
            self.__breed(max_tries)
        except ValueError:
            return graded[0], False

        return graded[0], False

    def __breed(self, max_tries):
        '''Breeds parents to create children.'''
        tries = 0

        while len(self.__pop) < self.__pop_size:
            male = random.choice(self.__pop)
            female = random.choice(self.__pop)

            if male != female:
                self.__pop.append(self._procreate(male, female))
                self.__pop = list(numpy.unique(numpy.array(self.__pop)))

            tries += 1

            if tries == max_tries:
                raise ValueError('Unable to generate unique population.')
