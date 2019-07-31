'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-public-methods
import random
import unittest

from liv_learn.optimisation import gen_alg


class TestChromosome(unittest.TestCase):
    '''Class to test Chromosome class.'''

    def test_breed(self):
        '''Test for breed method.'''
        length = 16
        chrom1 = gen_alg.Chromosome(0, length)
        chrom2 = gen_alg.Chromosome(2 ** length - 1, length)

        for _ in range(1024):
            chrom1.breed(chrom2)

        self.assertEqual(
            chrom1.get_chromosome() + chrom2.get_chromosome(), 2 ** length - 1)


class TestSumGeneticAlgorithm(unittest.TestCase):
    '''Class to test SumGeneticAlgorithm class.'''

    def test_run(self):
        '''Test for run method.'''
        args = dict(enumerate([[5, 10]] + [(random.randint(0, 20),
                                            random.randint(80, 100))
                                           for _ in range(10)]))
        target = 321
        genetic_algorithm = SumGeneticAlgorithm(100, args, target)
        result, optimised = genetic_algorithm.run(100000)
        self.assertTrue(optimised)
        self.assertEqual(sum(result.values()), target)

    def test_run_insoluable(self):
        '''Test for run method.'''
        args = dict(enumerate([[5, 10]] + [(random.randint(0, 20),
                                            random.randint(80, 100))
                                           for _ in range(10)]))
        target = 936073
        genetic_algorithm = SumGeneticAlgorithm(100, args, target)
        _, optimised = genetic_algorithm.run(10)
        self.assertFalse(optimised)


class SumGeneticAlgorithm(gen_alg.GeneticAlgorithm):
    '''Class to run a genetic algorithm.
    Basic implementation involves calculating a set of numbers that sum to
    a given target.'''

    def __init__(self, pop_size, args, target):
        super(SumGeneticAlgorithm, self).__init__(pop_size, args)
        self.__target = target

    def _fitness(self, individual):
        '''Determine the fitness of an individual.'''
        return abs(self.__target - sum(individual.values()))
