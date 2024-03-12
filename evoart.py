#!/usr/bin/env python3
"""
Genetic algorithm implemented with Evol solving the one max problem
(maximising number of 1s in a binary string).

"""
import math
import random

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np


import evol
from PIL import ImageChops
from PIL import Image, ImageDraw
from evol import Population, Evolution


SIDES = 3
POLYGON_COUNT = 200
INIT_SIGMA = 125
DECAY_RATE = 0.05
CURRENT_GEN = 0

MAX = 255 * 200 * 200
TARGET = Image.open("6b.png")
TARGET.load()  # read image and close the file


def evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX

def make_polygon():

  r = random.randrange(0,256)
  g = random.randrange(0,256)
  b = random.randrange(0,256)
  a = random.randrange(30,60)


  coords = [(random.randrange(11,189),random.randrange(11,189)) for _ in range (SIDES)]


  return[(r,g,b,a)]+ coords
  #return[(r,g,b,a),(x1,y1), (x2,y2), (x3,y3),(x4,y4) ]



def initialise():
  return [make_polygon() for i in range(POLYGON_COUNT)]




def evolve(population, args, cxp=0.5, mutp=0.5):
    population.survive(fraction=0.5)
    population.breed(parent_picker=select, combiner=combine)
    population.mutate(mutate_function=mutate, rate=0.185)
    #CHANGE MUTATE rate depending on fitness being up or down
    return population

def roulette_selection(population):
    total_fitness = sum(individual.fitness for individual in population)
    selection_point = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for individual in population:
        cumulative_fitness += individual.fitness
        if cumulative_fitness >= selection_point:
            return individual


def select(population):
  #print("selected")
  return [random.choice(population) for i in range(2)]
  #return [roulette_selection(population) for _ in range(2)]



def combine(*parents):
  #print("combined")
  return [a if random.random() < 0.7 else b for a, b in zip(*parents)]



def updated_std_dev(init_std_dev, generation, total_generations):
    return init_std_dev * (1 - generation / total_generations)

def mutate(solution, rate):
  global INIT_SIGMA, DECAY_RATE, CURRENT_GEN
  solution = list(solution)

  current_sigma = INIT_SIGMA / math.log(CURRENT_GEN + 3, 3)

  #len solution < max poly count and random.rand < p:
  if random.random() < 0.3:
      solution.append(make_polygon())
      return solution
  #generate new polys

  if random.random() < 0.5:
    # mutate points
    i = random.randrange(len(solution))
    polygon = list(solution[i])
    coords = [x for point in polygon[1:] for x in point]
    coords = [x + random.uniform(-0.5, 0.5) if random.random() > rate else x for x in coords]
    #coords = [x if random.random() > rate else
           #   x + random.normalvariate(0,current_sigma) for x in coords]


    coords = [max(0, min(int(x), 200)) for x in coords]
    polygon[1:] = list(zip(coords[::2], coords[1::2]))
    solution[i] = polygon

  if random.random() < 0.2:
      i = random.randrange(len(solution))
      polygon = list(solution[i])
      colors = [x for color in polygon[1:] for x in color]
      colors = [x if random.random() > rate else
                x + random.normalvariate(0,10) for x in colors]
      colors = [max(0, min(int(x), 255)) for x in colors]
      polygon[1:] = list(zip(colors[::2], colors[1::2]))
      solution[i] = polygon




#if random.random() < 0.2:
   #   for _ in range(random.randint(1, len(solution) // 2)):
  else:
    # reorder polygons
    random.shuffle(solution)


  return solution


def draw(solution):
  image = Image.new("RGB", (200, 200))
  canvas = ImageDraw.Draw(image, "RGBA")
  for polygon in solution:
    canvas.polygon(polygon[1:], fill=polygon[0])



  return image


# Generate values for generations
generations = np.arange(1, 100)
sigma_values = INIT_SIGMA / np.log(generations + 2)

def plot_graph(generations, sigma_values):

    plt.plot(generations, sigma_values)
    plt.xlabel('Generation')
    plt.ylabel('Current Sigma')
    plt.title('Decay of Sigma over Generations')
    plt.grid(True)
    plt.show()




plot_graph(generations,sigma_values)
