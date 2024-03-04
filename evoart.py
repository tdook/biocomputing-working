#!/usr/bin/env python3
"""
Genetic algorithm implemented with Evol solving the one max problem
(maximising number of 1s in a binary string).

"""
import random

from PIL import ImageChops
from PIL import Image, ImageDraw
from evol import Population, Evolution


SIDES = 3
POLYGON_COUNT = 100

MAX = 255 * 200 * 200
TARGET = Image.open("6a.png")
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
  x1 = random.randrange(11,189)
  y1 = random.randrange(11,189)
  x2 = random.randrange(11, 189)
  y2 = random.randrange(11, 189)
  x3 = random.randrange(11, 189)
  y3 = random.randrange(11, 189)

  return[(r,g,b,a),(x1,y1), (x2,y2), (x3,y3)]


def initialise():
  return [make_polygon() for i in range(POLYGON_COUNT)]




def evolve(population, args):
  print("evolving")

  mutate(solution="solution.png",rate=0.1)
  print("mutating")
  if(population.current_best):
    mutate()
    draw(population.current_best.chromosome).save("solution.png")

  exit()


def select(population):
  return [random.choice(population) for i in range(2)]


def combine(*parents):
  return [a if random.random() < 0.5 else b for a, b in zip(*parents)]

def mutate(solution, rate):
  solution = list(solution)

  if random.random() < 0.5:
    # mutate points
    i = random.randrange(len(solution))
    polygon = list(solution[i])
    coords = [x for point in polygon[1:] for x in point]
    coords = [x if random.random() > rate else
              x + random.normalvariate(0, 10) for x in coords]
    coords = [max(0, min(int(x), 200)) for x in coords]
    polygon[1:] = list(zip(coords[::2], coords[1::2]))
    solution[i] = polygon
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





population = Population.generate(initialise, evaluate, size=10, maximize=True)
draw(population[0].chromosome).save("solution.png")

