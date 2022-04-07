import torch
import numpy as np 
from tetrominoes import packTetra

def problem_creator(height, breadth):
    if (height*breadth<4):
        print("Illegible size")
        return
    dimensions = (2, height + 3, breadth + 3)
    
    grid = np.zeros(dimensions)
    grid[1] = np.ones((height+3, breadth+3))
    # grid = torch.from_numpy(old_grid)
    setTetra = height*breadth/20
    shapes = packTetra(int(setTetra))
    # print(grid)
    return [grid, shapes]

def shapesLeft(shapes):
    sum = 0
    for i in shapes:
        sum = sum + len(i)
    return sum
problem_creator(5,4)

def error_calculator(input_grid):
    depth, height, breadth = input_grid.shape
    height = height - 3
    breadth = breadth - 3
    grid = input_grid[0]
    subtractor = np.ones((height + 3, breadth + 3))
    for i in range(3):
        subtractor[:,breadth + i] = 0
        subtractor[height + i, :] = np.zeros((1, breadth + 3))
    error_grid = grid - subtractor
    sum = 0
    x = 0
    y = 0
    for i in error_grid:
        y = 0
        if x >= height:
            sum = sum + 2*np.sum(i**2)
        else:
            for j in i:
                if y >= breadth:
                    sum = sum + 2*(j**2)
                else:
                    if j != -1:
                        if j != 0:
                            sum = sum + (j+1)**3
                    else:
                        sum = sum + 10
                y = y + 1
        x = x +1
    return sum
# grid = np.zeros((2,7,13))
# print(error_calculator(grid))

# for i in range(4):
#     for j in range(10):
#         grid[0][i][j] = 1
# print(error_calculator(grid))
# print(error_calculator(problem_creator(5,10)[0]))
# print(shapesLeft(problem_creator(5,4)[1]))