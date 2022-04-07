import time
import numpy as np
import torch
# import cupy
from problem import problem_creator, error_calculator, shapesLeft
from tetrominoes import shapeI, shapeL, shapeT, shapeSq, shape4, rotator, flip
import multiprocessing

shape_encode = [2,3,5,7,11]
start = time.time()
class Problem:
    def __init__(self, height, breadth):
        self.height = height
        self.breadth = breadth
        self.problem_grid, self.problem_tetraPack = problem_creator(height, breadth)
        self.error = error_calculator(self.problem_grid)
        self.actions_grids = []    
        self.grid_shapes = np.zeros((5,1))
        self.start_grid = []
    def get_dim(self):
        return [self.height, self.breadth]
    def get_grid(self):
        return self.problem_grid
    def get_tetraShape(self):
        return self.problem_tetraPack
    def get_error(self):
        return self.error
    def get_actions(self):
        return self.actions_grids
    def add_action(self, actions):
        self.actions_grids.append(actions)
    def add_shape(self, encode):
        if encode == 2:
            self.grid_shapes[0] = self.grid_shapes[0] + 1
        elif encode == 3:
            self.grid_shapes[1] = self.grid_shapes[1] + 1
        elif encode == 5:
            self.grid_shapes[2] = self.grid_shapes[2] + 1
        elif encode == 7:
            self.grid_shapes[3] = self.grid_shapes[3] + 1
        elif encode == 11:
            self.grid_shapes[4] = self.grid_shapes[4] + 1
    def remove_shape(self, encode):
        if encode == 2:
            self.grid_shapes[0] = self.grid_shapes[0] - 1
        elif encode == 3:
            self.grid_shapes[1] = self.grid_shapes[1] - 1
        elif encode == 5:
            self.grid_shapes[2] = self.grid_shapes[2] - 1
        elif encode == 7:
            self.grid_shapes[3] = self.grid_shapes[3] - 1
        elif encode == 11:
            self.grid_shapes[4] = self.grid_shapes[4] - 1
    def set_maxShape(self):
        for j in self.grid_shapes:
            j[0] = self.height*self.breadth/20
    def get_shapes(self):
        return self.grid_shapes
    def set_grid(self, new_grid):
        self.problem_grid = new_grid
    def get_start(self):
        return self.start_grid
    def set_start(self, start):
        self.start_grid = start
def problem_generator():
    height = 4
    breadth = 10
    problem = Problem(height, breadth)
    shapes_types = problem.get_tetraShape()



def place_shape(shape, grid, i, j):
    for m in range(4):
        for n in range(4):
            grid[0][i+m][j+n] = shape[0][m][n]
            grid[1][i+m][j+n] = shape[1][m][n]
    return grid

# print(place_shape(shapeL,sample_problem.get_grid(),0,0))

# def actions_generator(shape, problem):
#     shapes = []
#     shape_flip = flip(shape)
#     for i in range(4):
#         shapes.append(rotator(shape))
#         shapes.append(rotator(shape_flip))
#     height, breadth = problem.get_dim()
#     grid = problem.get_grid()
#     gridCopy = np.copy(grid)
#     grids = []
#     for s in shapes:
#         for i in range(height):
#             for j in range(breadth):
#                 grid = np.copy(gridCopy)
#                 grids.append(place_shape(s, grid, i, j))
#     # print(grids)
    
#     return grids



def actions_generator(shape, problem):
    shapes = []
    '''Update after flip and rotator has been defined for the new figures (create the for loop and remove the shape from the shapes array)'''
    shape_flip = np.copy(flip(shape))
    flipCopy = np.copy(shape_flip)
    normCopy = np.copy(shape)
    for i in range(4):
        flipCopy = np.copy(rotator(flipCopy))
        normCopy = np.copy(rotator(normCopy))
        shapes.append(np.copy(flipCopy))
        shapes.append(np.copy(normCopy))
    height, breadth = problem.get_dim()
    grid = problem.get_grid()
    gridCopy = np.copy(grid)
    gridCopy[1] = np.ones(grid[1].shape)
    gridCopy[0] = np.zeros(grid[0].shape)
    grids = []
    for s in shapes:
        for i in range(height):
            for j in range(breadth):
                grid = np.copy(gridCopy)
                grids.append(place_shape(s, grid, i, j))
    # print(grids)
    
    return grids

def orientations(shape):
    shapes = []
    '''Update after flip and rotator has been defined for the new figures (create the for loop and remove the shape from the shapes array)'''
    shape_flip = np.copy(flip(shape))
    flipCopy = np.copy(shape_flip)
    normCopy = np.copy(shape)
    for i in range(4):
        flipCopy = rotator(flipCopy)
        normCopy = rotator(normCopy)
        shapes.append(np.copy(shape))
        shapes.append(np.copy(shape_flip))
    print(shapes)

# orientations(shapeL)
# print(rotator(shapeL))

# print(actions_generator(shapeI, sample_problem))

def actions_errors(grids):
    # print(grids)
    errors = []
    for i in grids:
        errors.append(error_calculator(i))
    return errors

# print(actions_errors(actions_generator(shapeL, sample_problem)))

def max_min(grids):
    max = 0
    for grid in grids:
        if error_calculator(grid) > max:
            max = error_calculator(grid)
    min = 1000000
    for grid in grids:
        if error_calculator(grid) < min:
            min = error_calculator(grid)
    return [max, min]

# max, min = max_min(actions_generator(shapeL, sample_problem))
# print(max)
# print(min)

def choice_grid(grids):
    errors = actions_errors(grids)
    denom = np.sum(np.reciprocal(errors))
    prob = np.reciprocal(errors)/denom
    choice = np.random.choice(len(grids), 1 , p = prob)
    return choice

def choice2_grid(grids, num):
    errors = actions_errors(grids)
    min_er = np.min(errors)
    for i in range(num):
        errors[i] = min_er
    denom = np.sum(np.reciprocal(errors))
    prob = np.reciprocal(errors)/denom
    choice = np.random.choice(len(grids), 1 , p = prob)
    return choice

def sampling(grids):
    output_grids = []
    choices = []
    for i in range(100):
        grid_choice = choice_grid(grids)
        choices.append(grid_choice[0])
        # print(grid_choice)
        output_grids.append(grids[grid_choice[0]])
    # print(choices)
    return output_grids

# sampling(sample_actions)

def grid_add(grid1, grid2):
    place_grid1 = grid1[0]
    place_grid2 = grid2[0]
    item_grid1 = grid1[1]
    item_grid2 = grid2[1]
    # print(grid1.shape)
    output_grid = np.zeros(grid1.shape)
    output_grid[0] = place_grid1 + place_grid2
    output_grid[1] = item_grid1 * item_grid2
    return output_grid

# print(grid_add(sample_actions[0], sample_actions[1]))

def check_mul(mul, no):
    if mul % no == 0:
        return True
    else:
        return False

# print(check_mul(101,4))

def shape_code(grid):
    items = np.zeros((5,1))
    mul = 1
    for i in grid[1]:
        for j in i:
            mul = mul * j
    while mul != 1:
        if check_mul(mul, 16):
            items[0] = items[0] + 1
            mul = mul/16
        elif check_mul(mul, 81):
            items[1] = items[1] + 1
            mul = mul/81
        elif check_mul(mul, 625):
            items[2] = items[2] + 1
            mul = mul/625
        elif check_mul(mul, 2401):
            items[3] = items[3] + 1
            mul = mul/2401        
        elif check_mul(mul, 14641):
            items[4] = items[4] + 1
            mul = mul/14641
    return items

# print(shape_code(sample_actions[0]))
    

# print(sample_actions2)

def check_I(grids):
    for grid in grids:
        for i in grid[0]:
            if sum(i)%12 == 0:
                return True
            else:
                return False


# print(check_I([rotator(shapeI)]))

def shape_sum(shape_code):
    return np.sum(shape_code)

# print(shape_sum(shape_code(sample_actions[0])))
    
def random_flip(grid):
    # print(grid)
    new_grid = np.zeros(grid.shape)
    h, b = grid[0].shape
    choice = np.random.choice(range(2), 1)
    # print(choice)
    if choice == 0:
        for i in range(h):
            for j in range(b):
                new_grid[0][i][j] = grid[0][i][b - 1 - j]
                new_grid[1][i][j] = grid[1][i][b - 1 - j]
    elif choice == 1:
        for i in range(h):
            for j in range(b):
                new_grid[0][i][j] = grid[0][h - 1 - i][j]
                new_grid[1][i][j] = grid[1][h - 1 - i][j]
    # print(new_grid)
    return new_grid



def genetic_add(grid1, grid2):
    output_grid = np.zeros(grid1.shape)
    output_grid[1] = np.ones(output_grid[0].shape)
    choice = np.random.choice(range(100), 1)
    grid1C = np.copy(grid1)
    grid2C = np.copy(grid2)
    if choice == 0:
        grid2C = random_flip(grid2C)
    output_grid[0] = grid1C[0] + grid2C[0]
    output_grid[1] = grid1C[1] * grid2C[1]
    return output_grid


# print(sample_actions[0])
# # genetic_add(sample_problem, sample_actions[0], sample_actions2[0])
# sample_problem = Problem(4, 10)
# sample_actions = actions_generator(shapeL, sample_problem)
# print(genetic_add(sample_actions[0], sample_actions[1]))

# print(sample_actions[0])
# random_flip(sample_actions[0])


def sample_creator(grids1, grids2):
    output_grid = []
    samples_grids1 = sampling(grids1)
    samples_grids2 = sampling(grids2)
    size = len(grids1)
    if size < len(grids2):
        size = len(grids2)
    for i in range(int(size*1.5)):
        grid1_choice = choice_grid(samples_grids1)
        grid2_choice = choice_grid(samples_grids2)
        ogrid = grid_add(samples_grids1[grid1_choice[0]], samples_grids2[grid2_choice[0]])
        output_grid.append(ogrid)
    errors = actions_errors(output_grid)
    print(np.min(errors))
    print(np.mean(errors))
    return output_grid
# def genetic_crossover(grids1, grids2):
#     output_grid = []
#     max1, min1 = max_min(grids1)
#     max2, min2 = max_min(grids2)
#     sample_grid1 = sampling(grids1)
#     sample_grid2 = sampling(grids2)
#     for i in range(1000):
#         grid1_choice = choice_grid(sample_grid1)
#         grid2_choice = choice_grid(sample_grid2)
#         ogrid = grid_add(sample_grid1[grid1_choice[0]], sample_grid2[grid2_choice[0]])
#         output_grid.append(ogrid)
'''
sample_actions3 = actions_generator(shapeSq, sample_problem)
sample_actions4 = actions_generator(shape4, sample_problem)
sample_actions5 = actions_generator(shapeT, sample_problem)
out1 = sample_creator(sample_actions, sample_actions2)
out2 = sample_creator(sample_actions3, sample_actions4)
out3 = sample_creator(out1, sample_actions5)
out4 = sample_creator(out2, out3)
# print(actions_errors(out4))
max, min = max_min(out4)
min_array = np.where(actions_errors(out4) == min)
# print(out4[min_array[0][0]])
'''
def sigmoid(probs):
    sum = np.sum(np.exp(probs))
    retVal = []
    for p in probs:
        retVal.append(np.exp(p)/sum)
    return retVal

def test_choice_grid(grids):
    errors = actions_errors(grids)
    max_er = np.max(errors)
    min_er = np.min(errors)
    corr = max_er - min_er
    # print(min_er)
    prob = []
    for error in errors:
        prob.append(corr - error)
    # print(prob)
    # denom = np.sum(np.reciprocal(errors))
    # prob = np.reciprocal(errors)/denom
    prob = sigmoid(prob)
    # print(prob)
    for i in range(100):
        choice = np.random.choice(len(grids), 1 , p = prob)
        # print(errors[choice[0]])
    return choice


# sample_problem = Problem(4, 10)
# sample_actions = actions_generator(shapeL, sample_problem)
# sample_actions2 = actions_generator(shapeI, sample_problem)
# sample_actions3 = actions_generator(shapeSq, sample_problem)
# sample_actions4 = actions_generator(shape4, sample_problem)
# sample_actions5 = actions_generator(shapeT, sample_problem)
# sample_actions6 = actions_generator(shapeL, sample_problem)
# sample_actions7 = actions_generator(shapeI, sample_problem)
# sample_actions8 = actions_generator(shapeSq, sample_problem)
# sample_actions9 = actions_generator(shape4, sample_problem)
# sample_actions10 = actions_generator(shapeT, sample_problem)
# out1 = sample_creator(sample_actions, sample_actions2)
# out2 = sample_creator(sample_actions3, sample_actions4)
# out3 = sample_creator(sample_actions5, sample_actions6)
# out4 = sample_creator(sample_actions7, sample_actions8)
# out5 = sample_creator(sample_actions9, sample_actions10)
# out6 = sample_creator(out1, out2)
# out7 = sample_creator(out3, out4)
# out8 = sample_creator(out5, out6)
# out9 = sample_creator(out7, out8)
# end = time.time()
# print("T1")
# print(end-start)


# print(out9[0])
# print(out9[1])
# print(out9[100])


# print(out4[test_choice_grid(out4)[0]])

def grid3_add(shape, grid, code, i, j):
    problem_grid = grid
    # problem.add_shape(code)
    for m in range(4):
        for n in range(4):
            problem_grid[0][i+m][j+n] = problem_grid[0][i+m][j+n] + shape[0][m][n]
            problem_grid[1][i+m][j+n] = problem_grid[1][i+m][j+n] * shape[1][m][n]
    # problem.set_grid(problem_grid)
    return problem_grid


# grid_add(shapeL, sample_problem, 2, 2, 3)
# print(sample_problem.get_grid())

def grid_remove(grid, code):
    height, breadth = grid[1].shape
    numShape = height*breadth/20
    dim = [] 
    dims = []
    prob_grid = grid
    # print(prob_grid)
    remove_grid = np.zeros(prob_grid.shape)
    remove_grid[1] = np.ones(prob_grid[1].shape)
    h, b = prob_grid[0].shape
    for k in range(int(numShape)):
        for i in range(h):
            for j in range(b):
                if prob_grid[1][i][j] % code == 0:
                    dims.append([i, j])
                    prob_grid[1][i][j] = int(prob_grid[1][i][j])/code
                    prob_grid[0][i][j] = prob_grid[0][i][j] - 1
                    remove_grid[1][i][j] = int(remove_grid[1][i][j]) * code
                    remove_grid[0][i][j] = remove_grid[0][i][j] + 1
    # print(dims)
    return remove_grid

# sample_problem.set_grid(out4[1])

# print(sample_problem.get_grid())
# sample_problem.set_maxShape()
# # print(sample_problem.get_shapes())
# rem1 = grid_remove(out4[1], 2)
# print(rem1) 
# rem2 = grid_remove(sample_problem, 3)
# print(genetic_add(rem1, rem2))



# print(out4[1])
# print(error_calculator(out4[0]))
# print(error_calculator(out4[1]))
# ogrid = grid_add(out4[0], out4[1])
# print(error_calculator(ogrid))

def check_sh(grid):
    sum = 0
    for i in grid[0]:
        for j in i:
            sum = sum + j
    return sum
def check_shapes(grids):
    numShapes = []
    for grid in grids:
        numShapes.append(check_sh(grid)/4)
    return numShapes 

# print(check_shapes(out4))

def grid2_add(grid1, grid2):
    choice = []
    # print(grid1)
    
    option_grids = []
    options_error = []
    # print(error_calculator(grid1))
    # print(error_calculator(grid2))
    
    for i in range(3):
        ogrid = np.zeros(grid1.shape)
        ogrid[1] = np.ones(grid1[1].shape)
        choice = []
        for i in range(5):
            choice.append(np.random.choice(range(2), 1))
        for i in range(5):
            if choice[i][0] == 0:
                ogrid = genetic_add(ogrid, grid_remove(grid1, shape_encode[i]))
            elif choice[i][0] == 1:
                ogrid = genetic_add(ogrid, grid_remove(grid2, shape_encode[i]))
        option_grids.append(ogrid)
        options_error.append(error_calculator(ogrid)) 
        # print(error_calculator(ogrid))
    # print(np.where(options_error == np.min(options_error))[0])
    ogrid = option_grids[np.where(options_error == np.min(options_error))[0][0]]    
    # print(error_calculator(ogrid))
    return ogrid
        # if error_calculator(grid1) > error_calculator(grid2):
        #     return grid2
        # else:
        #     return grid1
# g1 = np.copy(out4[1])
# g2 = np.copy(out4[2])
# print(error_calculator(g1))
# print(error_calculator(g2))
# for i in range(10):
#     g1 = np.copy(out4[1])
#     g2 = np.copy(out4[2])
#     grid2_add(g1, g2)


def genetic_crossover(grids, luckGenes):
    output_grid = []
    grid_errors = actions_errors(grids)
    min_array = np.where(grid_errors == np.min(grid_errors))[0][0]
    # if len(min_array) < 5:
    #     count = len(min_array)
    # else:
    #     count = 5
    # for i in range(5):
    low_grid = grids[min_array]
    output_grid.append(low_grid)
    for i in luckGenes:
        output_grid.append(i)
    sample_grid1 = sampling(grids)
    sample_grid2 = sampling(grids)
    size = len(grids)
    # print(check_shapes(sample_grid2))
    while len(output_grid) != 500:
        grid1_choice = choice2_grid(sample_grid1, len(luckGenes))
        grid2_choice = choice2_grid(sample_grid2, len(luckGenes))
        grid1 = np.copy(sample_grid1[grid1_choice[0]])
        grid2 = np.copy(sample_grid2[grid2_choice[0]])
        ogrid = grid2_add(grid1, grid2)
        # print(check_sh(ogrid))
        output_grid.append(ogrid)
    return output_grid 

def lucky_survivors(grids):
    lucky_ind = np.random.choice(range(len(grids)), 1)
    luckyGenes = []
    for i in lucky_ind:
        luckyGenes.append(grids[i])
    retVal = np.copy(luckyGenes)
    return retVal

# genetic_crossover(out4)
# mins = []
# sample_grids = np.copy(out9)
# good_grids = []
# for i in range(1):
#     sample_grids = np.copy(out9)
#     for sth in range(25):
#         sample_grids = genetic_crossover(sample_grids, lucky_survivors(out9))
#         # print(check_shapes(sample_grids))
#         error = actions_errors(sample_grids)
#         print(np.mean(error))
#         print(np.min(error))
#         print(sample_grids[0])
    #     mins.append(np.min(error))
    # for j in sample_grids:
    #     good_grids.append(j)

# better_grids = []
# final_grid = genetic_crossover(good_grids, lucky_survivors(out9))
# for j in final_grid:
#     better_grids.append(j)
# for sth in range(10):
#     for i in range(10):
#         final_grid = genetic_crossover(final_grid, lucky_survivors(good_grids))
#         errors = actions_errors(final_grid)
#         print(np.mean(errors))
#         print(np.min(errors))
#         mins.append(np.min(errors))
#     for j in final_grid:
#         better_grids.append(j)

# ffinal_grid = genetic_crossover(better_grids, lucky_survivors(final_grid))
# errors = actions_errors(ffinal_grid)
# print(np.mean(errors))
# print(np.min(errors))
# mins.append(np.min(errors))

# print(mins)


def parser(problem):
    grids = problem.get_actions()
    # print(len())
    if len(grids) == 1:
        max, min = max_min(grids[0])
        # print(min)
        errors = actions_errors(grids[0])
        min_array = np.where(errors == min)[0]
        # print(grids[0][min_array[0]])
        problem.set_start(grids[0])
        return grids[0]
    elif len(grids) == 0:
        time.sleep(1)
    else:
        problem.add_action(sample_creator(grids.pop(0), grids.pop(0)))
        parser(problem)
def parser2(problem, shape):
    grids = actions_generator(shape, problem)
    problem.add_action(grids)


if __name__ == '__main__':
    # starttime = time.time()
    start = time.time()
    problem = Problem(4, 10)
    shapes = []
    for shape_type in problem.get_tetraShape():
        for shape in shape_type:
            shapes.append(shape)        

    processes = []
    for shape in shapes:
        problem.add_action(actions_generator(shape, problem))

    # for process in processes:
    #     process.join()
    grids = problem.get_actions()
    # processes = []
    
    parser(problem)
    # p = multiprocessing.Process(target=parser, args=(problem, processes))
    # print(p.)
    # processes.append(p)
    # p.start()
    # smth = grids
    # parser(grids)
    # while len(grids) != 1:
    #     grids.append(genetic_crossover(grids.pop(), grids.pop()))
    # print(actions_errors(grids[0]))
    
    # for i in range(12):
    #     p = multiprocessing.Process(target=parser, args=(problem,))
    #     processes.append(p)
    #     p.start()
        
    

    grid = problem.get_actions().pop(0)
    mins = []
    sample_grids = np.copy(grid)
    good_grids = []
    for i in range(1):
        sample_grids = np.copy(grid)
        for sth in range(10):
            sample_grids = genetic_crossover(sample_grids, lucky_survivors(grid))
            # print(check_shapes(sample_grids))
            error = actions_errors(sample_grids)
            print(np.mean(error))
            print(np.min(error))
            # print(error)
            # print(sample_grids[0])
            max_er = np.max(error)
            print(max_er)
            # sth = sample_grids[np.where(error == max_er)[0][0]]
            # print(sth)
            mins.append(np.min(error))
        for j in sample_grids:
            good_grids.append(j)
    end = time.time()
    print("T2")
    print(end-start)