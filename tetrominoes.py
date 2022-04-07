import numpy as np

shape_code = [2,3,5,7,11]

shapeL = np.zeros((2,4,4))

shapeJ = np.zeros((2,4,4))
shapeT = np.zeros((2,4,4))
shapeI = np.zeros((2,4,4))
shapeSq = np.zeros((2,4,4))
shape4 = np.zeros((2,4,4))
shape4r = np.zeros((2,4,4))
# shapeL[0][1][1] = [0]

def init():
    shapeL[0][0][0] = 1
    shapeL[0][1][0] = 1
    shapeL[0][2][0] = 1
    shapeL[0][2][1] = 1 
    
    shapeI[0][0][0] = 1
    shapeI[0][1][0] = 1
    shapeI[0][2][0] = 1
    shapeI[0][3][0] = 1

    shapeJ[0][0][1] = 1
    shapeJ[0][1][1] = 1
    shapeJ[0][2][1] = 1
    shapeJ[0][2][0] = 1

    shapeT[0][0][0] = 1
    shapeT[0][0][1] = 1
    shapeT[0][0][2] = 1
    shapeT[0][1][1] = 1

    shapeSq[0][0][0] = 1
    shapeSq[0][1][0] = 1
    shapeSq[0][0][1] = 1
    shapeSq[0][1][1] = 1

    shape4[0][0][0] = 1
    shape4[0][1][0] = 1
    shape4[0][1][1] = 1
    shape4[0][2][1] = 1

    shape4r[0][0][1] = 1
    shape4r[0][1][0] = 1
    shape4r[0][2][0] = 1
    shape4r[0][1][1] = 1

    shapeL[1] = np.ones((4,4))
    shapeI[1] = np.ones((4,4))
    shapeJ[1] = np.ones((4,4))
    shapeT[1] = np.ones((4,4))
    shapeSq[1] = np.ones((4,4))
    shape4[1] = np.ones((4,4))
    shape4r[1] = np.ones((4,4))


    shapeL[1][0][0] = 2
    shapeL[1][1][0] = 2
    shapeL[1][2][0] = 2
    shapeL[1][2][1] = 2

    shapeI[1][0][0] = 3
    shapeI[1][1][0] = 3
    shapeI[1][2][0] = 3
    shapeI[1][3][0] = 3

    shapeJ[1][0][1] = 13
    shapeJ[1][1][1] = 13
    shapeJ[1][2][1] = 13
    shapeJ[1][2][0] = 13

    shapeT[1][0][0] = 5
    shapeT[1][0][1] = 5
    shapeT[1][0][2] = 5
    shapeT[1][1][1] = 5

    shapeSq[1][0][0] = 7
    shapeSq[1][1][0] = 7
    shapeSq[1][0][1] = 7
    shapeSq[1][1][1] = 7

    shape4[1][0][0] = 11
    shape4[1][1][0] = 11
    shape4[1][1][1] = 11
    shape4[1][2][1] = 11

    shape4r[1][0][1] = 17
    shape4r[1][1][0] = 17
    shape4r[1][2][0] = 17
    shape4r[1][1][1] = 17
init()
def add(shape1, shape2):
    # print(shape1)
    output_shape = np.zeros((2,4,4))
    output_shape[1] = np.ones((4,4))
    for i in range(4):
        for j in range(4):
            if shape1[0][i][j] == 1 or shape2[0][i][j] == 1:
                output_shape[1][i][j] = shape1[1][i][j] * shape2[1][i][j]
                output_shape[0][i][j] = shape1[0][i][j] + shape2[0][i][j]
    return output_shape

def packTetra(k):
    shape_L = []
    shape_4 = []
    shape_I = []
    shape_Sq = []
    shape_T = []
    for i in range(k):
        shape_L.append(shapeL)
        shape_I.append(shapeI)
        shape_4.append(shape4)
        shape_Sq.append(shapeSq)
        shape_T.append(shapeT)
    shapes = []
    shapes.append(shape_L)
    shapes.append(shape_I)
    shapes.append(shape_4)
    shapes.append(shape_Sq)
    shapes.append(shape_T)
    # print(shapes)
    return shapes

def reassignS(shape):
    while np.sum(shape[0]) == 0:
        shape[0] = shape[1]
        shape[1] = shape[2]
        shape[2] = shape[3]
        shape[3] = np.zeros((1,4))
    # print(shape)
    return shape

def reassignI(sh):
    shape = np.copy(sh)
    while np.sum(shape[0]) == 4:
        shape[0] = shape[1]
        shape[1] = shape[2]
        shape[2] = shape[3]
        shape[3] = np.ones((1,4))
    # print(shape)
    return shape

def rotator(sh):
    shape = np.copy(sh)
    shape[0] = np.rot90(shape[0])
    shape[0] = reassignS(shape[0])
    
    shape[1] = np.rot90(shape[1])
    shape[1] = reassignI(shape[1])
    # print(shape)
    return shape

def flip(sh):
    shape = np.copy(sh)
    shape[0] = np.flipud(shape[0])
    shape[0] = reassignS(shape[0])
    
    shape[1] = np.flipud(shape[1])
    shape[1] = reassignI(shape[1])
    
    # print(shape)
    return shape

# print(shapeL)
# rotator(rotator(shapeL))
# flip(shapeL)

# def orientations(shape):
#     shapes = []
#     '''Update after flip and rotator has been defined for the new figures (create the for loop and remove the shape from the shapes array)'''
#     shape_flip = np.copy(flip(shape))
#     flipCopy = np.copy(shape_flip)
#     normCopy = np.copy(shape)
#     for i in range(4):
#         flipCopy = np.copy(rotator(flipCopy))
#         normCopy = np.copy(rotator(normCopy))
#         shapes.append(np.copy(flipCopy))
#         shapes.append(np.copy(normCopy))
#     print(shapes)

# orientations(shapeL)