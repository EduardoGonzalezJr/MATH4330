import copy

"""
What do we have? (Input)
    A vector of length n
What do we want? (output)
    A scalar (which is the z-norm of our vector)
How do we get there?
    Sum the squares of the elements of our vector, then take square
    root of our sum. 

Calculates the 2-norm of a vector

Sums the squares of the elemtents of a given vector and returns the root of the sum

Args:
    Vector: A list of numbers representing a vector.
Returns:
    A scalar which is the 2-norm of the given vector.
"""


def two_norm(vector):
    result = 0
    for element in range(len(vector)):
        result = result + (vector[element] ** 2)
    result = result ** (1 / 2)
    # print(result)
    return result


test_vector01 = [2, 2, 2, 2]

two_norm(test_vector01)

########################################################
"""
Write and implement an algorithm which takes a vector and 
returns the normalized version (w/ respect to z-norm)

x, x/||x||
"""
"""
Pseudo Code
def normalize(vector):
    norm = two_norm(vector)
    if (norm == 0):
        print("Invalid input")
    elif (norm == 1):
        return vector
    else:
        return scalar-vector-multiply((1/norm), vector)
"""

"""
Args:
    scalar: A number
    vector: A list of numbers
"""


def normalize(vector):
    """Normalizes a given vector

    Checks to see if the vector is normal or the zero vector. If not
    the vector is divided by its norm.

    Args:
        Vector: A list of numbers representing a vector.
    Returns:
        A normalized vector if the input vector was not the zero vector.
        Prints an error otherwise.
    """

    norm = two_norm(vector)
    if (norm == 0):
        print("Invalid input")
    elif (norm == 1):
        return vector
    else:
        return scalarVecMulti((1 / norm), vector)  # what is that function?


########################################################################################

"""
Write and implement an algorithm to compute matrix vector multiplication

What do we have? (Input)
    An mxn matrix and an n vector

What do we want? (Output)
    An m vector representing the matrix vector multiplication

How do we get there? (algorithm)
    Multiplying each column of the matrix by the corresponding element of the vector


Pseudo code
    def matVec(matrix, vector)
        result = [0] * len(matrix[0])
        for element in range(len(vector)):
            result = vectorAdd(result, scalarVecMulti(vector[element], matrix[element]))
        return result

"""

"""
Write and implement an algorithm to compute the conjugate

What do we have?
    - A complex number

What do we want?
    - The conjugate of the complex number

How do we want to get there?
    - We will get the real number of the complex number and subtract it with the imaginary number 
"""


def conjugate(complex):
    result = complex.real
    result = result - (complex.imag)
    return result


# print(conjugate(3+3j))


def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        temp = []
        for element in range(len(matrix)):
            temp.append(matrix[element][i])
        result.append(temp)
    return result


# matrix = [[1+2j,2+1j,3], [4,5,6], [7,8,9]]
matrix = [[1, 2], [3, 4], [5, 6]]
print(transpose(matrix))


# print(transpose(transpose(matrix)))


def conjugateTranspose(matrix):
    result = transpose(matrix)
    for iterator in range(len(matrix)):
        for element in range(len(matrix[0])):
            result[element][iterator] = conjugate(result[element][iterator])
    return result


print("conjugate transpose")
print(conjugateTranspose(matrix))


# returns a vector columns
def vandermonde4(vector):
    result = []
    for exponent in range(5):
        temp = []
        for element in range(len(vector)):
            temp.append(vector[element] ** exponent)
        result.append(temp)
    return result


print(vandermonde4([2, 3, 4]))


def backsub(matrix, vector):
    result = vector
    for iterator in range(len(matrix[0])):
        print("result is")
        print(result)
        alpha = (len(matrix[0]) - 1)
        sum = 0
        for k in range(((alpha - iterator) + 1), (len(matrix) - 1)):
            sum += (matrix[k][alpha - iterator] * result[k])
        print("Changing result")
        result[alpha - iterator] = (vector[alpha - iterator] - sum) * (1 / matrix[alpha - iterator][alpha - iterator])
    return result


def scalarVectorMulti(scalar, vector):
    result = vector
    for iterator in range(len(vector)):
        result[iterator] *= scalar
    return result


testMatrix = [[3, 2, 1], [0, 1, 2], [0, 0, 2]]
testVector = [4, 6, 8]
# expected answer: [-11, 3, 0]

print(backsub(testMatrix, testVector))


# print(scalarVectorMulti(2, [1,2,3]))

def dotProduct(vector1, vector2):
    result = 0
    for element in range(len(vector1)):
        result += (vector1[element] * vector2[element])
    return result


def zeroMatrix(matrix):
    result = copy.deepcopy(matrix)
    index = -1
    for column in matrix:
        index += 1
        for element in range(len(column)):
            result[index][element] = 0
    return result


# print(dotProduct([1,2,3], [4,5,6]))

def vectorSubtraction(vector1, vector2):
    for element in range(len(vector1)):
        vector1[element] -= vector2[element]
    return vector1


vector1 = [4, 5, 6]
vector2 = [1, 2, 3]
print("subtraction")
print(vectorSubtraction(vector1, vector2))
print("everything below this is mgs")


# def modifiedGS(A):
#     v = copy.deepcopy(A)
#     r = zeroMatrix(A)
#     q = copy.deepcopy(A)
#     for j in range(len(A)):
#         r[j][j] = two_norm(v[j])
#         q[j] = scalarVectorMulti((1/r[j][j]), v[j])
#         for k in range((j+1), len(A)):
#             r[j][k] = dotProduct(q[j], v[k])
#             v[k] = vectorSubtraction(v[k], scalarVectorMulti(r[j][k], q[k]))
#     print("q is ")
#     print(q)
#     print("R is ")
#     print(r)
#
#
# notSquareMatrix = [[2,2], [-2,1], [18,0]]
# squareMatrix = [[2,2,1], [-2,1,2], [18,0,0]]
# modifiedGS(notSquareMatrix)

# print(modifiedGS([[1,2,3], [4,5,6], [7,8,9]]))

def printSystem(A, b):
    """
    prints the system of equations in augmented matrix form
    """
    for i in range(len(A)):
        print(str(A[i]) + " | " + str(b[i]))


def gaussianEliminate(A, b):
    """
    runs gaussian elimination on the system of equations represented by members A and b of this class
    """
    # do the gaussian elimination
    # note this changes the needed upper-triangular elements of A, but does not change the lower-triangular elements
    m = 0
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            m = A[j][i] / A[i][i]
            for k in range(i + 1, len(A)):
                A[j][k] = A[j][k] - m * A[i][k]
            b[j] = b[j] - m * b[i]
    # actually zero out the lower-triangular elements
    for i in range(1, len(A)):
        for j in range(0, i):
            A[i][j] = 0


def backSubstitute(A, b):
    gaussianEliminate(A, b)
    """
    after gaussian elimination, this solves the system using back-substitution
    """
    for i in range(len(A) - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, len(A)):
            x[i] = x[i] - A[i][j] * x[j]
        x[i] = x[i] / A[i][i]


if __name__ == "__main__":
    A = [
        [1, 2, 1, -1],
        [3, 2, 4, 4],
        [4, 4, 3, 4],
        [2, 0, 1, 5]
    ]
    b = [5, 16, 22, 15]
    x = [0, 0, 0, 0]

    backSubstitute(A, b)
    print("Answer: " + str(x))
