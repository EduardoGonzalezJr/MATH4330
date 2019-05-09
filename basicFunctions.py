import copy



#works
def two_norm(vector):
    """
    Computes the two norm of a vector

    This function sums the squares of the elements of the vector, then takes the square root of the sum

    Args:
        vector: a list of numbers as a vector

    Returns:
        A scalar which is the 2-norm of the given vector
    """
    result = 0
    for element in range(len(vector)):
        result = result + (vector[element] ** 2)
    result = result ** (1 / 2)
    print(result)
    return result



#works
def conjugate(complex):
    """
    Computes the conjugate of a complex number

    The function takes a complex number, stores the real value of the complex number into the result, and result is subtracted by
    the imaginary part of the complex number

    Args:
        complex: A complex number (e.g. 4+2j)

    Returns:
        The conjugate of a complex number, as a scalar
    """
    result = complex.real
    result = result - (complex.imag)
    return result


#works
def transpose(matrix):
    """
    Computes the transpose of the matrix

    This function takes a matrix, creates a result matrix and a temporary list, and loops through the matrix and stores the row of
    the matrix into temp and appends it to the result matrix

    Args:
        matrix: A matrix represented as a list of column vectors

    Returns:
        The transpose of the matrix whose rows are now the columns, represented as a list of column vectors
    """
    result = []
    for i in range(len(matrix[0])):
        temp = []
        for element in range(len(matrix)):
            temp.append(matrix[element][i])
        result.append(temp)
    return result


complexMatrix = [[1+2j,2+1j,3], [4,5,6], [7,8,9]]
matrix = [[1, 2], [3, 4], [5, 6]]


#works
def conjugateTranspose(matrix):
    """
    Computes the conjugate transpose of a matrix

    This function takes a matrix, computes the transpose of the matrix, then iterates through each element and conjugates the value

    Args:
        matrix: A matrix represented as a list of column vectors

    Returns:
        The conjugate transpose of a matrix represented as a list of column vectors
    """
    result = transpose(matrix)
    for iterator in range(len(matrix)):
        for element in range(len(matrix[0])):
            result[element][iterator] = conjugate(result[element][iterator])
    return result



#works
def vandermonde4(vector):
    """
    Computes the vandermonde matrix from a vector

    This function takes a vector, creates a temporary list and stores the respective element to its respective exponent, then appends that
    temporary list to the result matrix

    Args:
        vector: a list of numbers as a vector
    Returns:
        a vandermonde matrix of degree 4 as a list column vectors
    """
    result = []
    for exponent in range(5):
        temp = []
        for element in range(len(vector)):
            temp.append(vector[element] ** exponent)
        result.append(temp)
    return result



def scalarVectorMulti(scalar, vector):
    """
    Computes the multiplication of a scalar and a vector

    This function takes a scalar and a vector, and multiplies each element in the vector by the sclar.

    Args:
        scalar: a scalar value
        vector: a list of numbers as a vector

    Returns:
        A vector whose elements have been multiplied by the scalar value
    """
    result = vector
    for iterator in range(len(vector)):
        result[iterator] *= scalar
    return result


def dotProduct(vector1, vector2):
    """
    Computes the dot product of two vectors

    This function takes two vectors and multiplies each element with the corresponding element in the other vector

    Args:
        vector1: a list of numbers as a vector
        vector2: a list of numbers as a vector

    Returns:
        a scalar which is the dot product of the two vectors
    """
    result = 0
    for element in range(len(vector1)):
        result += (vector1[element] * vector2[element])
    return result


def zeroMatrix(matrix):
    """
    Computes the zero matrix with the same dimensions as the matrix passed in

    This function takes a matrix, has a result matrix do a deep copy of the matrix, then sets every element in the
    result matrix to 0

    Args:
        A matrix that is represented as a list of column vectors

    Returns:
        A zero matrix as a list of column vectors
    """
    result = copy.deepcopy(matrix)
    index = -1
    for column in matrix:
        index += 1
        for element in range(len(column)):
            result[index][element] = 0
    return result


def printSystem(A, b):
    """
    prints the system of equations in augmented matrix form
    """
    for i in range(len(A)):
        print(str(A[i]) + " | " + str(b[i]))

#A is a list of row vectors
#works
def gaussianEliminate(A, b):
    """
    Runs gaussian elimination on the system of equations represented by matrix A and vector b

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

#A is a list of row vectors
#works
def backSubstitute(A, b):
    """
        after gaussian elimination, this solves the system using back-substitution
    """
    gaussianEliminate(A, b)
    x = copy.deepcopy(b)

    for i in range(len(A) - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, len(A)):
            x[i] = x[i] - A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return x


A = [
    [1, 2, 1, -1],
    [3, 2, 4, 4],
    [4, 4, 3, 4],
    [2, 0, 1, 5]
]
b = [5, 16, 22, 15]

x = backSubstitute(A, b)
print("Answer: " + str(x))
