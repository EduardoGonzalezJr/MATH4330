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




#A is a list of row vectors
#works
def backSubstitute(A, b):
    """
        this solves the system using back-substitution

        Args:
            A: a matrix represented as a list of row vectors
            b: a vector represented as a list of numbers
        Returns:
                x: a vector represented as a list of numbers for which Ax=b
    """
    x = copy.deepcopy(b)
    for i in range(len(A) - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, len(A)):
            x[i] = x[i] - A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return x


#works
def printMatrix(A):
    """
    prints out a given matrix in a nice format

    prints out the matrix given using some special formatting. It first must do a deep copy of the matrix and compute the transpose
    of the matrix in order for it to print out properly

    Args:
        A matrix represented as a list of column vectors

    Returns:
        Nothing. It does print out a matrix in a nice format
    """
    copyA = copy.deepcopy(A)
    copyA = transpose(copyA)
    print('\n'.join([''.join(['{:13.5g}'.format(item) for item in row])
      for row in copyA]))
#works
def printQorR(matrix):
    """
    prints out a given matrix, Q or R, in a nice format

    prints out the matrix given using some special formatting

    Args:
        A matrix, Q or R, represented as a list of column vectors

    Returns:
        Nothing. It does print out a matrix in a nice format
    """
    print('\n'.join([''.join(['{:13.5g}'.format(item) for item in row])
        for row in matrix]))


#works
def vectorSubtraction(x,y):
    """
    Computes the subtraction between two vectors

    This function takes two vectors, deep copies one of the vectors to result, and iterates over the vectors subtracts the first vector passed in from
    the second vector and stores it into result

    Args:
        x: a list of numbers as a vector
        y: a list of numebrs as a vector

    Returns:
        A vector that is the result from the vector subtraction of  x - y
    """
    result = copy.deepcopy(x)
    for element in range(len(x)):
        result[element] = x[element] - y[element]
    return result

#works
def matrixVectorMult(A, x):
    """
    Computes the multiplication of a matrix and a vector

    This function takes a matrix a vector, creates a result vector with the length of x, iterates through the columns of the
    matrix and iterates through the elements of each column, and stores the result of the current element of A times the current
    element of x into the current element of result

    Args:
        A: a matrix represented as a list of column vectors
        x: a list of numbers as a vector

    Returns:
        a vector as a list of numbers
    """
    result = [0]*len(x)
    for i in range(len(A)):
        result[i]=0
        for j in range(len(A[0])):
            result[i] = result[i] + (A[i][j] * x[j])
    return result



def gsMod(A):
    """
    Computes Q and R from A using Modified Gram-Schmidt

    This function uses orthogonal decomposition and normalization to compute Q and R

    Args:
        A: A matrix represented as a list of column vectors

    Returns:
        Q: A unitary matrix represented as a list of column vectors
        R: an upper triangular matrix represented as a list of column vectors
    """
    length = len(A)
    R = [0] * length
    Q = [0] * length
    V = [0] * length

    for i in range(length):
        R[i] = [0] * length
        Q[i] = [0] * length
        V[i] = [0] * length
    for j in range(len(A)):
        V[j] = A[j]
    for j in range(len(V)):
        R[j][j] = two_norm(V[j])
        Q[j] = scalarVectorMulti((1 / (R[j][j])), V[j])

        for k in range(j + 1, len(V)):
            R[j][k] = dotProduct(Q[j], V[k])
            x = scalarVectorMulti(R[j][k], Q[j])
            V[k] = vectorSubtraction(V[k], x)
    print("Q (actual):")
    printQorR(Q)
    print("R (actual):")
    printQorR(R)
    return Q, R


def d4interpolation(B):
    """
    computes the degree 4 interpolations

    Take the summation of every column of B timest the vector x with its corresponding exponent

    Args:
        B: A vector represented as a list of numbers
    Returns:
        Nothing. It prints out the polynomial of degree 4 of B
    """
    for i in range(len(B),5):
        B.append(0);
    print((B[4]),"x**4 +",B[3],"x**3+",B[2],"x**2+",B[1],"x+",B[0])
