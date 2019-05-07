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
# def two_norm(vector):
#
#
#     result = 0
#     for element in range(len(vector)):
#         result = result + (vector[element]**2)
#     result = result**(1/2)
#     print(result)
#     return result
#
# test_vector01 = [2,2,2,2]
#
# two_norm(test_vector01)

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
        return scalarVecMulti((1/norm), vector) #what is that function?
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

print(conjugate(3+3j))



def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        temp = []
        for element in range(len(matrix)):
            temp.append(matrix[element][i])
        result.append(temp)
    return result

matrix = [[1+2j,2+1j,3], [4,5,6], [7,8,9]]
# print(transpose(matrix))
# print(transpose(transpose(matrix)))


def conjugateTranspose(matrix):
    result = transpose(matrix)
    for iterator in range(len(matrix)):
        for element in range(len(matrix[0])):
            result[iterator][element] = conjugate(result[iterator][element])
    return result

print(conjugateTranspose(matrix))


def vandermonde4(vector):
    result = []
    for exponent in range(5):
        temp = []
        for element in range(len(vector)):
            temp.append(vector[element]**exponent)
        result.append(temp)
    return result

print(vandermonde4([1,2,3]))

