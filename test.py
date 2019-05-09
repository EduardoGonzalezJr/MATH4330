
"""-----------------------------------------------"""
"""----------------BASIC FUNCTIONS----------------"""
"""-----------------------------------------------"""  
def transpose(A):
    result=[]
    for i in range(len(A)):
        temp=[]
        for element in range(len(A[0])):
            temp.append(A[element][i])
        result.append(temp)
    return result

def conjucate(z):
    result = z.real
    result = result - (z.imag)*1j
    return result

def conjucateTranspose(A):
    result = transpose(A)
    for i in range(len(A)):
        for element in range(len(A[0])):
            result[i][element] = conjucate(result[i][element])
    return result

def printMatrix(A): 
    print('\n'.join([''.join(['{:13.5g}'.format(item) for item in row]) 
      for row in A]))

def scalarVector(scalar, x):
    result=[]
    for element in range(len(x)):
        result.append(scalar*x[element])
    return result

def scalarMatrix(scalar, A): 
    for element in range(len(A)):
        A[element] = scalarVector(scalar, A[element])
    return A

def dotProduct(x,y):
    sum=0
    for element in range(len(x)):
        sum = sum + (x[element] * y[element])
    return sum

def twoNorm(x):
    result=0
    for element in range(len(x)):
        result = result + x[element]**2
    result = result**(1/2)
    return result

def vectorSubtraction(x,y):
    result=x
    for element in range(len(x)):
        result[element] = x[element] - y[element]
    return result

def matrixVectorMult(A, x):
    result = [0]*len(x)
    for i in range(len(A)):
        result[i]=0
        for j in range(len(A[0])):
            result[i] = result[i] + (A[i][j] * x[j])
    return result

"""-----------------------------------------------"""
"""-------------LEAST SQUARE FUNCTIONS------------"""
"""-----------------------------------------------"""
def deg4Vandermonde(x): 
    result=[]
    for exp in range(5):
        temp=[]
        for element in range(len(x)):
            temp.append(x[element]**exp)
        result.append(temp)
    return result 

def gsMod(A):
    length = len(A)
    R = [0]*length
    Q = [0]*length
    V = [0]*length
    
    for i in range(length):
        R[i] = [0]*length
        Q[i] = [0]*length
        V[i] = [0]*length
    for j in range(len(A)):
        V[j] = A[j]
    for j in range(len(V)):
        R[j][j] = twoNorm(V[j])
        Q[j]= scalarVector( (1/(R[j][j])) ,V[j] )
        
        for k in range(j+1,len(V)):
            R[j][k] = dotProduct(Q[j] , V[k])
            x=scalarVector(R[j][k],Q[j])
            V[k] = vectorSubtraction(V[k] , x )
    print("Q (actual):")
    printMatrix(Q)
    print("R (actual):")
    printMatrix(R) 
    return Q,R
    
def backsub(matrix, vector):
    result = vector
    for iterator in range(len(matrix[0])):
        alpha = (len(matrix[0]) - 1)
        sum = 0
        for k in range(((alpha - iterator) + 1), (len(matrix) - 1)):
            sum += (matrix[k][alpha - iterator] * result[k])
        result[alpha - iterator] = (vector[alpha - iterator] - sum) * (1 / matrix[alpha - iterator][alpha - iterator])
    return result

def d4interpolant(B):
    for i in range(len(B),5):
        B.append(0);
    print((B[4]),"x**4 +",B[3],"x**3+",B[2],"x**2+",B[1],"x+",B[0])
        
"""------------------------------------------------"""
"""-----------------FUNCTION CALLS-----------------"""
"""---------------START OF PROGRAM-----------------"""   
x = [1,2,3,4,5]
A = deg4Vandermonde(x)  #STEP 1
y = [6,7,8,9,10]
print("Inputs: \n\tx: ",x,"\n\ty: ",y)
print("Vandermonde matrix (fourth degree):")
printMatrix(A)

Q,R = gsMod(A)          #STEP 2

print("Inverse of Q:")
inverseQ = transpose(Q) #STEP 3
printMatrix(inverseQ)


b = matrixVectorMult(inverseQ,y)
B = backsub(R,b)        #STEP 4

print("\nBacksubstitution solution:\n",B)
print("\nInterpolating polynomial (4th degree): ")
d4interpolant(B)        #STEP 5

""" END OF PROGRAM """



