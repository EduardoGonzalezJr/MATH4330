import basicFunctions as BF

"""------------------------------------------------"""
"""-----------------FUNCTION CALLS-----------------"""
"""---------------START OF PROGRAM-----------------"""
if __name__ == "__main__":
    x = [1,2,3,4,5]
    A = BF.vandermonde4(x) #STEP 1
    y = [6,7,8,9,10]
    print("Inputs: \n\tx: ",x,"\n\ty: ",y)
    print("Vandermonde matrix (fourth degree):")
    BF.printMatrix(A)

    Q,R = BF.gsMod(A) #STEP 2

    print("Inverse of Q:")
    inverseQ = BF.conjugateTranspose(Q) #STEP 3
    BF.printMatrix(inverseQ)


    b = BF.matrixVectorMult(inverseQ,y)
    B = BF.backSubstitute(R, b) #STEP 4

    print(B)
    print("\nBacksubstitution solution:\n",B)
    print("\nInterpolating polynomial (4th degree): ")
    BF.d4interpolation(B) #STEP 5
