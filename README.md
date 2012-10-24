Code to find zeros of arbitrary dimension vector functions using Newton's method, and associated tests of this code

Included files:
newton.py
functions.py
testNewton.py
testFunctions.py

Required library:
numpy

Description of code and purpose:

//----------------------------
newton.py
-Requires numpy and functions.py

Class Newton containing implementation of Newton's method for arbitrary dimension vector functions.
--Required constructor arguments: f, function of which to find the root
--Optional constructor arguments and default values:
    tol=1.e-6, maxiter=20, dx=1.e-6, analyticDF=None, searchRadius = 10

*tol: the tolerance within which a root is considered to be zero.
*maxiter: the maximum number of Newton steps to attempt. Throws an Exception if no root found to tolerance within this number of iterations
*dx: dx used in calculating an approximate Jacobian
*analyticDF: analytic Jacobian function, takes as argument (vector) position at which to evaluate Jacobian
*searchRadius: distance from initial guess within which to search for root, throws and Exception if Newton's method takes a step outside this radius

--method: solve(x0)
Given an initial guess for the root, attempts to find the root. 
//---------------------------
testNewton.py
-Requires numpy, functions.py and newton.py

Suite of tests for all the current functionality of Newton.  Should all pass if Newton class is performing as expected.

//---------------------------
functions.py
-Requires numpy

Set of functions and a class Polynomial.  Class Polynomial contains an implementation of a general polynomial function, function ApproximateJacobian computes an approximate Jacobian. Also contains specialized functions for use in various test routines.

--method:ApproximateJacobian(f,x, dx=1e-6)
-Given a vector valued function, and a vector x, computes an approximate Jacobian, using dx for the 'step' size for the approximate derivative

--Class:Polynomial
-Example usage: to construct the polynomial p(x) = x^2 + 2x + 3, and evaluate p(5):
 p = Polynomial([1, 2, 3])
 p(5)
//---------------------------
testFunctions.py
-Requires numpy and functions.py

Suite of test functions for all the current functionality in functions.py.  Should all pass if function class is performing as expected.
//---------------------------
