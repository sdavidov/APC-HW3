#!/usr/bin/env python

import newton
import functions as F
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=5)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testNonlinear2D(self):
        solver = newton.Newton(F.TwoDNonlinear, tol=1.e-12, maxiter=10)
        x0 = N.matrix([[1.],[2.]])
        xRoot = N.matrix([[0.826031357654187],[0.563624162161259]])        
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, xRoot)

    def testNonlinear1D(self):
        poly = F.Polynomial([1,2,-10,2])
        solver = newton.Newton(poly, tol=1.e-12, maxiter=15)
        x = solver.solve(-1.)
        self.assertAlmostEqual(x, 0.209718777537)

    def test1DPathological(self):
        """Pathological example to break Newton in infinite loop. Newton should quit out once maxiter is reached so as to not wait forever"""
        poly = F.Polynomial([1,0,-2,2])
        solver = newton.Newton(poly, tol=1.e-12, maxiter=16)
        try:            
            x = solver.solve(0.1)
        except Exception as inst:
            message, = inst.args
            validmessage = 'Newton failed to converge with given number of max iterations'
            self.assertEqual(message,validmessage)            
        
if __name__ == "__main__":
    unittest.main()
