#!/usr/bin/env python

import newton
import functions as F
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
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
        f = lambda x : x
        solver = newton.Newton(poly, tol=1.e-12, maxiter=15)
        x = solver.solve(-1.)
        self.assertAlmostEqual(x, 0.209718777537)

if __name__ == "__main__":
    unittest.main()
