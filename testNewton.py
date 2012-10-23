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

    def testNonlinear2DBreaksWithBigDx(self):
        solver = newton.Newton(F.TwoDNonlinear, tol=1.e-12, maxiter=10, dx=1.)
        x0 = N.matrix([[1.],[2.]])
        xRoot = N.matrix([[0.826031357654187],[0.563624162161259]])        
        try:            
            x = solver.solve(x0)
        except Exception as inst:
            message, curRoot = inst.args
            validmessage = 'Newton failed to converge with given number of max iterations'
            self.assertEqual(message,validmessage)            
            assert(N.linalg.norm(curRoot - xRoot) > 0.5*10**(-3))

    def testNonlinear2DExact(self):
        solver = newton.Newton(F.TwoDNonlinear, tol=1.e-12, maxiter=10, dx=1., analyticDF=F.TwoDNonlinearExactJacobian)
        x0 = N.matrix([[1.],[2.]])
        xRoot = N.matrix([[0.826031357654187],[0.563624162161259]])        
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, xRoot)

    def testNonlinear1D(self):
        poly = F.Polynomial([1,2,-10,2])
        solver = newton.Newton(poly, tol=1.e-12, maxiter=15)
        x = solver.solve(-1.)
        self.assertAlmostEqual(x, 0.209718777537)

    def testNonlinear1DExact(self):
        poly = F.Polynomial([1,0,0,2,-5,-6,1])        
        solver = newton.Newton(poly, tol=1.e-12, maxiter=15, dx=1., analyticDF=F.OneDNonlinearExact)
        x = solver.solve(0.0)
        self.assertAlmostEqual(x, 0.149220440931)

    def testConvergenceRadiusLinear(self):
        f = lambda x : 3.1 * x + 31.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=5, searchRadius=10)
        try:    
            x = solver.solve(2.0)
        except Exception as inst:
            message, it, x = inst.args
            validmessage = 'Exceeded search radius, next values: <interation> <current postion>'
            self.assertEqual(message, validmessage)
            assert(it < 5)
            assert(abs(x+10) > 1.5)

    def testSearchRadiusNonlinear2DExact(self):
        solver = newton.Newton(F.TwoDNonlinear, tol=1.e-12, maxiter=10, dx=1., analyticDF=F.TwoDNonlinearExactJacobian, searchRadius=1)
        x0 = N.matrix([[1.2],[2.5]])
        xRoot = N.matrix([[0.826031357654187],[0.563624162161259]])        
        try:    
            x = solver.solve(x0)
        except Exception as inst:
            message, it, x = inst.args
            validmessage = 'Exceeded search radius, next values: <interation> <current postion>'
            self.assertEqual(message, validmessage)
            assert(it < 10)
            assert(N.linalg.norm(x-x0) <= 1)

    def test1DPathological(self):
        """Pathological example to break Newton in infinite loop. Newton should quit out once maxiter is reached so as to not wait forever"""
        poly = F.Polynomial([1,0,-2,2])
        solver = newton.Newton(poly, tol=1.e-12, maxiter=16)
        try:            
            x = solver.solve(0.1)
        except Exception as inst:
            message, extra = inst.args
            validmessage = 'Newton failed to converge with given number of max iterations'
            self.assertEqual(message,validmessage)            
        
if __name__ == "__main__":
    unittest.main()
