"""
boostrtrees.pyx

Cython wrapper of Boost C++ geometry rtree
"""

import cython

import numpy as np
cimport numpy as np

# import both numpy and the Cython declarations for numpy
import numpy as np

from libcpp.memory cimport unique_ptr

from libcpp.vector cimport vector
# from libcpp.vector cimport vector

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

cdef extern from "<array>" namespace "std" nogil :
    cdef cppclass three "3":
        pass

    cdef cppclass array[T, U]:
        T& operator[](size_t)
        # this is obviously very very cut down

cdef extern from "RTreePoint2D.hpp" namespace "rtrees":
    cdef cppclass RTreePoint2D:
        RTreePoint2D() except +
        void insertPoint(double, double, long)
        void insertPoints(double* , long, long)
        long size()
        vector[long] knn(double, double, int)
        double minDistance(double, double)
        vector[double] bounds()
        void move(int, int)

cdef extern from "RTreePoint3D.hpp" namespace "rtrees":
    cdef cppclass RTreePoint3D:
        RTreePoint3D() except +
        void insertPoint(double, double, double)
        void insertPoints(double*, long, long)
        void removePoints(double*, long, long)
        long size()
        vector[long] knn(double, double, double, int)
        vector[long] knn_np(double*, int)
        double minDistance(double, double, double)
        vector[double] bounds()
        vector[long] intersection(double*)
        vector[vector[double]] _all_points()
        void move(int, int)

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RTree:
    cdef unique_ptr[RTreePoint2D] thisptr
    def __cinit__(self):
        self.thisptr.reset(new RTreePoint2D())
    def insert_point(self, double x, double y, long value):
        deref(self.thisptr).insertPoint(x, y, value)
    def insert_points(self, np.ndarray[double, ndim=2, mode="c"] points not None):
        cdef long m, n
        m, n = points.shape[0], points.shape[1]
        assert(n == 3, "Input matrix must have 3 columns (x, y, value)")
        deref(self.thisptr).insertPoints(&points[0,0], m, n)
    def bounds(self):
        bnds = deref(self.thisptr).bounds()
        return {'min_x': bnds[0], 'max_x': bnds[2], 'min_y': bnds[1], 'max_y': bnds[3]}
    def size(self):
        return deref(self.thisptr).size()
    def knn(self, double x, double y, int k):
        return deref(self.thisptr).knn(x, y, k)
    def min_distance(self, double x, double y):
        return deref(self.thisptr).minDistance(x, y)

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RTree3D:
    cdef unique_ptr[RTreePoint3D] thisptr

    def __cinit__(self):
        '''Init
        '''
        self.thisptr.reset(new RTreePoint3D())
    
    def insert_point(self, double x, double y, double z):
        deref(self.thisptr).insertPoint(x, y, z)
    
    def insert_points(self, np.ndarray[double, ndim=2, mode="c"] points not None):
        cdef long m, n, i
        m, n = points.shape[0], points.shape[1]
        assert(n == 3, "Input matrix must have 3 columns (x, y, z)")
        deref(self.thisptr).insertPoints(&points[0,0], m, n)
    
    def remove_points(self, np.ndarray[double, ndim=2, mode="c"] points not None):
        cdef long m, n, i
        m, n = points.shape[0], points.shape[1]
        assert(n == 3, "Input matrix must have 3 columns (x, y, z)")
        deref(self.thisptr).removePoints(&points[0,0], m, n)
    
    def bounds(self):
        bnds = deref(self.thisptr).bounds()
        return {'min_x': bnds[0], 'max_x': bnds[3], 'min_y': bnds[1], 'max_y': bnds[4], 'min_z': bnds[2], 'max_z': bnds[5]}
    
    def size(self):
        return deref(self.thisptr).size()

    def _all_points(self):
        return deref(self.thisptr)._all_points()

    def knn(self, double x, double y, double z, int k):
        return deref(self.thisptr).knn(x, y, z, k)
    
    def knn_np(self, np.ndarray[double, ndim=2, mode="c"] coords not None, int k):
        assert(coords.shape[0] == 1 and coords.shape[1] == 3, "Coordinates must be in the form of a NumPy array(1, 3)")
        return deref(self.thisptr).knn_np(&coords[0,0], k)
    
    def intersection(self, np.ndarray[double, ndim=2, mode="c"] coords not None):
        assert(coords.shape[0] == 2 and coords.shape[1] == 3, "Coordinates must be in the form of a NumPy array(2, 3)")
        return deref(self.thisptr).intersection(&coords[0,0])
    
    def min_distance(self, double x, double y, double z):
        return deref(self.thisptr).minDistance(x, y, z)