.. _ml_vectors:

.. contents::
    :local:
    :depth: 2

Vectorization
=============


Vectors
-------

* Vectors are ordered arrays of numbers. 
* In notation, vectors are denoted with lower case bold letters such as :math:`\mathbf{x}`.
* The number of elements in the array is often referred to as the *dimension* though mathematicians may prefer *rank*. The vector shown has a dimension of :math:`n`. 
* The elements of a vector can be referenced with an index. In math settings, indexes typically run from 1 to n. In computer science, indexing will typically run from 0 to n-1.
* In notation, elements of a vector, when referenced individually will indicate the index in a subscript, for example, the :math:`0^{th}` element, of the vector :math:`\mathbf{x}` is :math:`x_0`. Note, the :math:`x_0` is not bold in this case because it is a scalar value.  

.. image:: images/ch2/ch2-vectors.png
    :align: center

Vectors in NumPy
----------------

* NumPy's array data structure is an indexable, n-dimensional *array* containing elements of the same type (`dtype`).
    - Notice that the term 'dimension' in a NumPy array refers to the number of indexes of the array. In vectors, 'dimension' refers to the number of elements in a vector. A one-dimensional or 1-D array has one index. 
* We will represent vectors as NumPy 1-D arrays. 
* 1-D array, shape (n,): n elements indexed [0] through [n-1]