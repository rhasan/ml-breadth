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

Vector Creation
---------------

By providing the vector shape as a scalar or as a tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Data creation routines in NumPy will generally have a first parameter which is the shape of the object. 
* The shape can either be a single value for a 1-D result or a tuple (n,m,...).

.. code-block:: python

    # NumPy routines which allocate memory and fill arrays with value
    a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

.. code-block:: bash
    :caption: Output

    np.zeros(4) :   a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
    np.zeros(4,) :  a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
    np.random.random_sample(4): a = [0.87179906 0.88970357 0.26155592 0.38375363], a shape = (4,), a data type = float64    

By providing the vector shape as a scalar only (1-D array only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
    a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

.. code-block:: bash
    :caption: Output

    np.arange(4.):     a = [0. 1. 2. 3.], a shape = (4,), a data type = float64
    np.random.rand(4): a = [0.94801867 0.94743382 0.27758285 0.01850384], a shape = (4,), a data type = float64 

Manually specifying the values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # NumPy routines which allocate memory and fill with user specified values
    a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
    a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

.. code-block:: bash
    :caption: Output

    np.array([5,4,3,2]):  a = [5 4 3 2],     a shape = (4,), a data type = int64
    np.array([5.,4,3,2]): a = [5. 4. 3. 2.], a shape = (4,), a data type = float64

Operations on Vectors
---------------------

Indexing
^^^^^^^^
Indexing means referring to an element of an array by its position within the array.

.. code-block:: python

    #vector indexing operations on 1-D vectors
    a = np.arange(10)
    print(a)

    #access an element
    print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

    # access the last element, negative indexes count from the end
    print(f"a[-1] = {a[-1]}")

    #indexs must be within the range of the vector or they will produce and error
    try:
        c = a[10]
    except Exception as e:
        print("The error message you'll see is:")
        print(e)

.. code-block:: 
    :caption: Output
    :force:

    [0 1 2 3 4 5 6 7 8 9]
    a[2].shape: () a[2]  = 2, Accessing an element returns a scalar
    a[-1] = 9
    The error message you'll see is:
    index 10 is out of bounds for axis 0 with size 10

Slicing
^^^^^^^^
* Slicing means getting a subset of elements from an array based on their indices.
* Slicing creates an array of indices using a set of three values (start:stop:step). A subset of values is also valid i.e. strat or stop or step can be missing.

.. code-block:: python

    #vector slicing operations
    a = np.arange(10)
    print(f"a         = {a}")

    #access 5 consecutive elements (start:stop:step)
    c = a[2:7:1];     print("a[2:7:1] = ", c)

    # access 3 elements separated by two 
    c = a[2:7:2];     print("a[2:7:2] = ", c)

    # access all elements index 3 and above
    c = a[3:];        print("a[3:]    = ", c)

    # access all elements below index 3
    c = a[:3];        print("a[:3]    = ", c)

    # access all elements
    c = a[:];         print("a[:]     = ", c)

.. code-block:: 
    :caption: Output

    a         = [0 1 2 3 4 5 6 7 8 9]
    a[2:7:1] =  [2 3 4 5 6]
    a[2:7:2] =  [2 4 6]
    a[3:]    =  [3 4 5 6 7 8 9]
    a[:3]    =  [0 1 2]
    a[:]     =  [0 1 2 3 4 5 6 7 8 9]

Single vector operations
^^^^^^^^^^^^^^^^^^^^^^^^
* There are a number of useful operations that involve operations on a single vector.

.. code-block:: python

    a = np.array([1,2,3,4])
    print(f"a             : {a}")
    # negate elements of a
    b = -a 
    print(f"b = -a        : {b}")

    # sum all elements of a, returns a scalar
    b = np.sum(a) 
    print(f"b = np.sum(a) : {b}")

    b = np.mean(a)
    print(f"b = np.mean(a): {b}")

    b = a**2
    print(f"b = a**2      : {b}")

.. code-block:: 
    :caption: Output

    a             : [1 2 3 4]
    b = -a        : [-1 -2 -3 -4]
    b = np.sum(a) : 10
    b = np.mean(a): 2.5
    b = a**2      : [ 1  4  9 16]

Vector Vector element-wise operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well. 
* These operators work on an element-by-element basis.
* For this to work correctly, the vectors must be of the same size:

.. code-block:: python

    a = np.array([ 1, 2, 3, 4])
    b = np.array([-1,-2, 3, 4])
    print(f"Binary operators work element wise: {a + b}")

.. code-block:: 
    :caption: Output

    Binary operators work element wise: [0 0 6 8]

Scalar Vector operations
^^^^^^^^^^^^^^^^^^^^^^^^^

* Vectors can be 'scaled' by scalar values. 
* A scalar value is a number.

.. code-block:: python

    a = np.array([1, 2, 3, 4])

    # multiply a by a scalar
    b = 5 * a 
    print(f"b = 5 * a : {b}")

.. code-block:: 
    :caption: Output

    b = 5 * a : [ 5 10 15 20]

Vector Vector dot product
-------------------------

* The dot product multiplies the values in two vectors element-wise and then sums the result.
* Vector dot product requires the dimensions of the two vectors to be the same.

.. image:: images/ch2/ch2-dot_notrans.gif
    :align: center

Dot product with for loop
^^^^^^^^^^^^^^^^^^^^^^^^^
**Using a for loop**, the code below shows an implementation of the following equation:

.. math::
     x = \sum_{i=0}^{n-1} a_i b_i

Here `a` and `b` vectors of the same dimension.


.. code-block:: python

    def my_dot(a, b): 
        """
    Compute the dot product of two vectors
    
        Args:
        a (ndarray (n,)):  input vector 
        b (ndarray (n,)):  input vector with same dimension as a
        
        Returns:
        x (scalar): 
        """
        x=0
        for i in range(a.shape[0]):
            x = x + a[i] * b[i]
        return x

.. code-block:: python

    # test 1-D
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, 4, 3, 2])
    print(f"my_dot(a, b) = {my_dot(a, b)}")

.. code-block:: 
    :caption: Output

    my_dot(a, b) = 24

Vectorized dot product with with :code:`np.dot`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`np.dot` is an optimized dot product where the operations are performed parallelly with speciliad hardwar for the operations.

.. code-block:: python

    # test 1-D
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, 4, 3, 2])
    c = np.dot(a, b)
    print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
    c = np.dot(b, a)
    print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

.. code-block:: 
    :caption: Output

    NumPy 1-D np.dot(a, b) = 24, np.dot(a, b).shape = () 
    NumPy 1-D np.dot(b, a) = 24, np.dot(a, b).shape = () 

:code:`np.dot` vs for loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Vectorization provides significant speed ups. 
* This is because NumPy makes better use of available data parallelism in the underlying hardware. 
* GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. 
* This is critical in Machine Learning where the data sets are often very large.

.. code-block:: python

    np.random.seed(1)
    a = np.random.rand(10000000)  # very large arrays
    b = np.random.rand(10000000)

    tic = time.time()  # capture start time
    c = np.dot(a, b)
    toc = time.time()  # capture end time

    print(f"np.dot(a, b) =  {c:.4f}")
    print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

    tic = time.time()  # capture start time
    c = my_dot(a,b)
    toc = time.time()  # capture end time

    print(f"my_dot(a, b) =  {c:.4f}")
    print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

    del(a);del(b)  #remove these big arrays from memory

.. code-block:: 
    :caption: Output

    np.dot(a, b) =  2501072.5817
    Vectorized version duration: 194.0281 ms 
    my_dot(a, b) =  2501072.5817
    loop version duration: 10901.2289 ms 


Why Vector Vector operations are important
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Vector Vector operations will appear frequently in machine learning. Here is why:

* Going forward, examples will be stored in an array, `X_train` of dimension (m,n). This is a 2 Dimensional array or matrix (see next section on matrices).
* `w` will be a 1-dimensional vector of shape (n,).
* we will perform operations by looping through the examples, extracting each example to work on individually by indexing X. For example:`X[i]`
* `X[i]` returns a value of shape (n,), a 1-dimensional vector. Consequently, operations involving `X[i]` are often vector-vector.  

That is a somewhat lengthy explanation, but aligning and understanding the shapes of your operands is important when performing vector operations. A common example is as below:

.. code-block:: python

    # show common example
    X = np.array([[1],[2],[3],[4]])
    w = np.array([2])
    c = np.dot(X[1], w)

    print(f"X[1] has shape {X[1].shape}")
    print(f"w has shape {w.shape}")
    print(f"c has shape {c.shape}")

.. code-block:: 
    :caption: Output

    X[1] has shape (1,)
    w has shape (1,)
    c has shape ()

Matrices
--------
* Matrices, are two dimensional arrays. The elements of a matrix are all of the same type. 
* In notation, matrices are denoted with capitol, bold letter such as :math:`\mathbf{X}`. 
* `m` is often the number of rows and `n` the number of columns. 
* The elements of a matrix can be referenced with a two dimensional index. 
* In math settings, numbers in the index typically run from 1 to n. In computer science and these labs, indexing will run from 0 to n-1.


.. image:: images/ch2/ch2-matrices.png
    :align: center


Matrices as NumPy Arrays
------------------------

* NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). These were described earlier. 
* Matrices have a two-dimensional (2-D) index [m,n].
* 2-D matrices are used to hold training data. Training data is  *m*  examples by  *n*  features creating an (m,n) array.

Matrix Creation
---------------
* The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples

By providing the matrix shape as a tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    a = np.zeros((1, 5))                                       
    print(f"a shape = {a.shape}, a = {a}")                     

    a = np.zeros((2, 1))                                                                   
    print(f"a shape = {a.shape}, a = {a}") 

    a = np.random.random_sample((1, 1))  
    print(f"a shape = {a.shape}, a = {a}") 

.. code-block:: 
    :caption: Output

    a shape = (1, 5), a = [[0. 0. 0. 0. 0.]]
    a shape = (2, 1), a = [[0.]
    [0.]]
    a shape = (1, 1), a = [[0.44236513]]

Matrix creation by manually specify data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # NumPy routines which allocate memory and fill with user specified values
    a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
    a = np.array([[5],   # One can also
                [4],   # separate values
                [3]]); #into separate rows
    print(f" a shape = {a.shape}, np.array: a = {a}")

.. code-block:: 
    :caption: Output

    a shape = (3, 1), np.array: a = [[5]
    [4]
    [3]]
    a shape = (3, 1), np.array: a = [[5]
    [4]
    [3]]

Operations on Matrices
----------------------

Indexing
^^^^^^^^

.. code-block:: python

    #vector indexing operations on matrices
    a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
    print(f"a.shape: {a.shape}, \na= {a}")

    #access an element
    print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

    #access a row
    print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

.. code-block:: 
    :caption: Output

    a.shape: (3, 2), 
    a= [[0 1]
    [2 3]
    [4 5]]

    a[2,0].shape:   (), a[2,0] = 4,     type(a[2,0]) = <class 'numpy.int64'> Accessing an element returns a scalar

    a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'>

Reshape
^^^^^^^^
* The previous example used reshape to shape the array.
* :code:`a = np.arange(6).reshape(-1, 2)` This line of code first created a 1-D Vector of six elements. It then reshaped that vector into a 2-D array using the reshape command. This could have been written: a = :code:`np.arange(6).reshape(3, 2)` to arrive at the same 3 row, 2 column array. 
* The -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns.

Slicing
^^^^^^^^
* Slicing creates an array of indices using a set of three values (start:stop:step). A subset of values is also valid.

.. code-block:: python

    #vector 2-D slicing operations
    a = np.arange(20).reshape(-1, 10)
    print(f"a = \n{a}")

    #access 5 consecutive elements (start:stop:step)
    print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

    #access 5 consecutive elements (start:stop:step) in two rows
    print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

    # access all elements
    print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

    # access all elements in one row (very common usage)
    print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
    # same as
    print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

.. code-block:: 
    :caption: Output

    a = 
    [[ 0  1  2  3  4  5  6  7  8  9]
    [10 11 12 13 14 15 16 17 18 19]]
    a[0, 2:7:1] =  [2 3 4 5 6] ,  a[0, 2:7:1].shape = (5,) a 1-D array
    a[:, 2:7:1] = 
    [[ 2  3  4  5  6]
    [12 13 14 15 16]] ,  a[:, 2:7:1].shape = (2, 5) a 2-D array
    a[:,:] = 
    [[ 0  1  2  3  4  5  6  7  8  9]
    [10 11 12 13 14 15 16 17 18 19]] ,  a[:,:].shape = (2, 10)
    a[1,:] =  [10 11 12 13 14 15 16 17 18 19] ,  a[1,:].shape = (10,) a 1-D array
    a[1]   =  [10 11 12 13 14 15 16 17 18 19] ,  a[1].shape   = (10,) a 1-D array