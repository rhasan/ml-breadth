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
We use a foor loop to implement the dot product using the following equation:

.. math::
     x = \sum_{i=0}^{n-1} a_i b_i

Here `a` and `b` are vectors of the same dimension.


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
* In notation, matrices are denoted with capital, bold letter such as :math:`\mathbf{X}`. 
* `m` is often the number of rows and `n` the number of columns. 
* The elements of a matrix can be referenced with a two dimensional index. 
* In math settings, numbers in the index typically run from 1 to n. In computer science, indexing will run from 0 to n-1.


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

Multiple Variable Linear Regression
===================================
In this section, we will extend our previous ideas and concepts to support multiple features for linear regression, otherwise known as Multiple Variable Linear Regression.

Notation
^^^^^^^^

Here is a summary of some of the notation we will encounter for multiple features.  

.. list-table:: Summary of notations
   :widths: 25 25 25
   :header-rows: 1

   * - Notation
     - Description
     - Python
   * - :math:`a`
     - scalar, non bold
     - 
   * - :math:`\mathbf{a}`
     - vector, bold
     - 
   * - :math:`\mathbf{A}`
     - matrix, bold capital 
     - 
   * - :math:`\mathbf{X}`
     - training example matrix
     - :code:`X_train`
   * - :math:`\mathbf{y}`
     - training example  targets
     - :code:`y_train`
   * - :math:`\mathbf{x}^{(i)}, y^{(i)}`
     - :math:`i_{th}` training Example
     - :code:`X[i], y[i]`
   * - :math:`m`
     - number of training examples
     - :code:`m`
   * - :math:`n`
     - number of features in each example
     - :code:`n`
   * - :math:`\mathbf{w}`
     - parameter: weight
     - :code:`w`
   * - :math:`b`
     - parameter: bias
     - :code:`b`
   * - :math:`f_{\mathbf{w},b}(\mathbf{x}^{(i)})`
     - The result of the model evaluation at :math:`\mathbf{x^{(i)}}` parameterized by :math:`\mathbf{w},b`: :math:`f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b`
     - :code:`f_wb`

Motivating example
^^^^^^^^^^^^^^^^^^
We will use a motivating example of housing price prediction. The training data contains 3 examples. An example has four features: size, bedrooms, floors, and age, as shown in the table below:


.. list-table:: Summary of notations
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Size (sqft)
     - Number of beedrooms
     - Number of floors
     - Age of home
     - Price (1000s dollars)
   * - 2104
     - 5
     - 1
     - 45
     - 460
   * - 1416
     - 3
     - 2
     - 40
     - 232
   * - 852
     - 2
     - 1
     - 35
     - 178

We create  `X_train` and `y_train` variables where we load the training inputs and target values.

.. code-block:: python

    import copy, math
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

Similar to the table above, examples are stored in a NumPy matrix `X_train`. Each row of the matrix represents one example. When you have :math:`m` training examples ( :math:`m` is three in our example), and there are :math:`n` features (four in our example), :math:`\mathbf{X}` is a matrix with dimensions (:math:`m`, :math:`n`) (m rows, n columns).

.. math::

    \mathbf{X} = 
    \begin{pmatrix}
    x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\ 
    x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
    \cdots \\
    x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1} 
    \end{pmatrix}

notation:

* :math:`\mathbf{x}^{(i)}` is vector containing example i. :math:`\mathbf{x}^{(i)} = (x^{(i)}_0, x^{(i)}_1, \cdots,x^{(i)}_{n-1})`
* :math:`x^{(i)}_j` is the element j in the example i. The superscript in parenthesis indicates the example number while the subscript represents an element.


The following block of Python code displays the loaded data:

.. code-block:: python

    # data is stored in numpy array/matrix
    print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
    print(X_train)
    print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
    print(y_train)


.. code-block:: 
    :caption: Output

    X Shape: (3, 4), X Type:<class 'numpy.ndarray'>)
    [[2104    5    1   45]
    [1416    3    2   40]
    [ 852    2    1   35]]
    y Shape: (3,), y Type:<class 'numpy.ndarray'>)
    [460 232 178]

Parameter vector w, b
^^^^^^^^^^^^^^^^^^^^^

* :math:`\mathbf{w}` is a vector with :math:`n` elements.

  * Each element contains the parameter associated with one feature.

  * in our dataset, n is 4.

  * notionally, we draw this as a column vector

.. math::

    \mathbf{w} = \begin{pmatrix}
    w_0 \\ 
    w_1 \\
    \cdots\\
    w_{n-1}
    \end{pmatrix}

* :math:`b` is a scalar parameter.  

For demonstration, :math:`\mathbf{w}` and :math:`b` will be loaded with some initial selected values that are near the optimal. :math:`\mathbf{w}` is a 1-D NumPy vector.

.. code-block:: python

    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

.. code-block:: 
    :caption: Output

    w_init shape: (4,), b_init type: <class 'float'>

Model Prediction With Multiple Variables
----------------------------------------

The model's prediction with multiple variables is given by the linear model:

.. math::

    f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}

or in vector notation:

.. math::

    f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2}

where :math:`\cdot` is a vector `dot product`

To demonstrate the dot product, we will implement prediction using (1) and (2).

Single Prediction element by element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our previous prediction multiplied one feature value by one parameter and added a bias parameter. A direct extension of our previous implementation of prediction to multiple features would be to implement (1) above using loop over each element, performing the multiply with its parameter and then adding the bias parameter at the end.

.. code-block:: python

    def predict_single_loop(x, w, b): 
        """
        single predict using linear regression
        
        Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters    
        b (scalar):  model parameter     
        
        Returns:
        p (scalar):  prediction
        """
        n = x.shape[0]
        p = 0
        for i in range(n):
            p_i = x[i] * w[i]  
            p = p + p_i         
        p = p + b                
        return p

Now let us predict using this method:

.. code-block:: python

    # get a row from our training data
    x_vec = X_train[0,:]
    print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

    # make a prediction
    f_wb = predict_single_loop(x_vec, w_init, b_init)
    print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

.. code-block:: 
    :caption: Output

    x_vec shape (4,), x_vec value: [2104    5    1   45]
    f_wb shape (), prediction: 459.9999976194083

Note the shape of :code:`x_vec`. It is a 1-D NumPy vector with 4 elements, (4,). The result, :code:`f_wb` is a scalar.

Single Prediction, vector
^^^^^^^^^^^^^^^^^^^^^^^^^^

Noting that equation (1) above can be implemented using the dot product as in (2) above. We can make use of vector operations to speed up predictions.

NumPy `np.dot() <https://numpy.org/doc/stable/reference/generated/numpy.dot.html)>`_ can be used to perform a vector dot product. 

.. code-block:: python

    def predict(x, w, b): 
        """
        single predict using linear regression
        Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters   
        b (scalar):             model parameter 
        
        Returns:
        p (scalar):  prediction
        """
        p = np.dot(x, w) + b     
        return p 

Now let us predict again using this vectorized implementation:

.. code-block:: python

    # get a row from our training data
    x_vec = X_train[0,:]
    print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

    # make a prediction
    f_wb = predict(x_vec,w_init, b_init)
    print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

.. code-block:: 
    :caption: Output

    x_vec shape (4,), x_vec value: [2104    5    1   45]
    f_wb shape (), prediction: 459.99999761940825


The results and shapes are the same as the previous version which used looping. Going forward, :code:`np.dot` will be used for these operations. The prediction is now a single statement. Most routines will implement it directly rather than calling a separate predict routine.


Compute Cost With Multiple Variables
------------------------------------

The equation for the cost function with multiple variables :math:`J(\mathbf{w},b)` is:

.. math::

    J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}

where:

.. math::

    f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4}


In contrast to univariate linear regression, :math:`\mathbf{w}` and :math:`\mathbf{x}^{(i)}` are vectors rather than scalars supporting multiple features.

Below is an implementation of equations (3) and (4). Note that this uses a for loop over all `m` examples is used. Therefore it is only partially vectorized, in the future we will see how to optimize this further.

.. code-block:: python

    def compute_cost(X, y, w, b): 
        """
        compute cost
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        cost (scalar): cost
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):                                
            f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
            cost = cost + (f_wb_i - y[i])**2       #scalar
        cost = cost / (2 * m)                      #scalar    
        return cost

Now let us compute the cost using our pre-chosen optimal parameters.

.. code-block:: python

    # Compute and display cost using our pre-chosen optimal parameters. 
    cost = compute_cost(X_train, y_train, w_init, b_init)
    print(f'Cost at optimal w : {cost}')

.. code-block:: 
    :caption: Output

    Cost at optimal w : 1.5578904880036537e-12

Gradient Descent With Multiple Variables
----------------------------------------

The gradient descent algorithm for multiple variables:

.. math::

    \begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
    & w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
    &b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
    \end{align*}

where, n is the number of features, parameters :math:`w_j`,  :math:`b`, are updated simultaneously and where  

.. math::

    \begin{align}
    \frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
    \frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
    \end{align}

* m is the number of training examples in the data set

    
*  :math:`f_{\mathbf{w},b}(\mathbf{x}^{(i)})` is the model's prediction, while :math:`y^{(i)}` is the target value


Compute Gradient with Multiple Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An implementation for calculating the equations (6) and (7) is below. There are many ways to implement this. In this version, there is an

* outer loop over all m examples. 

  * :math:`\frac{\partial J(\mathbf{w},b)}{\partial b}` for the example can be computed directly and accumulated
  * in a second loop over all n features:
    
    * :math:`\frac{\partial J(\mathbf{w},b)}{\partial w_j}` is computed for each :math:`w_j`.

.. code-block:: python

    def compute_gradient(X, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
        """
        m,n = X.shape           #(number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):                             
            err = (np.dot(X[i], w) + b) - y[i]   
            for j in range(n):                         
                dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
        dj_dw = dj_dw / m                                
        dj_db = dj_db / m                                
            
        return dj_db, dj_dw

Now let us compute gradient using our initialized parameter values:

.. code-block:: python

    #Compute and display gradient 
    tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
    print(f'dj_db at initial w,b: {tmp_dj_db}')
    print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

.. code-block:: 
    :caption: Output

    dj_db at initial w,b: -1.673925169143331e-06
    dj_dw at initial w,b: 
    [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]


Gradient Descent With Multiple Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The routine below implements equation (5) above.

.. code-block:: python

    def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters  
        b_in (scalar)       : initial model parameter
        cost_function       : function to compute cost
        gradient_function   : function to compute the gradient
        alpha (float)       : Learning rate
        num_iters (int)     : number of iterations to run gradient descent
        
        Returns:
        w (ndarray (n,)) : Updated values of parameters 
        b (scalar)       : Updated value of parameter 
        """
        
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(w_in)  #avoid modifying global w within function
        b = b_in
        
        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw               ##None
            b = b - alpha * dj_db               ##None
        
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                J_history.append( cost_function(X, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
            
        return w, b, J_history #return final w,b and J history for graphing


Now let us fit our model using this gradient descent implementation:

.. code-block:: python

    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                        compute_cost, compute_gradient, 
                                                        alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

.. code-block:: 
    :caption: Output

    Iteration    0: Cost  2529.46   
    Iteration  100: Cost   695.99   
    Iteration  200: Cost   694.92   
    Iteration  300: Cost   693.86   
    Iteration  400: Cost   692.81   
    Iteration  500: Cost   691.77   
    Iteration  600: Cost   690.73   
    Iteration  700: Cost   689.71   
    Iteration  800: Cost   688.70   
    Iteration  900: Cost   687.69   
    b,w found by gradient descent: -0.00,[ 0.2   0.   -0.01 -0.07] 
    prediction: 426.19, target value: 460
    prediction: 286.17, target value: 232
    prediction: 171.47, target value: 178