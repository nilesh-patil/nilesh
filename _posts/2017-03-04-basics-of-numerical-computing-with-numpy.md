---
title: "Basics of Numerical Computing with Numpy"
date: 2017-03-04T05:10:55-01:00
header :
    overlay_image: "blog/feature/numpy_header.jpeg"
categories: blog
excerpt: 'Working with numpy'
tags: 
    - numpy
    - python
image:
  feature:
---

## Sections

- [Sections](#sections)
  - [Vectors](#vectors)
  - [Calculate basic statistical measures](#calculate-basic-statistical-measures)
  - [Subset a vector](#subset-a-vector)
- [Matrices](#matrices)
  - [Create a matrix](#create-a-matrix)
  - [Matrix operations](#matrix-operations)
  - [Subset a matrix](#subset-a-matrix)
  - [References](#references)


 If you don't know, in short - [Numpy](http://www.numpy.org) is a python library which provides support for fast computations over arrays (vectors, matrices, tensors). Its faster compared to structuring the same computation in base python because operations are vectorized & in general you end up writing code that is pretty close to mathematical notation of your operations instead of writing low level code & dealing with errors & overheads that might creep in during these operations.

##### Import numpy  package for current session

```python
import numpy as np
```

### Vectors

#### Create vectors by generating different sequence of numbers : $\vec{V} = [a, b, c, d, e, . . . . . ]$


```python
# sequence of integers between given bounds

w = np.arange(10,25, step=1)

# 10 random integers between given bounds

x = np.random.randint(low=0,high=10,size=10)

# 10 real numbers drawn from standard normal distribution

y = np.random.randn(10)

# A vector of length 10 with all zeroes

z = np.zeros(10)

# Another convenient way to generate a vetor or even an array of zeroes is as follows:

z = np.zeros_like(y)

# generate sequence of numbers between given bounds & fixed step

s = np.arange(start=15, stop=35, step=2)

print(x)
```

    [8 5 8 2 2 6 1 0 2 5]


#### Single vector operations

- Sum of a sequence : $$Sum = \displaystyle\sum_{i=1}^{n} x_i$$


```python
x.sum()
```
    30


- Adding a constant to each element of vector : $$x_{i_{new}} = \displaystyle x_i+c$$


```python
c = 2

X_new = x+c
print(X_new)
```
    [ 2  2 11  4  5  8  7  6  3  2]

- Multiplying a constant to each element of vector : $$x_{i_{new}} = \displaystyle x_i*c$$


```python
c = 5
X_new = x*c
print(X_new)
```
    [ 0  0 45 10 15 30 25 20  5  0]

- Reverse a vector : $x_{ij} = x_{ji}$

```python
S_new = s[::-1]
print(S_new)
```
    [33 31 29 27 25 23 21 19 17 15]

### Calculate basic statistical measures

- Mean ($$\mu = \frac{1}{N}\displaystyle\sum_{i=1}^{N}{x_i}$$)


```python
x = np.random.randint(low=0,high=1000,size=100)
x.mean(dtype=np.float32)
```

    475.54001

- Standard deviation ( $$\sigma = \sqrt{\displaystyle\sum_{i=1}^{N}{\frac{(x_i - \mu)^2} {N}}}$$ )


```python
x.std(dtype=np.float32)
```
    298.57318

- Variance ( $$\sigma^2 = \displaystyle\sum_{i=1}^{N}{\frac{(x_i - \mu)^2} {N}}$$ )

```python
x.var(dtype=np.float32)
```
    89145.938


### Subset a vector

- Index for maximum & minimum values in a sequence

```python
x.argmax()
```
    37


```python
x.argmin()
```
    3

- Subset using index

```python
# 2nd to 5th element (excluding 5th)
x[2:5]
```
    array([ 31,   1, 561])

## Matrices

### Create a matrix

- Get a matrix of particular shape by providing values


```python
x = np.array([
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]
         ])
x
```
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])

- Transpose of a matrix

```python
y = np.array([
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]
         ]).T
y
```
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]])

- Get a matrix of particular shape

  - Zero matrix of a specific shape

```python
np.zeros(shape=(4,5))
```
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

  - All-ones matrix
```python
np.ones(shape=(4,5))
```
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])

  - All-ones matrix similar to another matrix

```python
np.ones_like(x)
```

    array([[1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1]])

  - Randomly initialized matrix

```python
np.random.rand(4,5)
```
    array([[ 0.09429882,  0.34480325,  0.11695385,  0.96194279,  0.53927071],
           [ 0.78844899,  0.7351646 ,  0.43960103,  0.20815778,  0.50149201],
           [ 0.26338585,  0.89077065,  0.20248855,  0.90770632,  0.91826611],
           [ 0.62807109,  0.48525764,  0.55865624,  0.88327996,  0.51471048]])


### Matrix operations

- Multiply a matrix by constant


```python
5 * x
```
    array([[ 5, 10, 15, 20],
           [ 5, 10, 15, 20],
           [ 5, 10, 15, 20]])

- Multiply a matrix by another


```python
y@x  # New matrix maultiplication operator in python3.5+ !
```
    array([[ 3,  6,  9, 12],
           [ 6, 12, 18, 24],
           [ 9, 18, 27, 36],
           [12, 24, 36, 48]])

```python
np.dot(y,x) # numpy based dot product
```
    array([[ 3,  6,  9, 12],
           [ 6, 12, 18, 24],
           [ 9, 18, 27, 36],
           [12, 24, 36, 48]])

```python
x*y.T # elementwise multiplication or hadamard product of two matrices with same shape
```
    array([[ 1,  4,  9, 16],
           [ 1,  4,  9, 16],
           [ 1,  4,  9, 16]])

### Subset a matrix

- Select all rows and columns (entire matrix)

```python
x[:,:]
```
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])

- Select all rows and specific range of columns

```python
x[:,1:3]
```
    array([[2, 3],
           [2, 3],
           [2, 3]])

---

### References

- [Numpy docs](https://numpy.org/doc/stable/)