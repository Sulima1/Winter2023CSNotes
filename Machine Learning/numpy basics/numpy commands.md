#CP322 #Python
- np.arange(START, STOP, STEP)
	- step defaults to 1
	- np.arange(np.datetime64('2016-12-31'), np.datetime64('2017-02-01')) - prints an array of dates

- There are 5 basic numerical data types are available viz. 
	- booleans (bool) 
	- integers (int) 
	- unsigned integers (uint) 
	- floating-point (float)
	- complex. 
- The array(...) function along with an optional argument `dtype` allows defining the expected data type of the array elements.
- ```
```python
  y = np.int32([1,2,4])
  # OR 
  z = np.arange(start=0,stop=8,step=2, dtype=np.uint8)
```

- To convert the type of array to another data type, we can use the `.astype(DATA_TYPE)` method
- To check the dimension of an array, we can use the `ndim` attribute
- To inspect the number of elements in an array, we can use `size` attribute
- To inspect the size in bytes of each element of the array, we can use the `itemsize` attribute
	- For example, an array of elements of type float64 has itemsize 8 (=64/8)
- To inspect the memory size of an array (in byte), we can use `nbytes` attribute
- To inspect general information of an array, we can use ``info`` method


```python
print("shape:", n_array.shape)
print("dtype:", n_array.dtype)
print("size:", n_array.size)
print("itemsize:", n_array.itemsize)
print("number of bytes of n_array:", n_array.nbytes)
```

- Easily reshape an array with ``reshape(ROWS, COLUMNS)``

```python
two_dim_array1 = np.array([(1,2,3,4), (5,6,7,8)])

two_dim_array1 = two_dim_array1.reshape(4,2)

print(two_dim_array1)

[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```


### Indexing and Slicing

```python
one_dim[1] # returns the second element

one_dim[0:2] # returns first two elements

one_dim[-1] # returns the last element
```

Can use this to interchange values in specific spots of the matrix

```python
two_dim_array[2, 2]=3

print(two_dim_array)

​

# Mutating with slicing

one_dim[0:3]=[4,4,5]

print(one_dim)

[[1 2 3]
 [4 5 6]
 [7 8 3]]
[4 4 5 4 5 6 7 8 9]
```


### Arithmetic Operations
-   np.subtract()
    
-   np.divide()
    
-   np.multiply()