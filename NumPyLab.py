import numpy as np
from functools import reduce
import sys

# 1. Import Numpy and display the version.
print(f'№1 version numpy: {np.__version__}')

# 2. Create a 1D array and a boolean array.
num = np.array([i for i in range(1, 11, 1)])
print(f'№2 {num}')
boolean = np.logical_and(num % 3 != 0, num % 2 != 0)
print(f'№2 {boolean}')

# 3. Extract all odd numbers from the 1D array and
# replace them with -1 without affecting the NumPyLab.py:10original array.
change_num = np.where((num % 2 != 0), -1, num)
print(f'№3 {change_num}')

# 4. Reshape the array into a 2D array.
change_num_2d = change_num.reshape((2, -1))
print(f'№4 {change_num_2d}')

# 5. Stack two array vertically and horizontally.
num_2d = np.random.randint(10, size=(2, 5))
col_sum = np.sum(num_2d, axis=0)
row_sum = np.sum(num_2d, axis=1)
print(f'№5 {col_sum}')
print(f'№5 {row_sum}')

# 6. Generate an array with a custom sequence.
seq_arr = np.r_[np.tile(num, 2)]
print(f'№6 {seq_arr}')
seq_arr = np.arange(2, 22, 2)

# 7. Get common items between the 3 arrays.
bet_it = reduce(np.intersect1d, (num, change_num_2d, seq_arr))
print(f'№7 {bet_it}')

# 8. Identify the position of similar elements between any two arrays.
pos_sim = np.intersect1d(num, seq_arr, return_indices=True)
print(f'№8 {pos_sim}')

# 9. Remove items from one tray that exist in the other two.
rem_unique = np.setdiff1d(num, seq_arr)
print(f'№9 {rem_unique}')

# 10. Extract all numbers between 5 and 50 from the array.
condition = (np.logical_and(5 < seq_arr, seq_arr < 50))
extract_arr = np.extract(condition, seq_arr)
print(f'№10 {extract_arr}')

# 11. Convert scalar function max to work on two arrays.
con_scalar = np.maximum(np.arange(1, 10), np.arange(10, 1, -1))
print(f'№11 {con_scalar}')

# 12. Swap two rows and two columns in a 2D array.
q_12 = np.array([[0, 1, 2], [3, 4, 5]])
q_12[:, [2, 0]] = q_12[:, [0, 2]]
q_12[[1, 0]] = q_12[[0, 1]]
print(f'№12 {q_12}')

# 13. Reverse rows and columns in the 2D array.
q_13 = np.array([[0, 1, 2], [3, 4, 5]])
q_13 = q_13[::-1]
q_13 = q_13[:, ::-1]
print(f'№13 {q_13}')

# 14. Create a new array containing random floats between 8 and 25 and print only 3 decimal places.
rand_arr = np.round((np.random.randint(low=8, high=25, size=(2, 5)) + np.random.random((2, 5))), 3)
print(f'№14 {rand_arr}')

# 15. Get the second largest value of an array when grouped by another array
sec_lar = np.random.randint(low=10, high=70, size=(1, 10))
q_15 = np.unique(np.sort(sec_lar))[-2]
print(f'№15 {q_15}')

# 16. Suppress the scientific notation in the float array.
sci_ran = np.random.random([1, 5])/1e6
np.set_printoptions(suppress=True)
print(f'№16 {sci_ran}')

# 17. Print limited number of items from the array.
q_17 = np.arange(2, 17)
np.set_printoptions(threshold=1)
print(f'№17 {q_17}')

# 18. Print all items in the array without truncating.
np.set_printoptions(threshold=sys.maxsize)
print(f'№18 {q_17}')

# 19. Import a dataset confining both text and numbers and keep text intact.
data = np.genfromtxt('H.txt', delimiter=' ', dtype=object)
print(f'№19 {data}')

# 20. Extract a column from 1D tuple.
data = np.genfromtxt('H.txt', delimiter=' ', dtype=object)
q_20 = np.array([row[-1] for row in data])
print(f'№20 {tuple(q_20)}')

# 21. Convert 1D tuple to 2D array.
data = np.genfromtxt('H.txt', delimiter=' ', dtype=object)
q_20 = np.array([row[-1] for row in data])
q_21 = q_20.reshape((2, -1))
print(f'№21 {tuple(q_21)}')

# 22. Compute the mean, median and the standard deviation of the array.
print(f'№22 {np.mean(np.random.randint(low=8, high=99, size=(1,10)))}')
print(f'№22 {np.median(np.random.randint(low=8, high=99, size=(1,10)))}')
print(f'№22 {np.std(np.random.randint(low=8, high=99, size=(1,10)))}')

# 23. Normalise the array so that the range of values is between 0 and 1.
norm_ran = np.random.rand(10)*10
norm = np.linalg.norm(norm_ran)
norm_arr = norm_ran/norm
print(f'№23 {norm_arr}')

# 24. Compute the softmax score and percentile scores.
data = np.genfromtxt('heh.txt', delimiter=' ')
q_20 = np.array([row[0] for row in data])


def softmax(x):
    e_x = np.exp(x) / np.sum(np.exp(x))
    return e_x


print(f'№24 {softmax(q_20)}')
data = np.genfromtxt('heh.txt', delimiter=' ')
q_20 = np.array([row[0] for row in data])
print(f'№24 {np.percentile(q_20, q=[5, 50])}')

# 25. Find and drop missing values and null values and insert random values in an array.
mis_val = np.array([0, 1, 2, np.nan, 4, np.nan])
print(f'№26 {mis_val[~np.isnan(mis_val)]}')
mis_val = np.array([0, 1, 2, np.nan, 4, np.nan])
print(f'№26 {np.nan_to_num(mis_val, nan=7)}')

# 26. Count unique values in the array.
data = np.genfromtxt('heh.txt', delimiter=' ')
print(f'№26 {len(np.unique(data))}')

# 27. Convert numeric to text array.
data = np.genfromtxt('convert to text.txt', delimiter=' ', dtype=int)
data = np.char.array(data)
print(f'№27 {data}')

# 28. Find the correlation between two columns of an array.
x_1 = np.random.randint(low=10, high=100, size=(1, 20))
y_1 = np.random.randint(low=10, high=100, size=(1, 20))
print(f'№28 {np.corrcoef(x_1, y_1)}')

# 29. Create a new column from the existing one of a Numpy array.
data = np.genfromtxt('convert to text.txt', delimiter=' ', dtype=int)
new_col = np.random.randint(low=10, high=30, size=(4, 1))
print(f'№29 {np.append(data, new_col, axis=1)}')

# 30. Get the positions of top n values from the array.
arr = np.array([1, 3, 2, 4])
print(f'№30 {(-arr).argsort()[:-2]}')

# 31. Sort a 2D array by the column.
ran_arr = np.random.randint(low=1, high=100, size=(3, 3))
sort_ran_arr = ran_arr[np.argsort(ran_arr[:, -1])]
print(f'№31 {sort_ran_arr}')
