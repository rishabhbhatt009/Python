name = 'Rishabh'
print(f'Hello World ! My Name is {name} \n')

# ----------------------------------------------------------

# Data Structures ------------------------------------------
# lst1 = list(name)
# lst2 = list(range(len(name)))
#
# assert name[len(name)-1:-len(name)-1:-1] == name[::-1]
#
# print(name+name)

# dict_ = dict(zip(lst2, lst1))
#
# print(dict_)
# print(dict_.items())
# print(list(enumerate(lst1)))


# List Comprehension ---------------------------------------
# nums = list(range(0, 11))

# lst1 = [i * i for i in nums]
# lst2 = list(map(lambda x: x * x, nums))
# print(lst1, lst2, sep='\n')
#
# lst3 = [i for i in nums if i % 2 == 0]
# lst4 = list(filter(lambda x: x % 2 == 0, nums))
# print(lst3, lst4, sep='\n')


# Spiral Question -------------------------------------
matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

while (matrix != []):
    print(matrix[1:][1:])
    # matrix =
