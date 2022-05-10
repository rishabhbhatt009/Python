# name = 'Rishabh'
# print(f'Hello World ! My Name is {name} \n')
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


# String Format --------------------------------------------
# print('{:-<100}'.format(f'Importing Data from : {12345}'))
# print('{:-<50}{:->50}'.format('Import Complete', f'run_time:{1.2334:.3f} secs'))
# s = 'xyz'
# print(f'{s:-<100}')
# print(f"{'xyz':-<100}" )

# pp = '{:<20}{:<20}{:>20}'
# print(pp.format(f'Epoch {1},', f'Step {1}/{10},', f'loss = {0.3424:.4f} '))
# print(pp.format(f'Epoch {1010},', f'Step {10}/{1000},', f'loss = {0.3424:.4f}'))
