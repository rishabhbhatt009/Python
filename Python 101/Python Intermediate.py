###################################################################################################

# - Collections
#       1. Counter      : returns a dict with key = elementm, value = count
#       2. nametuple    : similar to a struct in C
#       3. orderedDict  : remembers the order in which the elements are added, same as dict from 3.7
#       4. defaultdict  : returns default value on "KeyError"
#       5. deque        : double ended queue with helpful functions

# - itertools
#       1. product              : cartesian product
#       2. permutation          :
#       3. combinations         :
#       4. accumulate           : [*]
#       5. groupby              : [*]
#       6. infinite iterators   : count, cycly, repeat

# - lambda function

###################################################################################################

### Collections

# from collections import Counter
# str_ = 'aaaabbbbcccdefffff'
# cnt_ = Counter(str_)
#
# print(cnt_)
# print(cnt_.most_common(2))
# print(list(cnt_.elements()))

# from collections import namedtuple
# node = namedtuple('node', 'data,next,prev')
# curr = node(123,'next_node', 'prev_node')
# print(node)
# print(curr)
# print(curr.data)

# from collections import defaultdict
# dict_ = defaultdict(lambda : 'No Value', {'c':3, 'd':4}) # int returns 0, list return []
# dict_['a'] = 1
# dict_['b'] = 2
#
# print(dict_)
# print(dict_['a'])
# print(dict_['z'])

# from collections import deque
# dq = deque([1,2,3])
# print(dq)
# dq.append(100)
# dq.appendleft(200)
# print(dq)
# print(dq.pop())
# print(dq.popleft())
# dq.rotate(1)


### itertools
from itertools import product
a = [1,2]
b = [3,4]
prod = product(a,b)

print(prod, list(prod))

from itertools import permutations, combinations, combinations_with_replacement
a = [1,2,3]

print(list(permutations(a,2)))
print(list(combinations(a,2)))
print(list(combinations_with_replacement(a,2)))



# from itertools import count, cycle, repeat
# lst = ['A', 'B', 'C', 'D', 'E']
#
# for i in count(10,2):
#     print(i, end=' ')
#     if i > 20:
#         break
# print()
#
# count = 0
# for i in cycle(lst):
#     print(i, end=' ')
#     count += 1
#     if count>10:
#         break
#
# print()
#
#
# for i in repeat(10,20):
#     print(i, end=',')
# print()

###################################################################################################


