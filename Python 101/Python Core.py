###################################################################################################

### Data Structures

# 1. List
#   - ordered, mutable and stores duplicate elements
#   - To create a new copy of the list -> lst.copy(), list(lst), lst[:]
#   Methods
#   - add elements      : append, +, extend, insert
#   - remove elements   : pop, remove, clear
#   - order elements    : reverse, sort, sorted
#   - copy              : lst.copy, lst[:], list(lst)
#   - count elements    : count(element)
#   - first index       : index(element)

# 2. Tuple
#   - ordered, immutable and stores duplicate elements
#   - packing and unpacking
#   - more optimal for larger data [*]
#   Methods
#   - add elements      : immutable (create new tuple with +)
#   - remove elements   : immutable
#   - copy              : tup.copy, tup[:], tuple(lst)
#   - count elements    : count(element)
#   - first index       : index(element)

# 3. Dictionaries
#   - unordered, mutable, key-value pairs
#   - keys are hashed
#   - list are unhashable therefore can not be used as keys
#   Methods
#   - get keys          : dict.keys()
#   - get values        : dict.values()
#   - get key,values    : dict.items()
#   - add elements      : new key val assignment, update
#   - remove elements   : del, pop, popitem
#   - copy              : dict.copy, tuple(lst)

# 4. Sets
#   - unordered, mutable, no duplicates
#   - frozenset
#   Methods
#   - add elements      : add
#   - remove elements   : remove, discard, clear, pop
#   - union             : combine sets
#   - intersection      : intersection of 2 sets
#   - difference        : difference b/w 2 sets
#   - symetric diff     : all elements except those in both
#   - check for subset  : issubset, issuperset, isdisjoint [*]
#   - update set        : update, intersection_update, defference_update, symmetric_diff_update

# 5. Strings
#   - ordered, immutable
#   Methods
#   - strip, rstrip, lstrip
#   - upper, lower
#   - startswith, endswith
#   - find, count
#   - replace
#   - split, join [*]
#   - format

###################################################################################################