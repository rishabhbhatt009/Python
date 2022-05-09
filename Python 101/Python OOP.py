###############################################################################################
# 4 Principles of Object Oriented Programming
#   1. Inheritance
#   2. Polymorphism
#   3. Encapsulation
#   4. Abstraction

# Summary :
#   - Class Attributes
#   - Instance Attributes
#   - Instance Methods
#   - Dunder (Double underscore) Methods :
#       1. __str__  : called when the obj is converted to string
#       2. __ep__   : called when obj1 == obj2
#   - Static Method
#       - Used with as @staticmethod decorator
#   - Decorators
#       1. @staticmethod
#   -

###############################################################################################
'''
Part-1 Class, Instances and Function

# Class - blue print of an object/instance
class SDE:
    # Class Attributes
    Company = 'Google'

    def __init__(self, name, age):
        # Instance Attributes
        self.name = name
        self.age = age

    # Instance Methods
    def code(self, language: str) -> None:
        print(f'{self.name} is writing code in {language}...')

    def information(self):
        return f'Name = {self.name}, Age = {self.age}'

    # Dunder Methods
    def __str__(self):
        return f'Name = {self.name}, Age = {self.age}'

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    @staticmethod
    def calc_salary(age):
        return 5000 if age < 25 else 7000 if age < 30 else 9000


# Instance
obj1 = SDE('Rishabh', 25)
obj2 = SDE('Rohan', 27)

print(obj1.name, obj1.age, obj1.Company)
print(obj2.name, obj2.age, obj2.Company)
print(obj1.calc_salary(25))

print(SDE.Company)
print(SDE.calc_salary(25))

print(obj1)
obj1.code('Python')
obj2.code('C++')
'''
###############################################################################################
print('Hi')
print('Bye')