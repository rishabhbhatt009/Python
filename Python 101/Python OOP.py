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

###############################################################################################

# Part-1 Class, Instances and Function
#
# # Class - blue print of an object/instance
# class SDE:
#     # Class Attributes
#     Company = 'Google'
#
#     def __init__(self, name, age):
#         # Instance Attributes
#         self.name = name
#         self.age = age
#
#     # Instance Methods
#     def code(self, language: str) -> None:
#         print(f'{self.name} is writing code in {language}...')
#
#     def information(self):
#         return f'Name = {self.name}, Age = {self.age}'
#
#     # Dunder Methods
#     def __str__(self):
#         return f'Name = {self.name}, Age = {self.age}'
#
#     def __eq__(self, other):
#         return self.name == other.name and self.age == other.age
#
#     @staticmethod
#     def calc_salary(age):
#         return 5000 if age < 25 else 7000 if age < 30 else 9000
#
#
# # Instance
# obj1 = SDE('Rishabh', 25)
# obj2 = SDE('Rohan', 27)
#
# print(obj1.name, obj1.age, obj1.Company)
# print(obj2.name, obj2.age, obj2.Company)
# print(obj1.calc_salary(25))
#
# print(SDE.Company)
# print(SDE.calc_salary(25))
#
# print(obj1)
# obj1.code('Python')
# obj2.code('C++')

###############################################################################################

# Summary
#   1. Inheritance
#       - Child class inherits charecteristics from parent class
#       - Parent and Child Classes
#       - Inherit   : All attributes and instance methods from the base class
#       - Extend    : Child class can have its own attributes and methods (different name)
#       - Override  : Replacing functions of the parent class with its own functions (same name)
#   - super()

#   2. Polymorphism
#       - Objects behave differently in different situation

###############################################################################################

# Part-2 Inheritance, Polymorphism
#
# # Parent Class
# class Employee:
#     def __init__(self, name, age, salary):
#         self.name = name
#         self.age = age
#         self.salary = salary
#
#     def __str__(self):
#         return f'Name = {self.name}, Age = {self.age}, Salary = {self.salary}'
#
#     def work(self):
#         print(f'{self.name} is working...')
#
# # Child Classes
# class SDE(Employee):
#     def __init__(self, name, age, salary, level):
#         super().__init__(name, age, salary)
#         self.level = level
#
#     def __str__(self):
#         return f'Name = {self.name}, Age = {self.age}, Salary = {self.salary}, Level = {self.level}'
#
#     def work(self):
#         print(f'{self.name} is coding...')
#
# class Designer(Employee):
#     def work(self):
#         print(f'{self.name} is drawing...')
#
#
# # obj1 = SDE('Rishabh', 25, '5000', 'SDE4')
# # obj2 = Designer('Shreya', 30, '6000')
# # print(obj1)
# # print(obj2)
# # obj1.work()
# # obj2.work()
#
# company1 = [SDE('Rishabh', 25, '5000', 'SDE4'),
#             SDE('Rohan', 25, '5000', 'SDE2'),
#             Designer('Shreya', 30, '6000')]
#
# for emp in company1:
#     emp.work()

###############################################################################################

# Summary

#   3. Encapsulation
#       - Hiding of data implementation
#       - instance variables are kept private
#       - accesser method are used to access or change these variables
#       - which can be used to check value, enforce constraints
#   4. Abstraction
#       - Each object should only expose high level mechanism of using it
#       - Hide internal implementation details

#       - _var              : protected attribute (accessible from outside)
#       - __var             : private attribute (not accessible from outside)
#       - accesser method   : getter / setter methods

###############################################################################################

# class SDE:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#         self._salary = None
#         self._bugs_fixed = 0
#
#     def code(self):
#         self._bugs_fixed += 1
#
#     # getter
#     def get_salary(self):
#         return self._salary
#
#     # setter
#     def set_salary(self, base_salary):
#         # check value, enforce constraints
#         self._salary = self._calc_salary(base_salary)
#
#     def _calc_salary(self, base_salary):
#         mult = 1 if self._bugs_fixed < 50 else 2 if self._bugs_fixed < 100 else 3
#         return mult*base_salary
#
# obj1 = SDE('Rishabh', 25)
# print(obj1.name, obj1.age)
#
# for i in range(130):
#     obj1.code()
#
# obj1.set_salary(5000)
# print(obj1.get_salary())

###############################################################################################

#   Decorator
#       1. @property        : GETTER - func name same as the variable name
#       2. @<var_name>.setter : SETTER - func name same as the variable name

###############################################################################################
class SDE:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self._salary = None

    # getter
    @property
    def salary(self):
        return self._salary

    # setter
    @salary.setter
    def salary(self, base_salary):
        # check value, enforce constraints
        self._salary = base_salary

obj1 = SDE('Rishabh', 25)
print(obj1.name, obj1.age)

# obj1.set_salary(5000)
# print(obj1.get_salary())

# Pythonic Ways
obj1.salary = 6000
print(obj1.salary)
