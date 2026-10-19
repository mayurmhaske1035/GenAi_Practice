from pydantic import BaseModel , EmailStr ,Field
from typing import Optional 

# class student(BaseModel):
#     name: str
#     age: int

# obj = {'name' : "Mayur", 'age' : 22}
# result = student(**obj)
# print(result)

class Student(BaseModel):
    name : str = 'Mayur'
    age : Optional[int] = None
    email : EmailStr
    cgpa :float =  Field(gt=0, lt=10, default=5, description= 'A decimal value representing the cgpa of the student')

new_student = {'age':31, 'email': 'abc@gmail.com'}

obj = Student(**new_student)

new_student2 = {'name':'Om','age':'31', 'email': 'abc@gmail.com', 'cgpa':9}

obj2 = Student(**new_student2)

print(obj2)

print(obj)