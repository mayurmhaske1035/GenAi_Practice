from pydantic import BaseModel

class student(BaseModel):
    name: str
    age: int

obj = {'name' : "Mayur", 'age' : 22}
result = student(**obj)
print(result)
