from typing import TypedDict

class person(TypedDict):
    name= str
    age = int
    address = str

obj : person = {
    'name'    : "Mayur",
    'age'     : 22,
    'address' :  "Pune"
    }
print(obj)
    