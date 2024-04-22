import sys
import json

def experimental_fnc(param1, param2):
    print(type(param1))
    print(type(param2))
    print("Jalo todo bien")
    return param1, param2

if __name__ == "__main__":
    print(len(sys.argv))
    for i in sys.argv:
        print(i, type(i))
    if len(sys.argv) >= 2:  # Check if we have enough arguments
        param1 = json.loads(sys.argv[1])
        print("This is type param1", type(param1))
        param2 = json.loads(sys.argv[2])
        print(type(param2))
        experimental_fnc(param1=param1, param2=param2)
    else:
        print("Usage: python your_script.py param1 param2")
