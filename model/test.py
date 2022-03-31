import pickle
import json

my_dict = {
        "postcode": 1000,
        "kitchen_type": "Not installed",
        "bedroom": 3 ,
        "swimming_pool": "Yes",
        "surface_plot": 200,
        "living_area": 100,
        "property_type": "Apartment".upper()}

#with open('data.json', 'w') as fp:
    #json.dump(my_dict, fp)

with open('data.json', 'r') as fp:
    data = json.load(fp)
print(data)

print(data.keys())