import pickle
from pprint import pprint

with open("data/graph_event_train.pkl", "rb") as f:
    data = pickle.load(f)

print("type(data):", type(data))
print("len(data):", len(data))
print("sample[0] keys:")
print(data[0].keys())

print("\nfull sample[0]:")
pprint(data[0])