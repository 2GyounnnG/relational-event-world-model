from data.dataset import GraphEventDataset
from data.collate import graph_event_collate_fn
from torch.utils.data import DataLoader

ds = GraphEventDataset("data/graph_event_train.pkl")
print("dataset size:", len(ds))
print("sample keys:", ds[0].keys())

loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    collate_fn=graph_event_collate_fn,
)

batch = next(iter(loader))

for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, type(v), len(v))