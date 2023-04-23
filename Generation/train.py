from imports import *
url = r"D:\Grad\Planify_Dataset\Graph\graphs\boundaries.pkl"


with open(url, 'rb') as f:
    boundaries = pickle.load(f)
    
G = boundaries[1911]


print(G)
# draw_graph(G)

# edge = int(len(Graphs_pyTorch) * 0.9)
# batch_size = 64

# train_dataset = Graphs_pyTorch[:edge]
# train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = Planify_Dataset(Graphs_pyTorch[edge:])
# val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)