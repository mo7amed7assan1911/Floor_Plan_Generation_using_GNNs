# All Imports
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import distinctipy
from torch_geometric.utils import from_networkx

from tqdm import tqdm
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon


# Define device variable as a global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Functions
def draw_graph(G):
    """ 
        Draws a networkx graph G with the nodes positions and sizes
        according to the room size and color according to the room type.
        
        Input:
            G: networkx graph
        Output:
            None, Just plotting the graph.
    """
    geoms_columns = ['inner', 'living', 'master', 'kitchen', 'bathroom', 'dining', 'child', 'study',
                   'second_room', 'guest', 'balcony', 'storage', 'wall-in',
                    'outer_wall', 'front', 'inner_wall', 'interior',
                   'front_door', 'outer_wall', 'entrance']

    N = len(geoms_columns)
    colors = (np.array(distinctipy.get_colors(N)) * 255).astype(np.uint8)
    room_color = {room_name: colors[i] for i, room_name in enumerate(geoms_columns)}

    #  nodes positions for drawing, note that we invert the y pos
    pos = {node: (G.nodes[node]['actualCentroid_x'], -G.nodes[node]['actualCentroid_y']) for node in G.nodes}
    
    scales = [G.nodes[node]['roomSize'] * 10000 for node in G] 
    colormap = [room_color[G.nodes[node]['roomType_name']]/255 for node in G]
    
    nx.draw(G, pos=pos, node_size=scales, node_color=colormap, with_labels=True, font_size=12)
    
    # Drawing the graph inside a good boundary.
    x_coords  = [pos[node][0] for node in pos]
    y_coords  = [pos[node][1] for node in pos]
    threshold = max(scales) / 100
    
    plt.xlim(min(x_coords) - threshold, max(x_coords) + threshold)
    plt.ylim(min(y_coords) - threshold, max(y_coords) + threshold)



# Graph functions
def load_graphs(path):
    """
        Loads a list of networkx graphs from a pickle file.
        Input:
            path: path to the pickle file
        Output:
            graphs: list of networkx graphs
    """
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

def networkX_to_pytorch(networkx_graphs, features, edge_features=None):
    """
        Converts a list of networkx graphs to a list of pytorch_geometric graphs.
        Input:
            networkx_graphs: list of networkx graphs
            features: list of node features to be included in the pytorch_geometric graph
            edge_features: list of edge features to be included in the pytorch_geometric graph
        Output:
            Graphs_pyTorch: list of pytorch_geometric graphs
    """
    
    Graphs_pyTorch = []
    
    for G in tqdm(networkx_graphs):
        if isinstance(edge_features, list):
            G_new = from_networkx(G, group_node_attrs=features, group_edge_attrs=edge_features)
        else:
            G_new = from_networkx(G, group_node_attrs=features, group_edge_attrs=edge_features)
            
        Graphs_pyTorch.append(G_new)
        
    return Graphs_pyTorch

def minimize_nodes_types(Graphs_pyTorch):
    """
        Minimizing the number of nodes types to 7, and making all labels
        from 0 to 6 only to help one_hotting & considerig all rooms as the same type.
        Input:
            Graphs_pyTorch: list of pytorch_geometric graphs
        Output:
            Graphs_pyTorch: list of pytorch_geometric graphs
    """
    for G in tqdm(Graphs_pyTorch, total=len(Graphs_pyTorch)):
        for j ,value in enumerate(G.x):
            type_ = int(value[0].item())
            
            if type_ in [1, 4, 5, 6, 7, 8]:
                G.x[j][0] = 1
            
            # making all labels from 0 to 6 only to help one_hotting
            elif type_ == 9:
                G.x[j][0] = 4
            elif type_ == 10:
                G.x[j][0] = 5
            elif type_ == 11:
                G.x[j][0] = 6
                
    return Graphs_pyTorch

def normalize_features(Graphs_pyTorch):
    """
        Normalizing the features of the pytorch_geometric graphs.
        Input:
            Graphs_pyTorch: list of not normalized pytorch_geometric graphs
        Output:
            Graphs_pyTorch: list of normalized pytorch_geometric graphs
    """
    for G in tqdm(Graphs_pyTorch, total=len(Graphs_pyTorch)):
        x = G.x # The feature matrix
        for i in [1, 2]:
            mean = torch.mean(x[:, i])
            std  = torch.std(x[:, i])
            
            normalized_column = (x[:, i] - mean) / std
            G.x[:, i] = normalized_column
    
        # One hot encoding for the first column [type of rooms]
        first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
        
        G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)
        
        return Graphs_pyTorch



# # Classes
class Planify_Dataset(Dataset):
    """PyTorch Geometric Dataset for Planify dataset."""
    def __init__(self, data):
        self.Graphs = data
        
    def __len__(self):
        return len(self.Graphs)

    def __getitem__(self, index):
        G = self.Graphs[index].clone().to(device)
        # shuffling nodes inside the same graph
        permutation = torch.randperm(G.num_nodes).to(device)
        
        G.x = G.x[permutation]
        G.edge_index = permutation[G.edge_index]
        G.rec_w = G.rec_w[permutation]
        G.rec_h = G.rec_h[permutation]
        # padded_x = torch.nn.functional.pad(x, pad=(0, 0, 0, 8 - nu_nodes), mode='constant', value=0)
        # padded_y = torch.nn.functional.pad(y, pad=(0, 8 - nu_nodes), mode='constant', value=0)
        
        return G

class ModelTrainer():
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001, num_epochs=250, patience=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Optimization setup
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.950)
        
        # Variables for tracking progress
        self.best_val_loss = float('inf')
        self.best_model = None
        self.counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def train(self):
        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # Forward pass
            logits = self.model(data)
            targets = torch.cat((data.rec_w.unsqueeze(1), data.rec_h.unsqueeze(1)), dim=1)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Updating parameters
            self.optimizer.step()
            # Monitoring
            running_loss += loss.item()
            
        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)
        
        return train_loss
    
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                out = self.model(data)
                targets = torch.cat((data.rec_w.unsqueeze(1), data.rec_h.unsqueeze(1)), dim=1)
                loss = self.criterion(out, targets)
                running_loss += loss.item()

        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        
        return val_loss
    
    def train_step(self):
        for epoch in range(self.num_epochs):
            # Training loop
            train_loss = self.train()
            self.train_losses.append(train_loss)

            # Evaluation loop
            print('Validating ...')
            val_loss = self.evaluate()
            self.val_losses.append(val_loss)

            # Printing and monitoring
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                self.save_checkpoint(self.best_model, self.optimizer, epoch)
                self.counter = 0
            else:
                print('Model not saved!')
                self.counter += 1
                if self.counter >= self.patience:
                    print(f'Validation loss did not improve for {self.patience} epochs. Stopping early.')
                    break
                if self.counter in [3, 5, 7]:
                    self.scheduler.step()
                    print('Learning rate decreased!')
                    
    def save_checkpoint(self, model, optimizer, epoch):
        """Saves the model checkpoint to disk."""
        
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        checkpoint_path = f'./checkpoints/epoch_{epoch}.pth'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)

class GraphBoxes():
    def __init__(self, graph, prediction=None):
        self.graph       = graph
        self.prediction  = prediction
        
    def get_room_data(self, room_index):
        """
        Inputs: 
            room_index: index of the room in the graph
            
        Outputs: 
            centroid, w, h of that room.
        """
        Graph_data = list(self.graph.nodes(data=True))[room_index][1]
        w = Graph_data['rec_w']
        h = Graph_data['rec_h']
        centroid = (Graph_data['actualCentroid_x'], Graph_data['actualCentroid_y'])
        
        if isinstance(self.prediction, np.ndarray): # A  real array of predictions
            w_pre, h_pre = self.get_predictions(room_index)
            
        else:
            w_pre, h_pre = None, None
            
        data = {
            'centroid': centroid,
            'real_w': w,
            'real_h': h, 
            'predic_w': w_pre,
            'predic_h': h_pre
        }
        return data
    
    def create_box(self, room_data):
        """
        Inputs:
            room_data: a dictionary with centroid, w, h of that room.
            
        Outputs:
            box: a shapely box with the same centroid, w, h of that room.
        """
        centroid = room_data['centroid']
        if isinstance(self.prediction, np.ndarray): # A  real array of predictions
            half_w   = room_data['predic_w'] / 2
            half_h   = room_data['predic_h'] / 2
        
        else:
            half_w   = room_data['real_w'] / 2
            half_h   = room_data['real_h'] / 2

        bottom_left  = Point(centroid[0] - half_w, centroid[1] - half_h)
        bottom_right = Point(centroid[0] + half_w, centroid[1] - half_h)
        top_right    = Point(centroid[0] + half_w, centroid[1] + half_h)
        top_left     = Point(centroid[0] - half_w, centroid[1] + half_h)
        
        box = Polygon([bottom_left, bottom_right, top_right, top_left])
        return box

    def get_multipoly(self):
        """
        Outputs:
            multi_poly: a shapely multipolygon of all the rooms in the floor plan or graph.
        """
        num_of_rooms = self.graph.number_of_nodes()
        polygons = []
        for index in range(num_of_rooms):
            room_data = self.get_room_data(index)
            box = self.create_box(room_data)
            
            polygons.append(box)

        multi_poly = MultiPolygon(polygons)
        return multi_poly
    
    def get_predictions(self, room_index):
        """
        Inputs: 
            room_index: index of the room in the graph
        outputs: 
            w_predicted: predicted width for that room
            h_predicted: predicted height for that room
        """
        w_predicted = self.prediction[:, 0]
        h_predicted = self.prediction[:, 1]
        
        return w_predicted[room_index], h_predicted[room_index]