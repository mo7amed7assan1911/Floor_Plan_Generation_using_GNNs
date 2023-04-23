from imports import *

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        
        # Encoder layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim)
        
        # Type prediction layer
        self.type_fc = nn.Linear(hidden_dim, 1)
        
        # Centroid prediction layer
        self.centroid_fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Type prediction
        type_pred = self.type_fc(x)
        
        # Centroid prediction
        centroid_pred = self.centroid_fc(x)
        
        return type_pred, centroid_pred