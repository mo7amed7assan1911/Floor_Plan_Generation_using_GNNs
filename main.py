from utils import *
from model import *


def get_info():
    """
        Function to return:
            - Boundary: "POLYGON ((105.44810944559121 78 ....
            - front_door: "POLYGON  (105.44810944559121 78 ....
            - room_centroids: [(81, 105), (55, 151), (134, 105)]
            - bathroom_centroids: [(81, 105), (55, 151), (134, 105)]
            - kitchen_centroids: [(81, 105), (55, 151), (134, 105)]
    """
    pass


def preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kithchen_centroids):
    
    Boundary = shapely.wkt.loads(Boundary)
    front_door = shapely.wkt.loads(front_door)
    
    
    # Flip the y axis of all polygons and points
    Boundary = scale(Boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kithchen_centroids = [scale(x) for x in kithchen_centroids]
    
    # Retruning the centroids of the rooms and bathrooms from Point to tuple
    room_centroids = [x.coords[0] for x in room_centroids]
    bathroom_centroids = [x.coords[0] for x in bathroom_centroids]
    kithchen_centroids = [x.coords[0] for x in kithchen_centroids]
    
    living_centroid    = [(Boundary.centroid.x, Boundary.centroid.y)]
    
    user_constraints = {
        'living': living_centroid,
        'room': room_centroids,
        'bathroom': bathroom_centroids,
        # 'kitchen': kithchen_centroids
    }
    
    # Making networkX graphs
    boundary_graph = Handling_dubplicated_nodes(Boundary, front_door)
    G_n = centroids_to_graph(user_constraints, living_to_all=True)
    
    # Converting the networkX graph to pytorch geometric data
    B = from_networkx(boundary_graph, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])
    
    features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y']
    G = from_networkx(G_n, group_edge_attrs=['distance'], group_node_attrs=features)
    
    # To use them later for visualization
    B_not_normalized = B.clone()
    G_not_normalized = G.clone()
    
    # Normalization
    G_x_mean = G.x[:, 1].mean().item()
    G_y_mean = G.x[:, 2].mean().item()

    G_x_std = G.x[:, 1].std().item()
    G_y_std = G.x[:, 2].std().item()

    G.x[:, 1:] = (G.x[:, 1:] - torch.tensor([G_x_mean, G_y_mean])) / torch.tensor([G_x_std, G_y_std])
    
    first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
    G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)
    
    #### Normalization for the boundary graph
    B_x_mean = B.x[:, 1].mean().item()
    B_y_mean = B.x[:, 2].mean().item()

    B_x_std = B.x[:, 1].std().item()
    B_y_std = B.x[:, 2].std().item()
    
    B.x[:, 1:] = (B.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / torch.tensor([B_x_std, B_y_std])
    
    
    
    # Befor passing the data to the model
    G.x = G.x.to(torch.float32)
    G.edge_attr = G.edge_attr.to(torch.float32)
    G.edge_index = G.edge_index.to(torch.int64)

    B.x = B.x.to(G.x.dtype)
    B.edge_index = B.edge_index.to(G.edge_index.dtype)
    B.edge_attr = B.edge_attr.to(G.edge_attr.dtype)
        
    return G, B, G_not_normalized, B_not_normalized, Boundary, front_door






if __name__ == '__main__':
    
    # Get the data
    # Boundary, front_door, room_centroids, bathroom_centroids, kithchen_centroids = get_info()
    
    Boundary = "POLYGON ((105.44810944559121 78.39738655652667, 198.6042371321142 78.39738655652667, 198.6042371321142 91.70540479745854, 227.63991329414748 91.70540479745854, 227.63991329414748 177.60261344347333, 25.599999999999994 177.60261344347333, 25.599999999999994 101.38396351813626, 52.21603648186374 101.38396351813626, 52.21603648186374 78.39738655652667, 105.44810944559121 78.39738655652667))"

    front_door = "POLYGON ((230.4 176.09107942600446, 230.4 154.0103857791844, 227.63991329414748 154.0103857791844, 227.63991329414748 176.09107942600446, 230.4 176.09107942600446))"
    # Data of the inner rooms or bathrooms
    room_centroids  = [(81, 105), (55, 151), (134, 105)]
    bathroom_centroids = [(40, 115), (101, 160)]
    kithchen_centroids = [(100, 163)]
    
    # ========================================================================
    # Preprocessing
    G, B, G_not_normalized, B_not_normalized, Boundary_as_polygon, front_door_as_polygon = preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kithchen_centroids)
    the_door = Point(B_not_normalized.x[-1][1:].detach().cpu().numpy()).buffer(3)
    
    #=========================================================================
    # Model
    model_path = r"D:\Grad\Best models\v2\Best_model_V2.pt"
    # model_path = r"D:\Grad\Best models\v3_UnScalled\Best_model_V3.pt"
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #=========================================================================
    # Inference
    prediction    = model(G.to(device), B.to(device))
    w_predicted   = prediction[0].detach().cpu().numpy()
    h_predicted   = prediction[1].detach().cpu().numpy()
    prediction    = np.concatenate([w_predicted.reshape(-1, 1), h_predicted.reshape(-1, 1)], axis=1)
    
    #=========================================================================
    # Rescaling back to the original values
    # G.x[:, -2] = G.x[:, -2] * G_x_std + G_x_mean
    #=========================================================================
    # Visualization
    output = FloorPlan_multipolygon(G_not_normalized, prediction)
    polygons = output.get_multipoly(Boundary_as_polygon, the_door)
    polygons.plot(cmap='Dark2_r', figsize=(4, 4), alpha=0.8, linewidth=0.8, edgecolor='black');
    plt.savefig('hello.png')
    print(G_not_normalized.x)