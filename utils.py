# for data wrangling
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import distinctipy
from collections import defaultdict

from torch_geometric.utils import from_networkx
import torch

# for saving and loading the images
import os
import uuid
import time


# from shapely import Point, MultiPolygon, GeometryCollection, Polygon, ops, LineString, unary_union, intersection_all
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, Point, LineString, box
from shapely.ops import unary_union
import shapely.affinity as aff
from shapely.wkt import loads
import geopandas as gpd

room_embeddings = {
    'living': 0,
    'room': 1,
    'kitchen': 2,
    'bathroom': 3,
}

poly_types = list(room_embeddings.keys())
N = len(poly_types)
colors = (np.array(distinctipy.get_colors(N)) * 255).astype(np.uint8)
room_color = {room_name: colors[i] for i, room_name in enumerate(poly_types)}


# Functios
def Handling_dubplicated_nodes(boundary, door):
    
    """
    This function is used to handle the duplicated nodes in the boundary graph.
    As some coords of the boundary graph are near to each other, so we will consider them the same node.
    
    Input:
        boundary graph, front door as polygons
    Output:
        boundary graph with no duplicated nodes. Also with the front door embedded.
    """
    
    coords = boundary.exterior.coords[:]
        
    # creating points:
    points = []
    for p in coords:
        points.append(Point(p))

    graph = nx.Graph()
    # type of the node: 0 for boundary, 1 for front_door
    graph.add_node(0, type=0, centroid=coords[0])

    # to save the index if there is a node will not be added
    current = 0
    name = 1

    for i in range(1, len(coords)):
        dis = points[i].distance(points[current])
        if dis >= 5:
            # type of the node, edge = 0, front_door = 1
            graph.add_node(name, type=0, centroid=coords[i])
            current = i
            name += 1

    # Checking the distance between first and last node [if the distance is small, so we will consider them the same point]
    nodes_names = list(graph.nodes)
    first_node = Point(graph.nodes[nodes_names[0]]['centroid'])
    last_node  = Point(graph.nodes[nodes_names[-1]]['centroid'])
    if first_node.distance(last_node) <= 5:
        graph.remove_node(nodes_names[-1])
        nodes_names = list(graph.nodes)
        
    points_of_current_graph = []
    for node in graph:
        points_of_current_graph.append(Point(graph.nodes[node]['centroid']))

    # Adding edges between nodes.
    for i in range(len(nodes_names)-1):
        dis = points_of_current_graph[i].distance(points_of_current_graph[i+1])
        graph.add_edge(nodes_names[i],nodes_names[i+1], distance=dis)

    # Adding an edge between the last and the first nodes.
    dis = points_of_current_graph[nodes_names[0]].distance(points_of_current_graph[nodes_names[-1]])

    graph.add_edge(nodes_names[0], nodes_names[-1], distance=dis)
    
    # adding the front door
    graph = adding_door(graph, door, points_of_current_graph)
    
    return graph

def adding_door(boundary_graph, door, points):
    """
    This function is used to add the front door to the boundary graph.
    Input:
        boundary graph: graph of the boundary of the floor plan.
        door: front door as polygon.
        points: list of the points of the boundary graph. to use it to detect best place for the door.
    """
    nearest_edge = None
    nearest_dist = float('inf')
    
    dx = door.bounds[2] - door.bounds[0]
    dy = door.bounds[3] - door.bounds[1]
    door_oriantation_horizontal = dx > dy

    for edge in boundary_graph.edges():
        p1 = points[edge[0]]
        p2 = points[edge[1]]

        line = LineString([p1, p2])

        # checking the oriantation of the lines.
        p1x, p1y = p1.x, p1.y
        p2x, p2y = p2.x, p2.y
        dx = abs(p2x - p1x)
        dy = abs(p2y - p1y)
        line_oriantation_horizontal = dx > dy
        
        # print(f'edge: {edge}, line is: {line_oriantation_horizontal}, door is: {door_oriantation_horizontal}')
        if door_oriantation_horizontal == line_oriantation_horizontal:
            # getting nearest - with same oriantation - edge
            dist = door.distance(line)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_edge = edge

    # print(f'nearest is: {nearest_edge}')
    boundary_graph.remove_edge(*nearest_edge)
    
    door_ind = len(boundary_graph)
    door_centroid = door.centroid
    boundary_graph.add_node(door_ind, type=1, centroid=(door_centroid.x, door_centroid.y))

    dist = door_centroid.distance(Point(boundary_graph.nodes[nearest_edge[0]]['centroid']))
    boundary_graph.add_edge(nearest_edge[0], door_ind, distance=dist)

    dist = door_centroid.distance(Point(boundary_graph.nodes[nearest_edge[1]]['centroid']))
    boundary_graph.add_edge(nearest_edge[1], door_ind, distance=dist)
    
    return boundary_graph

def centroids_to_graph(floor_plan, living_to_all=False, all_conected=False):
    """
    Generating a graph for a specific floor plan
    
    Input: 
        floor_plan: a dictionary [key: type of room, value: list of centroids]
        living_to_all: boolean, if True, we will connect all rooms to the living room.
        all_conected: boolean, if True, we will connect all rooms to each other.
    
    Output:
        G: a networkx graph represents the floor plan.
    """
    # Creating new graph
    G = nx.Graph()
    
    # Embeding each room in a node.
    for type_, list_of_centroids in floor_plan.items():
        for i, centroid in enumerate(list_of_centroids):

            currentNodeName = f'{type_}_{i}'
            G.add_node(currentNodeName,
                roomType_name = type_,
                roomType_embd = room_embeddings[type_],
                actualCentroid_x = centroid[0],
                actualCentroid_y = centroid[1])
            
                                        
    # if we need to connect all nodes to the living                    
    if living_to_all: 
        living_cen = Point(G.nodes['living_0']['actualCentroid_x'], G.nodes['living_0']['actualCentroid_y'])
        for node in G.nodes():
                if G.nodes[node]['roomType_name'] != 'living':
                    point = Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])
                    dis = living_cen.distance(point)
                    # adding edges between the living and all geoms
                    G.add_edge('living_0', node, distance=round(dis, 3))
                    
    # if we need to connect all nodes to each others  
    if all_conected: 
        for node in G.nodes():
            current_node_centeroid = Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])

            for other_node in G.nodes():
                if other_node != node: # for all other rooms
                    other_node_centeroid = Point(G.nodes[other_node]['actualCentroid_x'], G.nodes[other_node]['actualCentroid_y'])

                    dis = current_node_centeroid.distance(other_node_centeroid)
                    # adding edges between the the c urrent node and the other nodes
                    G.add_edge(node, other_node, distance=round(dis, 3))

    return G

def boundary_to_image(boundary_wkt, front_door_wkt):
    """
    Taking the boundary and the front door as polygons and return them as Image.   
    """
    boundary = shapely.wkt.loads(boundary_wkt)
    front_door = shapely.wkt.loads(front_door_wkt)
    
    boundary = scale(boundary)
    front_door = scale(front_door)
    
    plt.figure(figsize=(5, 5))
    gpd.GeoSeries([boundary, front_door]).plot(cmap='tab10');
    # plt.xlim(0, 256);
    # plt.ylim(0, 256);
    
    path = os.getcwd() + "/Outputs/boundary.png"
    plt.savefig(path)
    plt.close()
    
    return path
    
def get_user_inputs_as_image(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids):
    """
        Covert the user inputs [boundary, front_door, centroids] to an image.
        to be more understandable.
    """
    boundary = shapely.wkt.loads(boundary_wkt)
    front_door = shapely.wkt.loads(front_door_wkt)
    
    boundary = scale(boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kitchen_centroids = [scale(x) for x in kitchen_centroids]
    
    polys = defaultdict(list)

    for center in room_centroids:
        polys['room'].append(center)

    for center in bathroom_centroids:
        polys['bathroom'].append(center)

    for center in kitchen_centroids:
        polys['kitchen'].append(center)

    Input_format = []
    Input_format.append(boundary)
    Input_format.append(front_door)

    for _, poly_list in polys.items():
        Input_format.append(unary_union(poly_list))

    Input_format = gpd.GeoSeries(Input_format)
    Input_format.plot(cmap='twilight', alpha=0.8, linewidth=0.8, edgecolor='black');
    
    # plt.xlim(0, 256);
    # plt.ylim(0, 256);
    
    path = os.getcwd() + '/Outputs/user_inputs.png'
    plt.savefig(path)
    plt.close()
    
    return path

def draw_graph(G):
    """
    This function is used to draw the graph of rooms and user constrains.
    """
    #  nodes positions for drawing, note that we invert the y pos
    pos = {node: (G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y']) for node in G.nodes}
    
    colormap = [room_color[G.nodes[node]['roomType_name']]/255 for node in G]
    
    nx.draw(G, pos=pos, node_color=colormap, with_labels=True, font_size=12)
    
    # plt.xlim(-10, 266)
    # plt.ylim(-266, 10)
    
def draw_graph_boundary(G):
    """
    This function is used to draw the graph of the boundary of the floor plan.
    """
    
    #  nodes positions for drawing, note that we invert the y pos
    pos = {node: (G.nodes[node]['centroid'][0], G.nodes[node]['centroid'][1])  for node in G.nodes}
    
    door_color = '#90EE90'
    other_nodes_color = '#0A2A5B'
    color_map = [door_color if G.nodes[node]['type'] == 1 else other_nodes_color for node in G.nodes]
    
    nx.draw(G, pos=pos, with_labels=True, node_color=color_map, font_color='w', font_size=12)
    
    # plt.xlim(-10, 266)
    # plt.ylim(-266, 10)
    
def draw_both_graphs(boundary_graph, entire_graph):
    # Create a new figure
    plt.figure()
    
    # Draw the boundary graph
    draw_graph_boundary(boundary_graph)
        
    # Draw the entire graph
    draw_graph(entire_graph)
    
    # Save the figure as an image
    path = os.getcwd() + '/Outputs/both_graphs.png'
    plt.savefig(path)
    plt.close()  # Close the figure to free up resources
    
    return path
    
def scale(x):
    if isinstance(x, tuple):
        x = Point(*x)
        
    return aff.scale(x, xfact=1, yfact=-1, origin=(128, 128))

class FloorPlan_multipolygon():
    def __init__(self, graph, prediction):
        self.graph       = graph
        self.prediction  = prediction
        
    def get_room_data(self, room_index):
        """
        Inputs: 
            room_index: index of the room in the graph
            
        Outputs: 
            centroid, w, h of that room.
        """
        # # Using networkX graphs
        # Graph_data = list(self.graph.nodes(data=True))[room_index][1]
        # w = Graph_data['rec_w']
        # h = Graph_data['rec_h']
        # centroid = (Graph_data['actualCentroid_x'], Graph_data['actualCentroid_y'])
        # category = Graph_data['roomType_embd']
        
        # Using pytorhc Garphs
        
        centroid = (self.graph.x[room_index][-2].item(), self.graph.x[room_index][-1].item())
        category = torch.argmax(self.graph.x[:, :7], axis=1)[room_index].item()
        w_pre, h_pre = self.get_predictions(room_index)
            

        data = {
            'centroid': centroid,
            'predic_w': w_pre,
            'predic_h': h_pre,
            'category': category
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
        half_w   = room_data['predic_w'] / 2
        half_h   = room_data['predic_h'] / 2
        
        # bottom_left  = Point(centroid[0] - half_w, centroid[1] - half_h)
        # bottom_right = Point(centroid[0] + half_w, centroid[1] - half_h)
        # top_right    = Point(centroid[0] + half_w, centroid[1] + half_h)
        # top_left     = Point(centroid[0] - half_w, centroid[1] + half_h)
        
        x1 = centroid[0] - half_w
        x2 = centroid[0] + half_w
        y1 = centroid[1] - half_h
        y2 = centroid[1] + half_h
        
        # print(bottom_left, bottom_right, top_right, top_left)
        # box = Polygon([bottom_left, bottom_right, top_right, top_left])
        box_poly = box(x1, y1, x2, y2)
        return box_poly

    def get_multipoly(self, boundary=False, door=False):
        """
        Outputs:
            multi_poly: a shapely multipolygon of all the rooms in the floor plan or graph.
        """
        num_of_rooms = self.graph.x.shape[0]
        similar_polygons = defaultdict(list)
        
        for index in range(num_of_rooms):
            room_data = self.get_room_data(index)
            box = self.create_box(room_data)
            box = box.intersection(boundary.buffer(-3, cap_style=3, join_style=2))
        
            # add each pox to its similar boxes
            room_category = room_data['category']
            if room_category != 0:
                similar_polygons[room_category].append(box)
        

        all_polygons = []
        all_polygons.append(boundary)
        similar_polygons_2 = defaultdict(list)
        already_inside_bath = False
        for room_category, polygons in similar_polygons.items():
            # if room_category == 2:
            #     for poly in polygons:
            #         similar_polygons_2[room_category].append(poly)
                
            # elif room_category == 3:
            #     if len(polygons) == 1:
            #         similar_polygons_2[room_category].append(polygons[0])
            #     else:
            #         # check most polgon has intersection with other polygons
            #         for 
            
            
                
                
            if room_category in (2, 3): # If bathroom or kitchen.
                # combined_polygon = unary_union(polygons)
                # all_polygons.append(combined_polygon)
                # for poly in polygons:
                #     similar_polygons_2[room_category].append(poly)
                
                for bath_or_kitchen in polygons:
                    if any(bath_or_kitchen.intersects(room) for room in similar_polygons[1]): # Chcek if the current bathroom or kitchen intersectes with any room
                        for i, room in enumerate(similar_polygons[1]):
                            if bath_or_kitchen.intersects(room):
                                intersection = bath_or_kitchen.intersection(room)
                                if (intersection.area >= (0.2 * bath_or_kitchen.area)) and (already_inside_bath == False):
                                    # new_bath_or_kitchen = intersection
                                    print('>= 50%')
                                    bath_or_kitchen = bath_or_kitchen.intersection(room.buffer(-3, cap_style=3, join_style=2))
                                    already_inside_bath = True
                                    # bath_or_kitchen   = room.intersection(bath_or_kitchen.buffer(-3, cap_style=3, join_style=3))
                                else:
                                    print('Not >= 50%')
                                    ## If we need to cut from the room
                                    room = room.difference(intersection.buffer(0.3))
                                    similar_polygons[1][i] = room
                                    
                                    ## If we need to cut from the bathroom or kitchen
                                    bath_or_kitchen = bath_or_kitchen.difference(intersection.buffer(4))
                                    
                    similar_polygons_2[room_category].append(bath_or_kitchen)
                    
            else: # If rooms
                existing_polygons = []
                for poly in polygons: # for room in rooms
                    # print(f'Current poly: {poly.centroid}')
                    if any(poly.intersects(exist) for exist in existing_polygons):
                        for exist in existing_polygons:
                            if poly.intersects(exist): # If there is an intersection between current poly and the checking polygon.
                                # print(f'Intersects with: {exist.centroid}')
                                intersection = poly.intersection(exist)
                                if exist.area < poly.area:
                                    # print('1')
                                    difference_polygon = exist.difference(intersection.buffer(4))
                                    
                                    # We cut from the exist so we will remove the old version and add the new version.
                                    similar_polygons_2[room_category].remove(exist)
                                    similar_polygons_2[room_category].append(difference_polygon)
                                    
                                    # Also we add the current polygon.
                                    similar_polygons_2[room_category].append(poly)
                                    
                                    # The same step we didi in similar_polygons_2 we make it here to make the existing_polys the same.
                                    existing_polygons.remove(exist)
                                    existing_polygons.append(difference_polygon)
                                    
                                    existing_polygons.append(poly)
                                    
                                else:
                                    # print('2')
                                    difference_polygon = poly.difference(intersection.buffer(4))
                                    similar_polygons_2[room_category].append(difference_polygon)
                                    # existing_polygons.append(difference_polygon)
                                    # similar_polygons_2[room_category].append(exist)
                                    
                    else: # For the first one
                        # print('No intersection')
                        existing_polygons.append(poly)
                        similar_polygons_2[room_category].append(poly)
                        
                        
        for _, polygons in similar_polygons_2.items():
            all_polygons.append(MultiPolygon(polygons))
        
        if door:
            all_polygons.append(door)
            
        compined_polygons_seperated = gpd.GeoSeries(all_polygons)
        
        return compined_polygons_seperated
    
    def get_predictions(self, room_index):
        """
        Inputs: 
            room_index: index of the room in the graph
        outputs: 
            w_predicted: predicted width for that room
            h_predicted: predicted width for that room
        """
        w_predicted = self.prediction[room_index, 0]
        h_predicted = self.prediction[room_index, 1]
        
        return w_predicted, h_predicted
