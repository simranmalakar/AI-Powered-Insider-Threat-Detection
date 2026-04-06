import networkx as nx
import pandas as pd
from pyvis.network import Network
import os

def build_behavioral_graph(df: pd.DataFrame, filter_type: str = "all") -> nx.Graph:
    """
    Builds a NetworkX graph from access logs to map relationships 
    between Users (nodes) and their accessed resources (files, usbs, emails).
    Supports filters like 'anomalies' and 'users'.
    """
    G = nx.Graph()
    
    # Cap the graph interactions to the first 1000 rows. 
    # PyVis ForceAtlas2 physics equation physically crashes browsers with >5000 edges.
    if len(df) > 1000:
        df = df.head(1000)
    
    if filter_type == 'users':
        # Connect users that share the same role
        roles = {}
        for idx, row in df.iterrows():
            user = row['user']
            role = row.get('role', 'User')
            G.add_node(user, group='user', title=role, color='#4da6ff', size=25)
            roles.setdefault(role, set()).add(user)
            
        for role, users in roles.items():
            user_list = list(users)
            for i in range(len(user_list)):
                for j in range(i+1, len(user_list)):
                    # Edge weight 1 to represent same role connection
                    G.add_edge(user_list[i], user_list[j], weight=1, color='#555555')
        return G

    for idx, row in df.iterrows():
        is_anomaly = row.get('anomaly_score', 0) > 0.7 # threshold for red edges
        
        # Apply anomalies filter
        if filter_type == 'anomalies' and not is_anomaly:
            continue
            
        user = row['user']
        event_type = row['event_type']
        details = row.get('details', 'Unknown')
        
        # Add user node
        G.add_node(user, group='user', title=row.get('role', 'User'), color='#4da6ff', size=25)
        
        # The target depends on event type
        target_node = f"{event_type}:{details[:15]}..."
        
        # Node group definitions
        if event_type == 'file_access':
            group, color = 'file', '#ff9933'
        elif event_type == 'usb_connect':
            group, color = 'usb', '#ff4d4d'
        else:
            group, color = 'email', '#4dff4d'
            
        G.add_node(target_node, group=group, title=details, color=color, size=15)
        
        # Edge styling
        edge_color = '#ff0000' if is_anomaly else '#cccccc'
        weight = 3 if is_anomaly else 1
        
        # Add or update edge weight
        if G.has_edge(user, target_node):
            G[user][target_node]['weight'] += weight
            if is_anomaly:
                G[user][target_node]['color'] = edge_color
        else:
            G.add_edge(user, target_node, weight=weight, color=edge_color)
            
    return G

def export_graph_to_pyvis(G: nx.Graph, output_dir: str = 'app/static', filename: str = 'graph.html') -> str:
    """
    Exports a NetworkX graph to an interactive HTML file using PyVis.
    Returns the path to the HTML file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create PyVis network
    net = Network(height='100vh', width='100%', bgcolor='#1a1a1a', font_color='white')
    net.from_nx(G)
    
    # Configure physics for better layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    html_path = os.path.join(output_dir, filename)
    net.save_graph(html_path)
    return html_path
