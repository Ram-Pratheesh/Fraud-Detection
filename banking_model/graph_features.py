import pandas as pd
import networkx as nx
import os
import numpy as np

def build_graph_features():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data", "Fraud_Data.csv")
    output_path = os.path.join(base_dir, "output", "card_features.csv")
        
    print(f"Loading REAL E-Commerce Fraud dataset from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Please ensure 'Fraud_Data.csv' is in 'banking_model/data/'.")
        return

    df = pd.read_csv(input_path)
    
    # 1. MAP NATIVE COLUMNS TO OUR ENGINE SCHEMA
    rename_rules = {
        'user_id': 'card_id',
        'ip_address': 'ip_address',
        'device_id': 'device_fingerprint',
        'purchase_time': 'txn_timestamp',
        'purchase_value': 'amount_inr',
        'class': 'is_fraud'
    }
    df.rename(columns=rename_rules, inplace=True)
    
    df['card_id'] = df['card_id'].astype(str)
    df['ip_address'] = df['ip_address'].fillna('UNKNOWN_IP').astype(str)
    df['device_fingerprint'] = df['device_fingerprint'].fillna('UNKNOWN_DEVICE').astype(str)

    print("Building bipartite Graph (Node relationships)...")
    G = nx.Graph()
    
    for _, row in df.iterrows():
        c_node = f"CARD_{row['card_id']}"
        ip_node = f"IP_{row['ip_address']}"
        dev_node = f"DEV_{row['device_fingerprint']}"
        
        G.add_node(c_node, type='card')
        G.add_node(ip_node, type='ip')
        G.add_node(dev_node, type='device')
        
        G.add_edge(c_node, ip_node, relation='used_ip')
        G.add_edge(c_node, dev_node, relation='used_device')

    print("Extracting Graph Network Features (Degrees of Separation)...")
    dc = nx.degree_centrality(G)
    ip_shared_counts = []
    dev_shared_counts = []
    card_centralities = []
    accounts_reachable_2hops = []
    distinct_ips_linked = []

    for _, row in df.iterrows():
        c_node = f"CARD_{row['card_id']}"
        c_ip_shared = 0
        c_dev_shared = 0
        reachable = 0
        linked_ips = 0

        if c_node in G:
            # 2-hop logic (Card -> IP/Device -> Card)
            path_lengths = nx.single_source_shortest_path_length(G, c_node, cutoff=2)
            for node, dist in path_lengths.items():
                if dist == 2 and G.nodes[node].get('type') == 'card':
                    reachable += 1

            for neighbor in G.neighbors(c_node):
                shared = max(0, G.degree(neighbor) - 1)
                if G.nodes[neighbor]['type'] == 'ip':
                    c_ip_shared += shared
                    linked_ips += 1
                elif G.nodes[neighbor]['type'] == 'device':
                    c_dev_shared += shared
            card_centralities.append(dc.get(c_node, 0.0))
        else:
            card_centralities.append(0.0)

        ip_shared_counts.append(c_ip_shared)
        dev_shared_counts.append(c_dev_shared)
        accounts_reachable_2hops.append(reachable)
        distinct_ips_linked.append(linked_ips)

    df['ip_shared_card_count'] = ip_shared_counts
    df['device_shared_card_count'] = dev_shared_counts
    df['degree_centrality'] = card_centralities
    df['accounts_reachable_in_2_hops'] = accounts_reachable_2hops
    df['distinct_ips'] = distinct_ips_linked

    # 2. VALIDATING SUPERVISED LABELS
    print(f"Dataset successfully mapped! Native Fraud Count detected: {df['is_fraud'].sum()} out of {len(df)}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except:
        pass
        
    df.to_csv(output_path, index=False)
    print(f"Done! Network topological features exported to {output_path}.")

if __name__ == "__main__":
    build_graph_features()
