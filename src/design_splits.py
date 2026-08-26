"""
Deterministic Event-Level Data Splitting Utility (Seed 42)
Computes leak-free Event-Level Group splits for URFD, MultiCamera, and Le2i LOLO.
"""

import os
import pandas as pd
import numpy as np

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(root_dir, "R&D", "Dataset_Analysis", "dataset_manifest.csv")
    
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    np.random.seed(42)

    print("=== DETERMINISTIC SPLIT DESIGN (SEED 42) ===")

    # 1. URFD SPLIT
    urfd_df = df[df['dataset'] == 'URFD']
    urfd_events = urfd_df.groupby('event_id')['label'].first().reset_index()
    falls = urfd_events[urfd_events['label'] == 'FALL']['event_id'].tolist()
    normals = urfd_events[urfd_events['label'] == 'NORMAL']['event_id'].tolist()

    np.random.shuffle(falls)
    np.random.shuffle(normals)

    urfd_split = {
        'train': sorted(falls[:21] + normals[:28]),
        'val': sorted(falls[21:25] + normals[28:34]),
        'test': sorted(falls[25:] + normals[34:])
    }

    print(f"\nURFD Event Split (Total: {len(urfd_events)}):")
    print(f"  Train ({len(urfd_split['train'])} events): {urfd_split['train']}")
    print(f"  Val   ({len(urfd_split['val'])} events): {urfd_split['val']}")
    print(f"  Test  ({len(urfd_split['test'])} events): {urfd_split['test']}")

    # 2. MULTICAMERA SPLIT
    mc_df = df[df['dataset'] == 'MultiCamera']
    mc_events = mc_df.groupby('event_id')['label'].first().reset_index()
    mc_falls = mc_events[mc_events['label'] == 'FALL']['event_id'].tolist()
    np.random.shuffle(mc_falls)

    mc_split = {
        'train': sorted(mc_falls[:16] + ['chute23']),
        'val': sorted(mc_falls[16:19]),
        'test': sorted(mc_falls[19:] + ['chute24'])
    }

    print(f"\nMultiCamera Event Split (Total: {len(mc_events)}):")
    print(f"  Train ({len(mc_split['train'])} events): {mc_split['train']}")
    print(f"  Val   ({len(mc_split['val'])} events): {mc_split['val']}")
    print(f"  Test  ({len(mc_split['test'])} events): {mc_split['test']}")

if __name__ == "__main__":
    main()
