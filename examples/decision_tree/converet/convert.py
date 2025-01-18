import joblib
import numpy as np
import struct


clf = joblib.load('../train/iris_decision_tree_model.pkl')


def convert_tree_to_ggml(clf, filename='iris_ggml_model.bin'):
    with open(filename, 'wb') as fout:
        fout.write(struct.pack("i", 0x67676d6c))  # Magic: "ggml"

        n_features = clf.n_features_in_
        fout.write(struct.pack("i", n_features)) 
        
        # Save decision tree parameters
        tree_ = clf.tree_
        n_nodes = tree_.node_count  # Total number of nodes in the tree
        
        fout.write(struct.pack("i", n_nodes))
        
        # Write the tree structure (node information)
        for i in range(n_nodes):
            # Split feature
            feature = tree_.feature[i]
            # Threshold for the split
            threshold = tree_.threshold[i]
            # Left and right children (indices)
            left_child = tree_.children_left[i]
            right_child = tree_.children_right[i]
            # Value is an array of size (n_classes)
            value = tree_.value[i].flatten()

            fout.write(struct.pack("i", feature))  # Feature index
            fout.write(struct.pack("f", threshold))  # Threshold
            fout.write(struct.pack("i", left_child))  # Left child
            fout.write(struct.pack("i", right_child))  # Right child
            fout.write(struct.pack(f"{len(value)}f", *value))  # Class probabilities

    print(f"Model converted to GGML format: {filename}")

convert_tree_to_ggml(clf)
