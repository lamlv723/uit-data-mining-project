import pandas as pd
import numpy as np

class CARTDecisionTree:
    def __init__(self):
        self.tree = None
        self.default_class = None

    def _gini(self, s):
        """Tính chỉ số Gini: 1 - sum(p^2)"""
        counts = np.unique(s, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _calculate_gain(self, df, attribute, target_col):
        """Tính độ giảm bất định (Gini Gain)"""
        total_gini = self._gini(df[target_col])
        
        values, counts = np.unique(df[attribute], return_counts=True)
        weighted_gini = 0
        for v, count in zip(values, counts):
            subset = df[df[attribute] == v]
            weighted_gini += (count / len(df)) * self._gini(subset[target_col])
            
        return total_gini - weighted_gini

    def _build_tree(self, df, attributes, target_col, parent_node_class=None):
        unique_targets = np.unique(df[target_col])

        if len(unique_targets) <= 1:
            return unique_targets[0]
        
        if len(attributes) == 0 or len(df) == 0:
            return parent_node_class
        
        parent_node_class = df[target_col].mode()[0]

        # Chọn thuộc tính tốt nhất dựa trên Gini Gain
        gains = [self._calculate_gain(df, attr, target_col) for attr in attributes]
        best_attr_index = np.argmax(gains)
        best_attr = attributes[best_attr_index]
        
        tree = {best_attr: {}}
        remaining_attributes = [i for i in attributes if i != best_attr]
        
        for value in np.unique(df[best_attr]):
            subset = df[df[best_attr] == value]
            subtree = self._build_tree(subset, remaining_attributes, target_col, parent_node_class)
            tree[best_attr][value] = subtree
            
        return tree

    def fit(self, df, target_col, drop_cols=None):
        data = df.copy()
        if drop_cols:
            cols_to_drop = [c for c in drop_cols if c in data.columns]
            data = data.drop(columns=cols_to_drop)
            
        self.default_class = data[target_col].mode()[0]
        attributes = [col for col in data.columns if col != target_col]
        
        self.tree = self._build_tree(data, attributes, target_col)
        return self.tree

    def predict(self, sample):
        if not self.tree: return "Chưa huấn luyện"
        node = self.tree
        while isinstance(node, dict):
            attr = list(node.keys())[0]
            val = sample.get(attr)
            if val not in node[attr]:
                return self.default_class 
            node = node[attr][val]
        return node

    def get_rules(self):
        if not self.tree: return pd.DataFrame()
        rules = []
        def traverse(node, current_rule):
            if not isinstance(node, dict):
                rule_str = " AND ".join([f"{k} == '{v}'" for k, v in current_rule])
                rules.append({"Luật (Condition)": rule_str, "Kết quả (Prediction)": node})
                return
            attr = list(node.keys())[0]
            for val, child in node[attr].items():
                traverse(child, current_rule + [(attr, val)])
        traverse(self.tree, [])
        return pd.DataFrame(rules)

    def get_graphviz_dot(self):
        if not self.tree: return ""
        dot = [
            "digraph Tree {", 
            'node [shape=box, style="filled, rounded", fontname="Helvetica", penwidth=1, color="#444444"];', 
            'edge [fontname="Helvetica"];'
        ]
        node_counter = 0
        def traverse(node, parent_id=None, edge_label=""):
            nonlocal node_counter
            current_id = str(node_counter)
            node_counter += 1
            node_val = str(node)
            label_val = str(edge_label)
            
            if not isinstance(node, dict):
                dot.append(f'{current_id} [label="{node_val}", fillcolor="#ff4b4b", fontcolor="white", shape=oval, color="#cc0000"];')
                if parent_id is not None:
                    dot.append(f'{parent_id} -> {current_id} [label="{label_val}"];')
                return

            attr = str(list(node.keys())[0])
            dot.append(f'{current_id} [label="{attr}", fillcolor="#e3f2fd"];') # Màu xanh nhạt để phân biệt với ID3
            
            if parent_id is not None:
                dot.append(f'{parent_id} -> {current_id} [label="{label_val}"];')
            for val, child in node[attr].items():
                traverse(child, current_id, val)
        traverse(self.tree)
        dot.append("}")
        return "\n".join(dot)