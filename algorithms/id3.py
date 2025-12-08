import pandas as pd
import numpy as np

class ID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.root_node = None
        self.default_class = None  # Lớp phổ biến nhất (dự phòng khi gặp nhánh cụt)

    def _entropy(self, s):
        """Tính độ hỗn loạn (Entropy)"""
        counts = np.unique(s, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, df, attribute, target_col):
        """Tính độ lợi thông tin (Information Gain)"""
        total_entropy = self._entropy(df[target_col])
        
        values, counts = np.unique(df[attribute], return_counts=True)
        weighted_entropy = 0
        for v, count in zip(values, counts):
            subset = df[df[attribute] == v]
            weighted_entropy += (count / len(df)) * self._entropy(subset[target_col])
            
        return total_entropy - weighted_entropy

    def _id3(self, df, attributes, target_col, parent_node_class=None):
        """Đệ quy xây dựng cây"""
        unique_targets = np.unique(df[target_col])

        # 1. Nếu tất cả mẫu cùng 1 lớp -> Trả về lá
        if len(unique_targets) <= 1:
            return unique_targets[0]
        
        # 2. Nếu hết thuộc tính -> Trả về lớp phổ biến nhất
        if len(attributes) == 0:
            return parent_node_class
        
        # Backup lớp phổ biến nhất hiện tại
        if len(df) == 0:
            return parent_node_class
        parent_node_class = df[target_col].mode()[0]

        # 3. Chọn thuộc tính tốt nhất
        gains = [self._information_gain(df, attr, target_col) for attr in attributes]
        best_attr_index = np.argmax(gains)
        best_attr = attributes[best_attr_index]
        
        tree = {best_attr: {}}
        remaining_attributes = [i for i in attributes if i != best_attr]
        
        # 4. Phân nhánh
        for value in np.unique(df[best_attr]):
            subset = df[df[best_attr] == value]
            subtree = self._id3(subset, remaining_attributes, target_col, parent_node_class)
            tree[best_attr][value] = subtree
            
        return tree

    def fit(self, df, target_col, id_col=None):
        """Huấn luyện mô hình từ dữ liệu thô"""
        data = df.copy()
        if id_col and id_col in data.columns:
            data = data.drop(columns=[id_col])
            
        # Lưu lớp phổ biến nhất của toàn bộ dữ liệu (để dự đoán trường hợp lạ)
        self.default_class = data[target_col].mode()[0]
        
        attributes = [col for col in data.columns if col != target_col]
        self.tree = self._id3(data, attributes, target_col)
        return self.tree

    def predict(self, sample):
        """
        Dự đoán kết quả cho 1 mẫu.
        sample: Dictionary {'Outlook': 'Sunny', 'Temp': 'Hot', ...}
        """
        if not self.tree:
            return "Mô hình chưa được huấn luyện"

        node = self.tree
        while isinstance(node, dict):
            # Lấy tên thuộc tính tại nút hiện tại
            attribute = list(node.keys())[0]
            
            # Lấy giá trị của thuộc tính đó trong mẫu cần dự đoán
            value = sample.get(attribute)
            
            # Nếu giá trị không tồn tại trong cây (nhánh lạ) -> Trả về mặc định
            if value not in node[attribute]:
                return f"Không xác định (Dự đoán: {self.default_class})"
            
            # Đi tiếp xuống nhánh con
            node = node[attribute][value]
        
        return node

    def get_graphviz_dot(self):
        """Tạo mã DOT để vẽ cây"""
        if not self.tree:
            return ""
        
        dot = ["digraph Tree {", "node [shape=box, fontname=\"Arial\"];", "edge [fontname=\"Arial\"];"]
        node_counter = 0
        
        def traverse(node, parent_id=None, edge_label=""):
            nonlocal node_counter
            current_id = str(node_counter)
            node_counter += 1
            
            if not isinstance(node, dict):
                # Node Lá (Kết quả)
                dot.append(f'{current_id} [label="{node}", style=filled, fillcolor="#ff4b4b", fontcolor=white, shape=oval];')
                if parent_id is not None:
                    dot.append(f'{parent_id} -> {current_id} [label="{edge_label}"];')
                return

            # Node Gốc/Nhánh (Thuộc tính)
            attr_name = list(node.keys())[0]
            dot.append(f'{current_id} [label="{attr_name}", style=filled, fillcolor="#f0f2f6"];')
            
            if parent_id is not None:
                dot.append(f'{parent_id} -> {current_id} [label="{edge_label}"];')
            
            for value, child in node[attr_name].items():
                traverse(child, current_id, value)

        traverse(self.tree)
        dot.append("}")
        return "\n".join(dot)