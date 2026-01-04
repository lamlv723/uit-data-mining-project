import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self, use_laplace=False):
        self.use_laplace = use_laplace
        self.priors = {}      # P(Ci)
        self.likelihoods = {} # P(Xk|Ci)
        self.classes = []
        self.features = []
        self.target_col = ""
        self.vocab = {}       # Lưu các giá trị duy nhất của từng cột để tính Laplace

    def fit(self, df, target_col, drop_cols=None):
        self.target_col = target_col
        self.classes = df[target_col].unique()
        self.features = [col for col in df.columns if col != target_col]

        # Xử lý loại bỏ cột nhiễu
        data = df.copy()
        if drop_cols:
            cols_to_drop = [c for c in drop_cols if c in data.columns]
            data = data.drop(columns=cols_to_drop)

        self.target_col = target_col
        self.classes = data[target_col].unique()
        self.features = [col for col in data.columns if col != target_col]
        
        # Lưu vocabulary cho từng cột (dùng cho Laplace: số giá trị rời rạc r)
        for col in self.features:
            self.vocab[col] = df[col].unique()

        total_samples = len(data)
        num_classes = len(self.classes)

        # 1. Tính xác suất tiên nghiệm P(Ci)
        for c in self.classes:
            class_count = len(df[df[target_col] == c])
            if self.use_laplace:
                # Công thức Laplace: (count + 1) / (total + m)
                self.priors[c] = (class_count + 1) / (total_samples + num_classes)
            else:
                self.priors[c] = class_count / total_samples

        # 2. Tính xác suất có điều kiện P(Xk|Ci)
        for col in self.features:
            self.likelihoods[col] = {}
            for c in self.classes:
                # Lấy tập dữ liệu con thuộc lớp c
                subset = df[df[target_col] == c]
                class_count = len(subset)
                
                # Đếm số lần xuất hiện của từng giá trị trong cột col thuộc lớp c
                value_counts = subset[col].value_counts()
                
                self.likelihoods[col][c] = {}
                
                # Duyệt qua tất cả giá trị có thể có của cột
                unique_values = self.vocab[col]
                
                for val in unique_values:
                    count = value_counts.get(val, 0)
                    
                    if self.use_laplace:
                        # Công thức Laplace: (count + 1) / (class_count + r)
                        r = len(unique_values) # Số giá trị rời rạc của thuộc tính
                        prob = (count + 1) / (class_count + r)
                    else:
                        # Cách tính thường
                        prob = count / class_count if class_count > 0 else 0
                        
                    self.likelihoods[col][c][val] = prob

    def predict(self, sample):
        """
        Dự đoán lớp cho 1 mẫu.
        sample: Dictionary {'Weather': 'Sunny', 'Temp': 'Hot', ...}
        """
        posteriors = {} # P(Ci|X)
        
        # P(Ci|X) tỷ lệ thuận với P(Ci) * Product(P(Xk|Ci))
        for c in self.classes:
            posterior = self.priors[c]
            
            details = []
            details.append(f"P({c})={posterior:.4f}")
            
            for col, val in sample.items():
                if col in self.features:
                    # Lấy xác suất P(val|c). Nếu giá trị lạ chưa từng gặp -> 0
                    if val in self.likelihoods[col][c]:
                        prob = self.likelihoods[col][c][val]
                    else:
                        # Xử lý giá trị lạ (chưa có trong train set)
                        prob = 1 / (len(self.vocab[col]) + 1) if self.use_laplace else 0
                    
                    posterior *= prob
                    details.append(f"P({val}|{c})={prob:.4f}")
            
            posteriors[c] = {
                "score": posterior,
                "details": " × ".join(details)
            }
            
        # Chọn lớp có xác suất cao nhất
        best_class = max(posteriors, key=lambda k: posteriors[k]["score"])
        return best_class, posteriors

    def get_details(self):
        """Trả về bảng P(Ci) và P(Xk|Ci) để hiển thị"""
        return self.priors, self.likelihoods