import pandas as pd
import numpy as np

class KMeansClustering:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.steps = [] # Lưu lịch sử các bước để hiển thị

    def _euclidean_distance(self, point, centroid):
        """Tính khoảng cách Euclidean giữa 1 điểm và trọng tâm"""
        return np.sqrt(np.sum((point - centroid) ** 2))

    def fit(self, df):
        # Chỉ lấy các cột dữ liệu số (bỏ cột tên điểm nếu có, ví dụ: 'Point')
        data_numeric = df.select_dtypes(include=[np.number])
        X = data_numeric.values
        self.steps = []

        # 1. Khởi tạo: Chọn ngẫu nhiên k điểm làm trọng tâm ban đầu
        # (Để giống ví dụ slide, ta có thể chọn ngẫu nhiên, hoặc cố định nếu muốn)
        np.random.seed(42) # Cố định seed để kết quả ổn định
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Gán cụm (Assign Clusters)
            clusters = [[] for _ in range(self.k)]
            labels = []
            
            for point in X:
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_index = np.argmin(distances)
                clusters[closest_centroid_index].append(point)
                labels.append(closest_centroid_index)
            
            # Lưu trạng thái hiện tại
            self.steps.append({
                "iteration": i + 1,
                "centroids": self.centroids.copy(),
                "labels": np.array(labels),
                "data": df.copy() # Lưu dataframe gốc để map lại tên điểm
            })

            # 3. Cập nhật trọng tâm (Update Centroids)
            new_centroids = []
            for cluster in clusters:
                if cluster: # Nếu cụm không rỗng
                    new_centroids.append(np.mean(cluster, axis=0))
                else: 
                    # Nếu cụm rỗng, giữ nguyên trọng tâm cũ (hoặc chọn lại ngẫu nhiên)
                    new_centroids.append(self.centroids[len(new_centroids)])
            
            new_centroids = np.array(new_centroids)

            # 4. Kiểm tra hội tụ (Nếu trọng tâm không đổi thì dừng)
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids

        return self.steps