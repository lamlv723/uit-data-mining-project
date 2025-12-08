import pandas as pd
from itertools import combinations

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}  
        self.rules = []     

    def _get_support(self, transactions, itemset):
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(transactions)

    def fit(self, df):
        """
        Input: df với 2 cột [TransactionID, Item]
        Logic: Gom nhóm theo TransactionID để tạo thành list các tập hợp
        """
        # 1. Tự động lấy tên 2 cột đầu tiên
        col_id = df.columns[0]   # Ví dụ: TransactionID
        col_item = df.columns[1] # Ví dụ: Item

        # 2. Chuyển đổi từ dạng bảng dọc (Long format) sang danh sách tập hợp
        # Ví dụ: 
        # 01, i1
        # 01, i2  --> Tập giao dịch: {'i1', 'i2'}
        # groupby(col_id)[col_item] sẽ gom tất cả Item của cùng 1 ID lại
        transactions = df.groupby(col_id)[col_item].apply(set).tolist()

        # --- BẮT ĐẦU THUẬT TOÁN APRIORI ---
        
        # Tạo tập phổ biến 1 phần tử (L1)
        all_items = set()
        for t in transactions:
            all_items.update(t)
            
        current_l = {}
        for item in all_items:
            itemset = frozenset([item])
            supp = self._get_support(transactions, itemset)
            if supp >= self.min_support:
                current_l[itemset] = supp
        
        self.itemsets.update(current_l)

        # Vòng lặp tìm k-itemset (k=2, 3...)
        k = 2
        while True:
            candidates = set()
            l_list = list(current_l.keys())
            
            for i in range(len(l_list)):
                for j in range(i + 1, len(l_list)):
                    # Hợp 2 tập lại
                    union_set = l_list[i].union(l_list[j])
                    if len(union_set) == k:
                        candidates.add(union_set)
            
            next_l = {}
            for cand in candidates:
                supp = self._get_support(transactions, cand)
                if supp >= self.min_support:
                    next_l[cand] = supp
            
            if not next_l:
                break
                
            self.itemsets.update(next_l)
            current_l = next_l
            k += 1

    def generate_rules(self):
        self.rules = []
        for itemset, support in self.itemsets.items():
            if len(itemset) < 2:
                continue
            
            for i in range(1, len(itemset)):
                for lhs in combinations(itemset, i):
                    lhs = frozenset(lhs)
                    rhs = itemset - lhs
                    
                    if lhs in self.itemsets:
                        conf = support / self.itemsets[lhs]
                        if conf >= self.min_confidence:
                            self.rules.append({
                                'Vế trái': ', '.join(list(lhs)),
                                'Vế phải': ', '.join(list(rhs)),
                                'Support': round(support, 4),
                                'Confidence': round(conf, 4)
                            })
        
        if not self.rules:
            return pd.DataFrame()
        return pd.DataFrame(self.rules)

    def get_itemsets(self):
        data = []
        for itemset, supp in self.itemsets.items():
            data.append({
                "Tập hạng mục": ', '.join(list(itemset)),
                "Kích thước": len(itemset),
                "Support": round(supp, 4)
            })
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)