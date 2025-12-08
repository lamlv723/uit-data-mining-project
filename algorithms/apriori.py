# algorithms/apriori.py
import pandas as pd
from itertools import combinations

class Apriori:
    def __init__(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}  # Lưu tất cả tập phổ biến
        self.rules = []     # Lưu luật kết hợp

    def _get_support(self, df, itemset):
        # Đếm số dòng chứa tất cả item trong itemset
        count = 0
        for index, row in df.iterrows():
            items_in_transaction = set(row['Items'].split(','))
            if set(itemset).issubset(items_in_transaction):
                count += 1
        return count / len(df)

    def fit(self, df):
        # 1. Tạo tập ứng viên C1 (1-itemset)
        all_items = set()
        for items in df['Items']:
            all_items.update(items.split(','))
        
        # Tìm L1 (Tập phổ biến 1 phần tử)
        l1 = {}
        for item in all_items:
            supp = self._get_support(df, [item])
            if supp >= self.min_support:
                l1[frozenset([item])] = supp
        
        self.itemsets.update(l1)
        current_l = l1

        # 2. Vòng lặp tìm k-itemset (k=2, 3...)
        k = 2
        while True:
            # Tạo Ck từ L(k-1)
            candidates = set()
            l_keys = list(current_l.keys())
            for i in range(len(l_keys)):
                for j in range(i + 1, len(l_keys)):
                    # Join logic: hợp 2 tập lại
                    union = l_keys[i].union(l_keys[j])
                    if len(union) == k:
                        candidates.add(union)
            
            # Lọc Ck để lấy Lk (thỏa mãn min_support)
            lk = {}
            for cand in candidates:
                supp = self._get_support(df, cand)
                if supp >= self.min_support:
                    lk[cand] = supp
            
            if not lk:
                break
                
            self.itemsets.update(lk)
            current_l = lk
            k += 1

    def generate_rules(self):
        # Sinh luật từ các tập phổ biến
        for itemset, support in self.itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Thử tất cả các tập con làm vế trái (Left Hand Side)
            for i in range(1, len(itemset)):
                for lhs in combinations(itemset, i):
                    lhs = frozenset(lhs)
                    rhs = itemset - lhs
                    
                    # Tính confidence
                    supp_lhs = self.itemsets[lhs]
                    conf = support / supp_lhs
                    
                    if conf >= self.min_confidence:
                        self.rules.append({
                            'Vế trái (A)': ', '.join(lhs),
                            'Vế phải (B)': ', '.join(rhs),
                            'Support': round(support, 4),
                            'Confidence': round(conf, 4)
                        })
        return pd.DataFrame(self.rules)