import pandas as pd
import numpy as np
from itertools import combinations

class RoughSets:
    def __init__(self):
        self.reducts = []
        self.core = []
        self.dependency = 0.0

    def get_indiscernibility(self, df, attributes):
        if not attributes:
            return {(): df.index.tolist()}
        return df.groupby(attributes).groups

    def get_lower_approximation(self, df, conditional_attrs, decision_attr):
        c_groups = self.get_indiscernibility(df, conditional_attrs)
        d_groups = self.get_indiscernibility(df, [decision_attr])
        d_sets = [set(indices) for indices in d_groups.values()]
        
        pos_region = set()
        for indices in c_groups.values():
            current_set = set(indices)
            is_subset = False
            for d_set in d_sets:
                if current_set.issubset(d_set):
                    is_subset = True
                    break
            if is_subset:
                pos_region.update(current_set)
        return pos_region

    def calculate_dependency(self, df, conditional_attrs, decision_attr):
        pos_region = self.get_lower_approximation(df, conditional_attrs, decision_attr)
        return len(pos_region) / len(df)

    def fit(self, df, target_col, id_col=None):
        data = df.copy()
        if id_col and id_col in data.columns:
            data = data.drop(columns=[id_col])
            
        decision_attr = target_col
        conditional_attrs = [c for c in data.columns if c != decision_attr]
        
        # 1. Tính độ phụ thuộc gốc
        full_dependency = self.calculate_dependency(data, conditional_attrs, decision_attr)
        self.dependency = full_dependency
        
        # 2. Tìm Reducts
        self.reducts = []
        for r in range(1, len(conditional_attrs) + 1):
            for subset in combinations(conditional_attrs, r):
                subset = list(subset)
                
                # Kiểm tra tính tối thiểu (Minimal)
                is_superset = False
                for existing_reduct in self.reducts:
                    if set(existing_reduct).issubset(set(subset)):
                        is_superset = True
                        break
                if is_superset: continue
                
                # Kiểm tra độ phụ thuộc
                dep = self.calculate_dependency(data, subset, decision_attr)
                if abs(dep - full_dependency) < 1e-9:
                    self.reducts.append(subset)

        # 3. Tìm Core
        if self.reducts:
            core_set = set(self.reducts[0])
            for reduct in self.reducts[1:]:
                core_set = core_set.intersection(set(reduct))
            self.core = list(core_set)
        else:
            self.core = []

        return self.reducts, self.core

    # --- HÀM SINH LUẬT ---
    def get_rules(self, df, target_col, id_col=None):
        """Sinh luật từ các Reducts đã tìm được"""
        if not self.reducts:
            return pd.DataFrame()

        data = df.copy()
        if id_col and id_col in data.columns:
            data = data.drop(columns=[id_col])

        all_rules = []
        
        # Duyệt qua từng Reduct để sinh luật
        for i, reduct in enumerate(self.reducts):
            cols = list(reduct) + [target_col]
            
            # Lấy các mẫu duy nhất theo các cột trong Reduct
            unique_patterns = data[cols].drop_duplicates()
            
            for _, row in unique_patterns.iterrows():
                conditions = [f"{col}='{row[col]}'" for col in reduct]
                cond_str = " AND ".join(conditions)
                
                decision = row[target_col]
                
                all_rules.append({
                    "Nguồn": f"Reduct #{i+1}",
                    "Điều kiện (IF)": cond_str,
                    "Quyết định (THEN)": decision
                })
                
        return pd.DataFrame(all_rules)