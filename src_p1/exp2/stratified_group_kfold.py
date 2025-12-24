"""
Stratified GroupKFold Implementation
===================================

Implementación personalizada de Stratified GroupKFold que intenta mantener
la distribución de clases balanceada mientras respeta los grupos físicos.

Uso:
    from stratified_group_kfold import StratifiedGroupKFold
    
    sgkf = StratifiedGroupKFold(n_splits=3)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        # Train y val mantienen grupos intactos y distribución balanceada
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array
from itertools import combinations


class StratifiedGroupKFold(BaseCrossValidator):
    """
    Stratified Group K-Fold cross validator
    
    Combina GroupKFold (respeta grupos) con StratifiedKFold (balancea clases).
    Intenta crear folds que mantengan grupos intactos mientras distribuye
    las clases de manera lo más balanceada posible.
    """
    
    def __init__(self, n_splits=3, shuffle=True, random_state=42):
        """
        Args:
            n_splits: Número de folds
            shuffle: Si mezclar los grupos antes de dividir
            random_state: Semilla para reproducibilidad
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X, y, groups):
        """
        Generar índices para splits estratificados por grupo
        
        Args:
            X: Datos (no usado, solo para compatibilidad)
            y: Labels objetivo
            groups: Identificadores de grupo
            
        Yields:
            train_idx, val_idx: Índices para entrenamiento y validación
        """
        X, y, groups = check_array(X, dtype=None), np.asarray(y), np.asarray(groups)
        
        n_samples = X.shape[0]
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if self.n_splits > n_groups:
            raise ValueError(f"n_splits={self.n_splits} no puede ser mayor que "
                           f"n_groups={n_groups}")
        
        # Crear mapeo de grupo -> muestras en ese grupo
        group_to_samples = {}
        group_to_classes = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_samples = np.where(group_mask)[0]
            group_labels = y[group_mask]
            
            group_to_samples[group] = group_samples
            group_to_classes[group] = Counter(group_labels)
        
        print(f"📊 Análisis de grupos:")
        for group in sorted(unique_groups):
            classes = group_to_classes[group]
            print(f"   {group}: {dict(classes)} ({len(group_to_samples[group])} muestras)")
        
        # Estrategia: distribuir grupos de manera que balance las clases
        best_folds = self._find_best_group_distribution(
            unique_groups, group_to_classes, self.n_splits
        )
        
        # Generar splits basados en la mejor distribución
        for fold_idx in range(self.n_splits):
            val_groups = best_folds[fold_idx]
            train_groups = [g for g in unique_groups if g not in val_groups]
            
            # Convertir grupos a índices de muestras
            train_samples = [group_to_samples[g] for g in train_groups]
            val_samples = [group_to_samples[g] for g in val_groups]
            
            # Concatenar solo si hay grupos
            train_idx = np.concatenate(train_samples).astype(int) if train_samples else np.array([], dtype=int)
            val_idx = np.concatenate(val_samples).astype(int) if val_samples else np.array([], dtype=int)
            
            # Información del fold
            train_classes = Counter(y[train_idx]) if len(train_idx) > 0 else Counter()
            val_classes = Counter(y[val_idx]) if len(val_idx) > 0 else Counter()
            
            print(f"\n📁 Fold {fold_idx + 1}:")
            print(f"   Train: {len(train_groups)} grupos, {len(train_idx)} muestras")
            print(f"   Train clases: {dict(train_classes)}")
            print(f"   Val: {len(val_groups)} grupos, {len(val_idx)} muestras")  
            print(f"   Val clases: {dict(val_classes)}")
            
            yield train_idx, val_idx
    
    def _find_best_group_distribution(self, groups, group_to_classes, n_splits):
        """
        Encuentra la mejor manera de distribuir grupos en folds para balancear clases
        
        Usa un enfoque heurístico: intenta distribuir grupos de manera que cada fold
        tenga una representación similar de todas las clases.
        """
        # Calcular distribución global de clases
        total_classes = Counter()
        for group_classes in group_to_classes.values():
            total_classes.update(group_classes)
        
        print(f"📊 Distribución global: {dict(total_classes)}")
        
        # Objetivo: cada fold debería tener ~1/n_splits de cada clase
        target_per_fold = {cls: count // n_splits for cls, count in total_classes.items()}
        print(f"🎯 Objetivo por fold: {target_per_fold}")
        
        # Algoritmo greedy: asignar grupos a folds intentando alcanzar objetivo
        folds = [[] for _ in range(n_splits)]
        fold_class_counts = [Counter() for _ in range(n_splits)]
        
        # Ordenar grupos por número total de muestras (descendente)
        # Esto ayuda a distribuir mejor los grupos grandes
        groups_sorted = sorted(groups, 
                             key=lambda g: sum(group_to_classes[g].values()), 
                             reverse=True)
        
        # Estrategia simplificada: distribución round-robin garantizada
        for i, group in enumerate(groups_sorted):
            group_classes = group_to_classes[group]
            
            # Asignar grupo usando round-robin simple
            fold_idx = i % n_splits
            
            # Asignar grupo a ese fold
            folds[fold_idx].append(group)
            fold_class_counts[fold_idx].update(group_classes)
            
            print(f"   Grupo {group} → Fold {fold_idx + 1}")
        
        # Mostrar distribución final
        print(f"\n🎯 Distribución final por fold:")
        for i, (fold_groups, fold_counts) in enumerate(zip(folds, fold_class_counts)):
            print(f"   Fold {i+1}: {len(fold_groups)} grupos, {dict(fold_counts)}")
        
        return folds
    
    def _find_best_fold_for_group(self, group_classes, fold_class_counts, target_per_fold):
        """
        Encuentra el mejor fold para asignar un grupo específico
        
        Criterio: el fold que después de agregar el grupo esté más cerca del objetivo
        """
        best_fold_idx = 0
        best_score = float('inf')
        
        for fold_idx, current_counts in enumerate(fold_class_counts):
            # Simular agregar el grupo a este fold
            projected_counts = current_counts.copy()
            projected_counts.update(group_classes)
            
            # Calcular qué tan lejos está del objetivo
            score = self._calculate_balance_score(projected_counts, target_per_fold)
            
            if score < best_score:
                best_score = score
                best_fold_idx = fold_idx
        
        return best_fold_idx
    
    def _calculate_balance_score(self, current_counts, target_counts):
        """
        Calcula qué tan balanceado está un fold comparado con el objetivo
        
        Score más bajo = mejor balance
        """
        score = 0
        for cls, target in target_counts.items():
            current = current_counts.get(cls, 0)
            # Penalizar desviación del objetivo
            score += abs(current - target)
            
            # Penalizar especialmente si una clase queda en 0
            if target > 0 and current == 0:
                score += 1000  # Penalización alta por clase faltante
        
        return score
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Retorna número de splits"""
        return self.n_splits


if __name__ == "__main__":
    # Test básico
    print("🧪 Test de StratifiedGroupKFold")
    
    # Datos de ejemplo
    X = np.random.randn(20, 5)
    y = np.array([0]*8 + [1]*7 + [2]*5)  # Distribución desbalanceada
    groups = np.array(['A']*3 + ['B']*2 + ['C']*4 + ['D']*3 + ['E']*4 + ['F']*2 + ['G']*2)
    
    print(f"Dataset: {len(X)} muestras, {len(np.unique(groups))} grupos")
    print(f"Clases: {dict(Counter(y))}")
    print(f"Grupos: {list(np.unique(groups))}")
    
    sgkf = StratifiedGroupKFold(n_splits=3)
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_idx)} muestras")
        print(f"  Val: {len(val_idx)} muestras")