"""
Métricas y Visualizaciones Estándar para Experimentos
===================================================

Funciones reutilizables para evaluación consistente entre todos los experimentos.
Incluye métricas de clasificación, detección de data leakage y visualizaciones.

Uso:
    from utils.experiment_metrics import ExperimentEvaluator
    
    evaluator = ExperimentEvaluator(results_dir="exp1/results", show_plots=False)
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
    evaluator.plot_training_curves(train_history)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

class ExperimentEvaluator:
    """
    Evaluador estándar para experimentos de clasificación multiclase.
    Genera métricas consistentes y visualizaciones comparables.
    """
    
    def __init__(self, results_dir, show_plots=False, experiment_name=""):
        """
        Inicializar evaluador
        
        Args:
            results_dir: Directorio para guardar plots y reportes
            show_plots: Si True, muestra plots en pantalla (para notebooks)
            experiment_name: Nombre del experimento para archivos
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.show_plots = show_plots
        self.experiment_name = experiment_name
        
        # Configurar estilo matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
    def evaluate_classification(self, y_true, y_pred, y_proba=None, 
                               class_names=None, split_name="test"):
        """
        Evalúa rendimiento de clasificación con métricas estándar
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones del modelo  
            y_proba: Probabilidades de clase (opcional, para ROC)
            class_names: Nombres de las clases (default: N0, N1, N2, N3)
            split_name: Nombre del split evaluado (test, val, etc.)
            
        Returns:
            dict: Diccionario con todas las métricas
        """
        if class_names is None:
            class_names = [f"N{i}" for i in sorted(np.unique(y_true))]
            
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Métricas agregadas
        f1_macro = np.mean(f1)
        f1_weighted = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]
        
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # ROC-AUC (si se proporcionan probabilidades)
        auc_scores = {}
        if y_proba is not None:
            try:
                # One-vs-Rest para multiclase
                y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
                if len(class_names) == 2:
                    y_true_bin = y_true_bin.ravel()
                    auc_scores['overall'] = roc_auc_score(y_true_bin, y_proba[:, 1])
                else:
                    auc_scores['overall'] = roc_auc_score(y_true_bin, y_proba, 
                                                        multi_class='ovr', average='macro')
                    # AUC por clase
                    for i, class_name in enumerate(class_names):
                        auc_scores[class_name] = roc_auc_score(
                            y_true_bin[:, i], y_proba[:, i]
                        )
            except Exception as e:
                print(f"Warning: No se pudo calcular AUC: {e}")
                auc_scores['overall'] = 0.0
        
        # Organizar métricas
        metrics = {
            'split': split_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cohen_kappa': kappa,
            'auc_macro': auc_scores.get('overall', 0.0),
            'per_class': {}
        }
        
        # Métricas por clase
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i],
                'auc': auc_scores.get(class_name, 0.0)
            }
        
        # Generar reporte detallado
        self._save_classification_report(metrics, y_true, y_pred, class_names, split_name)
        
        return metrics
    
    def detect_data_leakage(self, train_acc, val_acc, test_acc, cv_scores=None):
        """
        Detecta posible data leakage usando métricas heurísticas
        
        Args:
            train_acc: Accuracy en entrenamiento
            val_acc: Accuracy en validación  
            test_acc: Accuracy en test
            cv_scores: Scores de cross-validation (opcional)
            
        Returns:
            dict: Alertas y métricas de data leakage
        """
        alerts = []
        metrics = {}
        
        # Test 1: Gap entre test y validation
        test_val_gap = test_acc - val_acc
        metrics['test_val_gap'] = test_val_gap
        if test_val_gap > 0.05:  # 5% threshold
            alerts.append(f"⚠️ Test accuracy demasiado alta vs validation ({test_val_gap:.3f})")
        
        # Test 2: Gap entre test y training  
        test_train_gap = abs(test_acc - train_acc)
        metrics['test_train_gap'] = test_train_gap
        if test_train_gap < 0.02:  # <2% gap es sospechoso
            alerts.append(f"⚠️ Test accuracy muy cerca de train accuracy ({test_train_gap:.3f})")
        
        # Test 3: Validation muy superior a training (imposible normalmente)
        if val_acc > train_acc + 0.03:
            alerts.append(f"🚨 Validation accuracy > Train accuracy ({val_acc:.3f} > {train_acc:.3f})")
        
        # Test 4: Varianza de CV muy baja (si disponible)
        if cv_scores is not None:
            cv_std = np.std(cv_scores)
            cv_mean = np.mean(cv_scores)
            metrics['cv_std'] = cv_std
            metrics['cv_mean'] = cv_mean
            metrics['cv_coefficient_variation'] = cv_std / cv_mean if cv_mean > 0 else 0
            
            if cv_std < 0.01:  # Varianza muy baja
                alerts.append(f"⚠️ Varianza de CV sospechosamente baja ({cv_std:.4f})")
        
        # Test 5: Accuracy general muy alta (>98% es sospechoso)
        if test_acc > 0.98:
            alerts.append(f"🚨 Accuracy sospechosamente alta ({test_acc:.3f})")
        
        leakage_report = {
            'alerts': alerts,
            'metrics': metrics,
            'risk_level': self._assess_leakage_risk(len(alerts), metrics)
        }
        
        # Guardar reporte
        self._save_leakage_report(leakage_report)
        
        return leakage_report
    
    def plot_training_curves(self, history, metrics=['loss', 'accuracy']):
        """
        Plotea curvas de entrenamiento y validación
        
        Args:
            history: Diccionario con historial de entrenamiento
                    {'train_loss': [...], 'val_loss': [...], etc.}
            metrics: Lista de métricas a plotear
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Obtener datos del historial
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history and val_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                
                ax.plot(epochs, history[train_key], 'b-', label=f'Training {metric}', linewidth=2)
                ax.plot(epochs, history[val_key], 'r-', label=f'Validation {metric}', linewidth=2)
                
                ax.set_title(f'{metric.capitalize()} Curves', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epochs', fontsize=12)
                ax.set_ylabel(metric.capitalize(), fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Marcar mejor epoch si es accuracy
                if metric == 'accuracy':
                    best_epoch = np.argmax(history[val_key])
                    ax.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7, 
                             label=f'Best epoch: {best_epoch+1}')
                    ax.legend()
        
        plt.tight_layout()
        
        # Guardar y mostrar
        filename = f"{self.experiment_name}_training_curves.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, normalize=True):
        """
        Plotea matriz de confusión
        """
        if class_names is None:
            class_names = [f"N{i}" for i in sorted(np.unique(y_true))]
            
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Guardar y mostrar
        filename = f"{self.experiment_name}_confusion_matrix.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_classification_metrics(self, metrics, class_names=None):
        """
        Plotea métricas por clase (Precision, Recall, F1-Score)
        """
        if class_names is None:
            class_names = list(metrics['per_class'].keys())
            
        # Extraer métricas
        precisions = [metrics['per_class'][cls]['precision'] for cls in class_names]
        recalls = [metrics['per_class'][cls]['recall'] for cls in class_names] 
        f1_scores = [metrics['per_class'][cls]['f1_score'] for cls in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)  
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Añadir valores en las barras
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2) 
        autolabel(bars3)
        
        plt.tight_layout()
        
        # Guardar y mostrar
        filename = f"{self.experiment_name}_classification_metrics.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_roc_curves(self, y_true, y_proba, class_names=None):
        """
        Plotea curvas ROC para clasificación multiclase (One-vs-Rest)
        """
        if class_names is None:
            class_names = [f"N{i}" for i in sorted(np.unique(y_true))]
            
        if y_proba is None:
            print("Warning: No se proporcionaron probabilidades para ROC curves")
            return
            
        # Binarizar labels para One-vs-Rest
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        plt.figure(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(class_names))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if len(class_names) == 2:
                # Caso binario
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba[:, 1])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{class_name} (AUC = {auc_score:.3f})')
                break
            else:
                # Caso multiclase
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{class_name} (AUC = {auc_score:.3f})')
        
        # Línea diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Guardar y mostrar
        filename = f"{self.experiment_name}_roc_curves.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def generate_experiment_summary(self, metrics_dict, leakage_report=None):
        """
        Genera resumen completo del experimento
        
        Args:
            metrics_dict: Dict con métricas de diferentes splits
            leakage_report: Reporte de data leakage (opcional)
        """
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics_dict,
            'data_leakage': leakage_report,
            'plots_generated': [
                f"{self.experiment_name}_training_curves.png",
                f"{self.experiment_name}_confusion_matrix.png", 
                f"{self.experiment_name}_classification_metrics.png",
                f"{self.experiment_name}_roc_curves.png"
            ]
        }
        
        # Guardar como JSON
        import json
        summary_file = self.results_dir / f"{self.experiment_name}_experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Guardar como texto legible
        self._save_readable_summary(summary)
        
        return summary
    
    def _save_classification_report(self, metrics, y_true, y_pred, class_names, split_name):
        """Guarda reporte detallado de clasificación"""
        report_file = self.results_dir / f"{self.experiment_name}_{split_name}_classification_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"CLASSIFICATION REPORT - {split_name.upper()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n") 
            f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
            
            # Métricas generales
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
            f.write(f"AUC (macro): {metrics['auc_macro']:.4f}\n\n")
            
            # Métricas por clase
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'AUC':<10}\n")
            f.write("-" * 60 + "\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"{class_name:<8} {class_metrics['precision']:<10.4f} "
                       f"{class_metrics['recall']:<10.4f} {class_metrics['f1_score']:<10.4f} "
                       f"{class_metrics['support']:<10} {class_metrics['auc']:<10.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("SKLEARN CLASSIFICATION REPORT:\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    def _save_leakage_report(self, leakage_report):
        """Guarda reporte de data leakage"""
        report_file = self.results_dir / f"{self.experiment_name}_data_leakage_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DATA LEAKAGE DETECTION REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Risk Level: {leakage_report['risk_level']}\n\n")
            
            if leakage_report['alerts']:
                f.write("⚠️ ALERTS DETECTED:\n")
                f.write("-" * 20 + "\n")
                for alert in leakage_report['alerts']:
                    f.write(f"{alert}\n")
            else:
                f.write("✅ No data leakage alerts detected\n")
            
            f.write(f"\nMETRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in leakage_report['metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
    
    def _save_readable_summary(self, summary):
        """Guarda resumen legible del experimento"""
        summary_file = self.results_dir / f"{self.experiment_name}_experiment_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Experiment: {summary['experiment_name']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            # Métricas por split
            for split_name, metrics in summary['metrics'].items():
                f.write(f"{split_name.upper()} METRICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n")
                f.write(f"AUC (macro): {metrics['auc_macro']:.4f}\n\n")
            
            # Data leakage
            if summary['data_leakage']:
                f.write("DATA LEAKAGE STATUS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Risk Level: {summary['data_leakage']['risk_level']}\n")
                f.write(f"Alerts: {len(summary['data_leakage']['alerts'])}\n\n")
            
            f.write("GENERATED FILES:\n")
            f.write("-" * 20 + "\n")
            for plot_file in summary['plots_generated']:
                f.write(f"📊 {plot_file}\n")
    
    def _assess_leakage_risk(self, num_alerts, metrics):
        """Evalúa nivel de riesgo de data leakage"""
        if num_alerts == 0:
            return "LOW"
        elif num_alerts <= 2:
            return "MEDIUM" 
        else:
            return "HIGH"