#!/usr/bin/env python3
"""
Configuración centralizada de colores y estilos para todos los plots de la tesis
==============================================================================

Este archivo define una paleta de colores consistente y estilos de matplotlib
para garantizar coherencia visual en toda la documentación de la tesis.

Uso:
    from src.utils.plot_config import ThesisColors, ThesisStyles, setup_plot_style
    
    # Configurar estilo global
    setup_plot_style()
    
    # Usar colores para clases de daño
    colors = ThesisColors.damage_classes['N1']
    
    # Usar colores para experimentos
    exp_color = ThesisColors.experiments['exp2']
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ThesisColors:
    """Paleta de colores estandarizada para la tesis"""
    
    # Colores para clases de daño (N1, N2, N3)
    # Paleta coherente y distinguible para daltónicos
    damage_classes = {
        'N1': '#2E4057',  # Azul oscuro - Sin daño
        'N2': '#048A81',  # Verde azulado - Daño moderado  
        'N3': '#54C6EB'   # Azul claro - Daño severo
    }
    
    # Colores para experimentos
    experiments = {
        'exp1': '#DC143C',  # Rojo - Experimento inválido (para cuando se muestre)
        'exp2': '#FF9999',  # Rosa suave - FFT Baseline
        'exp3': '#90EE90',  # Verde suave - FFT Balanceado
        'exp4': '#D4AF8C'   # Dorado - Features Agregadas
    }
    
    # Colores para métricas de entrenamiento
    training = {
        'train': '#FF9999',      # Rosa suave - Entrenamiento
        'validation': '#90EE90', # Verde suave - Validación
        'test': '#D4AF8C'        # Dorado - Test
    }
    
    # Colores para comparaciones (original vs procesado)
    comparison = {
        'original': '#2E4057',   # Azul oscuro
        'processed': '#048A81',  # Verde azulado
        'augmented': '#54C6EB'   # Azul claro
    }
    
    # Colores de estado para validaciones
    status = {
        'success': '#28a745',    # Verde éxito
        'warning': '#ffc107',    # Amarillo advertencia
        'error': '#dc3545',      # Rojo error
        'info': '#17a2b8'        # Azul información
    }
    
    @classmethod
    def get_damage_class_list(cls):
        """Obtener lista de colores en orden N1, N2, N3"""
        return [cls.damage_classes['N1'], cls.damage_classes['N2'], cls.damage_classes['N3']]
    
    @classmethod
    def get_experiment_list(cls, experiments=['exp2', 'exp3', 'exp4']):
        """Obtener lista de colores para experimentos especificados"""
        return [cls.experiments[exp] for exp in experiments]
    
    @classmethod
    def get_roc_colors(cls):
        """Colores para curvas ROC (uno por clase)"""
        return cls.get_damage_class_list()

class ThesisStyles:
    """Estilos estandarizados para plots"""
    
    # Configuración base de matplotlib
    mpl_config = {
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    
    # Tamaños de figura estándar
    figure_sizes = {
        'single': (8, 6),      # Plot individual
        'double': (12, 6),     # Dos plots lado a lado
        'triple': (15, 5),     # Tres plots en fila
        'quad': (12, 10),      # Cuatro plots en cuadrícula
        'wide': (14, 6),       # Plot panorámico
        'tall': (8, 10),       # Plot alto
        'comparison': (15, 8), # Plots de comparación
    }
    
    # Configuraciones específicas para tipos de plot
    plot_configs = {
        'confusion_matrix': {
            'cmap': 'Blues',
            'annot': True,
            'fmt': '.2f',
            'cbar': True,
            'square': True
        },
        'training_history': {
            'alpha': 0.8,
            'linewidth': 2,
            'grid_alpha': 0.3
        },
        'roc_curve': {
            'linewidth': 2,
            'alpha': 0.8,
            'linestyle': '-'
        },
        'bar_plot': {
            'alpha': 0.8,
            'edgecolor': 'black',
            'linewidth': 0.5,
            'capsize': 5
        },
        'scatter_plot': {
            'alpha': 0.7,
            's': 50,
            'edgecolors': 'black',
            'linewidth': 0.5
        }
    }

def setup_plot_style():
    """Configurar estilo global de matplotlib para toda la tesis"""
    
    # Configurar matplotlib
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update(ThesisStyles.mpl_config)
    
    # Configurar seaborn con paleta personalizada
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.1)
    
    print("✓ Estilo de plots configurado para tesis")

def get_figure_size(plot_type='single'):
    """Obtener tamaño de figura estándar"""
    return ThesisStyles.figure_sizes.get(plot_type, ThesisStyles.figure_sizes['single'])

def create_figure(plot_type='single', **kwargs):
    """Crear figura con configuración estándar"""
    figsize = get_figure_size(plot_type)
    return plt.figure(figsize=figsize, **kwargs)

def save_figure(fig, filepath, **kwargs):
    """Guardar figura con configuración estándar"""
    default_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    default_kwargs.update(kwargs)
    fig.savefig(filepath, **default_kwargs)
    print(f"   ✓ Plot guardado: {filepath}")

# Funciones de conveniencia
def get_damage_colors():
    """Función de conveniencia para obtener colores de clases de daño"""
    return ThesisColors.damage_classes

def get_experiment_colors():
    """Función de conveniencia para obtener colores de experimentos"""
    return ThesisColors.experiments

def get_training_colors():
    """Función de conveniencia para obtener colores de entrenamiento"""
    return ThesisColors.training

# Configurar automáticamente al importar
setup_plot_style()

if __name__ == "__main__":
    # Test de los colores
    import matplotlib.pyplot as plt
    
    print("Testing ThesisColors...")
    
    # Test damage classes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Damage classes
    ax = axes[0, 0]
    classes = ['N1', 'N2', 'N3']
    colors = ThesisColors.get_damage_class_list()
    ax.bar(classes, [1, 1, 1], color=colors)
    ax.set_title('Damage Classes Colors')
    
    # Experiments
    ax = axes[0, 1]
    exps = ['exp2', 'exp3', 'exp4']
    colors = ThesisColors.get_experiment_list()
    ax.bar(exps, [1, 1, 1], color=colors)
    ax.set_title('Experiment Colors')
    
    # Training
    ax = axes[1, 0]
    phases = list(ThesisColors.training.keys())
    colors = list(ThesisColors.training.values())
    ax.bar(phases, [1, 1, 1], color=colors)
    ax.set_title('Training Phase Colors')
    
    # Status
    ax = axes[1, 1]
    statuses = list(ThesisColors.status.keys())
    colors = list(ThesisColors.status.values())
    ax.bar(statuses, [1, 1, 1], color=colors)
    ax.set_title('Status Colors')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Color test completed")