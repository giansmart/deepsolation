# 🏗️ DeepIsolation - Clasificación de Daño en Aisladores Sísmicos

## 📋 Descripción del Proyecto

Este proyecto de tesis desarrolla un modelo de machine learning para **clasificar el nivel de daño** en aisladores sísmicos mediante análisis de señales de vibración. El sistema permite predecir automáticamente el estado estructural de aisladores basándose en mediciones de aceleración en múltiples ejes.


## 📁 Estructura del Proyecto

```
deepsolation/
│
├── data/                              # Datos y preprocesamiento
│   ├── raw/                          # Datos originales (no modificar)
│   │   └── isolators/                # 71 mediciones originales
│   ├── processed/                    # Datos procesados y estandarizados
│   │   ├── stage1_autoencoder/      # 71 mediciones para autoencoder
│   │   └── stage2_classifier/       # 51 aisladores únicos para CNN
│   ├── augmented/                    # Datos aumentados (N1×1, N2×6, N3×21)
│   └── features/                     # Features relacionales H(ω) pre-calculadas
│       └── relational_features.csv   # 18 características por aislador
│
├── src/
│   ├── preprocessing/                # Módulos de preprocesamiento
│   │   ├── __init__.py
│   │   ├── loader.py                # Cargar señales RAW
│   │   ├── standardizer.py          # Estandarizar longitudes (→60k)
│   │   ├── augmentation.py          # Time-shift, noise, scaling
│   │   └── relational_features.py   # Calcular 18 features H(ω)
│   │
│   ├── models/                       # Arquitecturas de redes
│   │   ├── __init__.py
│   │   ├── autoencoder.py           # ETAPA 1: Autoencoder
│   │   ├── cnn_classifier.py        # ETAPA 2: CNN Classifier
│   │   └── combined_model.py        # Modelo completo (encoder + classifier)
│   │
│   ├── training/                     # Lógica de entrenamiento
│   │   ├── __init__.py
│   │   ├── train_autoencoder.py     # Entrenar ETAPA 1
│   │   ├── train_classifier.py      # Entrenar ETAPA 2 (Phase A + B)
│   │   └── trainer_utils.py         # EarlyStopping, Checkpoints, etc.
│   │
│   ├── validation/                   # Validación y evaluación
│   │   ├── __init__.py
│   │   ├── cross_validation.py      # GroupKFold CV
│   │   ├── metrics.py               # Accuracy, F1, Kappa, AUC, etc.
│   │   └── visualizations.py        # Confusion matrix, PCA, t-SNE
│   │
│   ├── utils/                        # Utilidades generales (ya existe)
│   │   ├── __init__.py
│   │   ├── config.py                # Configuraciones globales
│   │   ├── logger.py                # Logging customizado
│   │   └── data_utils.py            # Helpers para manejo de datos
│   │
│   └── notebooks/                    # Notebooks experimentales (ya existe)
│       ├── 0_data_exploration.ipynb              # EDA inicial
│       ├── 1_preprocessing_pipeline.ipynb        # Pipeline completo
│       ├── 2_clustering_fft_kmeans.ipynb         # (ya existe)
│       ├── 3_stage1_autoencoder_training.ipynb   # ETAPA 1
│       ├── 4_stage2_classifier_training.ipynb    # ETAPA 2
│       ├── 5_full_pipeline_evaluation.ipynb      # Evaluación final
│       └── 6_results_analysis.ipynb              # Análisis y visualizaciones
│
│
├── results/                          # Resultados finales consolidados
│   ├── metrics/                     # CSVs con métricas por fold
│   ├── figures/                     # Gráficos para tesis
│   └── reports/                     # Reportes en Markdown/PDF
│
├── configs/                          # Archivos de configuración
│   ├── autoencoder_config.yaml      # Hiperparámetros ETAPA 1
│   ├── classifier_config.yaml       # Hiperparámetros ETAPA 2
│   └── augmentation_config.yaml     # Parámetros de augmentación
│
├── scripts/                          # Scripts ejecutables
│   ├── run_stage1.py               # Ejecutar ETAPA 1 completa
│   ├── run_stage2.py               # Ejecutar ETAPA 2 completa
│   ├── run_full_pipeline.py        # Pipeline end-to-end
│   └── evaluate_model.py           # Evaluación sobre test set
│
├── tests/                            # Tests unitarios (opcional)
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_augmentation.py
│
├── requirements.txt                  # Dependencias del proyecto
├── setup.py                         # Para instalación como paquete
├── README.md                        # Documentación del proyecto
└── PROPUESTA_CNN_ARQUITECTURA.md   # (ya existe) Propuesta arquitectural
```

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### 🚀 **Ejecución del Demo**

#### **Paso 1: Instalación**
```bash
pip install -r requirements.txt
cd notebooks
```

#### **Paso 2: Ejecución**
```bash
python demo_signals.py
```



## 🤝 **Contribuciones**

Este proyecto forma parte de una tesis de maestría enfocada en la aplicación de machine learning para el monitoreo estructural de infraestructura sísmica.

---

**Autor**: Giancarlo Poémape Lozano
**Institución**: UTEC - Universidad de Ingeniería y Tecnología
**Año**: 2026