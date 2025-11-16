# Experimento 4: Enfoque Metodológicamente Correcto

## Resumen Ejecutivo

El Experimento 4 implementa el enfoque **metodológicamente correcto** recomendado por el experto para resolver los problemas fundamentales de pseudo-replicación presentes en los Experimentos 1-3.

## Problema Fundamental en Exp1-3

### ❌ Enfoque Incorrecto (Exp1-3)
- **Bins de FFT como observaciones independientes**
- Miles de "observaciones" del mismo dispositivo físico
- **Pseudo-replicación severa**
- Unidad de observación ≠ Unidad de inferencia
- Métricas artificialmente infladas
- **Metodológicamente inválido**

### ✅ Solución Correcta (Exp4)
- **Un dispositivo = Una observación**
- Características estadísticas agregadas
- Una etiqueta por dispositivo físico
- Alineación correcta de unidades
- Métricas científicamente válidas
- **Metodológicamente correcto**

## Metodología Exp4

### Enfoque de Procesamiento
1. **Análisis Temporal por Ventanas Deslizantes**
   - Ventanas de 1000 samples (10s @ 100Hz)
   - Overlap del 50%
   - Estadísticos robustos por ventana

2. **Características Espectrales Agregadas**
   - Centroide espectral
   - Dispersión espectral
   - Rolloff espectral
   - Entropía espectral
   - Energía por bandas de frecuencia

3. **Agregación a Nivel de Dispositivo**
   - Estadísticos de estadísticos
   - Características globales del dispositivo
   - Sin bins individuales

### Estructura de Datos Correcta

```
Exp1-3 (INCORRECTO):
- 1 dispositivo → Miles de observaciones (bins FFT)
- Pseudo-replicación masiva

Exp4 (CORRECTO):
- 1 dispositivo → 1 observación
- Características agregadas
```

## Resultados Esperados

### Dataset Final
- **~36 observaciones** (una por dispositivo físico)
- **~200+ características** agregadas por dispositivo
- **Distribución natural** de clases de daño
- **Metodológicamente válido** para ML

### Ventajas Metodológicas
1. **Validez Estadística**: Sin pseudo-replicación
2. **Interpretabilidad**: Características interpretables
3. **Aplicabilidad**: Resultados transferibles a la práctica
4. **Rigor Científico**: Metodología estadísticamente correcta

## Archivos del Experimento

### `1_preprocess_signals.py`
Script principal de preprocesamiento que implementa el enfoque metodológicamente correcto.

**Características:**
- Procesamiento por dispositivo completo
- Extracción de características agregadas
- Export de dataset científicamente válido

### `2_train_dcnn.py` 
Script de entrenamiento del modelo DNN con características agregadas.

**Características:**
- Arquitectura Fully Connected DNN (no convolucional)
- GroupKFold por specimen físico 
- Métricas consistentes con experimentos anteriores
- Visualizaciones con colores de thesis
- Una observación = Un dispositivo físico

### `exp4_signal_preprocessing.py`
Módulo de preprocesamiento especializado para el enfoque correcto.

**Componentes principales:**
- `extract_windowed_statistics()`: Análisis temporal por ventanas
- `extract_spectral_statistics()`: Características espectrales agregadas
- `preprocess_device_signal()`: Pipeline completo por dispositivo

### `exp4_model.py`
Modelo DNN especializado para características estadísticas agregadas.

**Componentes principales:**
- `Exp4DamageNet`: Arquitectura Fully Connected 
- `Exp4Trainer`: Entrenador especializado
- Entrada: 303 características estadísticas
- Salida: 3 clases de daño (N1, N2, N3)

## Uso

```bash
# 1. Preprocesamiento metodológicamente correcto
cd deepsolation/
python src/exp4/1_preprocess_signals.py

# Con parámetros personalizados
python src/exp4/1_preprocess_signals.py --window-size 2000 --overlap 0.3

# 2. Entrenamiento del modelo DNN
python src/exp4/2_train_dcnn.py

# Con configuración personalizada
python src/exp4/2_train_dcnn.py --epochs 100 --batch-size 8 --learning-rate 0.0005
```

## Comparación con Experimentos Anteriores

| Aspecto | Exp1 (Incorrecto) | Exp2/Exp3 (Correctos) | Exp4 (Correcto) |
|---------|-------------------|------------------------|------------------|
| Unidad de observación | Bin de frecuencia | Dispositivo/matriz completa | Dispositivo completo |
| Observaciones por dispositivo | Miles | 1 | 1 |
| Pseudo-replicación | ✗ Severa | ✅ Ninguna | ✅ Ninguna |
| Validez metodológica | ❌ Inválida | ✅ Válida | ✅ Válida |
| Tipo de características | Bins FFT individuales | Matrices FFT completas | Estadísticos agregados |
| Arquitectura | CNN | CNN | DNN Fully Connected |
| Accuracy obtenido | 81.8% (inflado) | 50.0% / 22.2% | Por determinar |
| Aplicabilidad | ❌ Cuestionable | ✅ Real | ✅ Real |
| Interpretabilidad | ❌ Baja | ✅ Media | ✅ Alta |

## Fundamento Teórico

### Principios Estadísticos Correctos
1. **Independencia de observaciones**: Cada dispositivo es independiente
2. **Alineación de unidades**: Observación = Inferencia
3. **Validez externa**: Resultados generalizables
4. **Rigor metodológico**: Sin inflación artificial

### Recomendación del Experto
> "El enfoque por bins de frecuencia genera pseudo-replicación al tratar componentes del mismo dispositivo como observaciones independientes. El enfoque correcto debe alinear la unidad de observación con la unidad de inferencia."

## Objetivos del Experimento 4

El Experimento 4 busca **mejorar las métricas de los Exp2/Exp3** manteniendo la validez metodológica:

1. **Exp2**: 50.0% accuracy - metodológicamente correcto pero métricas modestas
2. **Exp3**: 22.2% accuracy - metodológicamente correcto con balanceo
3. **Exp4**: Por determinar - enfoque agregado para mejor discriminación

### Hipótesis
Las características estadísticas agregadas pueden proporcionar mejor discriminación entre niveles de daño que las matrices FFT completas, potencialmente mejorando el accuracy sobre el 50.0% de Exp2.

## Resultados Científicos Válidos

Este experimento producirá **resultados metodológicamente válidos** comparables con Exp2/Exp3:

1. **Métricas reales** de capacidad predictiva
2. **Evaluación científica** del enfoque de características agregadas
3. **Comparación directa** con enfoques CNN (Exp2/Exp3)
4. **Base sólida** para conclusiones científicas

## Conclusión Metodológica

El Experimento 4 representa un enfoque alternativo metodológicamente correcto que busca mejorar las métricas de Exp2/Exp3 usando características estadísticas más discriminativas. Junto con Exp2/Exp3, constituye la base científicamente válida del proyecto.

---

**Nota Importante**: Este experimento implementa las recomendaciones críticas del documento de consulta experta para asegurar validez metodológica y aplicabilidad práctica de los resultados.