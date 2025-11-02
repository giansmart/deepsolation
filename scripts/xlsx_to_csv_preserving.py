#!/usr/bin/env python3
"""
Script para convertir XLSX a CSV preservando exactamente el formato y precisión original
Mantiene todos los tipos numéricos sin pérdida de información
"""

import pandas as pd
import numpy as np
from pathlib import Path

def xlsx_to_csv_preserve_format(xlsx_path, csv_output_path):
    """
    Convierte XLSX a CSV preservando exactamente el formato original
    
    Args:
        xlsx_path: Ruta al archivo Excel
        csv_output_path: Ruta donde guardar el CSV
    """
    
    print(f"📂 Cargando archivo Excel: {xlsx_path}")
    
    # Cargar Excel con configuración para preservar precisión
    df = pd.read_excel(
        xlsx_path,
        engine='openpyxl',  # Motor más confiable
        keep_default_na=True,  # Mantener valores NaN como están
        na_values=[''],  # Solo strings vacíos como NaN
    )
    
    print(f"✅ Archivo cargado exitosamente")
    print(f"📊 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar información de tipos de datos
    print(f"\n📋 Tipos de datos:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    print(f"   • Columnas numéricas: {len(numeric_cols)}")
    print(f"   • Columnas de texto: {len(text_cols)}")
    
    if len(text_cols) > 0:
        print(f"   • Columnas de texto: {list(text_cols)}")
    
    # Verificar valores muy pequeños
    small_values_info = []
    for col in numeric_cols:
        mask = (df[col] > 0) & (df[col] < 0.001)
        if mask.any():
            count = mask.sum()
            min_val = df[col][mask].min()
            small_values_info.append((col, count, min_val))
    
    print(f"\n🔬 Valores muy pequeños (<0.001):")
    print(f"   • Columnas con valores <0.001: {len(small_values_info)}")
    if small_values_info:
        print("   • Ejemplos:")
        for col, count, min_val in small_values_info[:3]:
            print(f"     - {col}: {count} valores, mínimo = {min_val:.2e}")
    
    # Guardar a CSV con configuración para preservar precisión
    print(f"\n💾 Guardando como CSV: {csv_output_path}")
    
    df.to_csv(
        csv_output_path,
        index=False,  # No incluir índice
        float_format=None,  # Mantener formato original de floats
        encoding='utf-8',
        na_rep='',  # Representar NaN como string vacío
    )
    
    print(f"✅ CSV guardado exitosamente")
    
    # Verificación: cargar el CSV y comparar
    print(f"\n🔍 Verificando integridad...")
    df_csv = pd.read_csv(csv_output_path)
    
    print(f"   • Dimensiones CSV: {df_csv.shape}")
    print(f"   • Dimensiones coinciden: {df.shape == df_csv.shape}")
    
    # Comparar algunos valores numéricos
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        excel_sample = df[sample_col].iloc[0]
        csv_sample = df_csv[sample_col].iloc[0]
        
        print(f"   • Muestra de verificación ({sample_col}):")
        print(f"     - Excel: {excel_sample} (tipo: {type(excel_sample)})")
        print(f"     - CSV:   {csv_sample} (tipo: {type(csv_sample)})")
        
        if isinstance(excel_sample, (int, float)) and isinstance(csv_sample, (int, float)):
            diff = abs(excel_sample - csv_sample)
            print(f"     - Diferencia: {diff}")
            print(f"     - Valores idénticos: {diff < 1e-15}")
    
    return df, df_csv

def main():
    """Función principal"""
    
    print("🚀 CONVERTIDOR XLSX → CSV (Preservando Formato Original)\n")
    
    # Rutas de archivos
    xlsx_file = Path("docs/ARR3_DF_FINAL.xlsx")
    csv_file = Path("data/ARR3_DF_FINAL_preserved.csv")
    
    # Verificar que existe el archivo Excel
    if not xlsx_file.exists():
        print(f"❌ Error: No se encuentra el archivo {xlsx_file}")
        return
    
    # Crear directorio de salida si no existe
    csv_file.parent.mkdir(exist_ok=True)
    
    try:
        # Realizar la conversión
        df_original, df_converted = xlsx_to_csv_preserve_format(xlsx_file, csv_file)
        
        print(f"\n🎉 ¡Conversión completada exitosamente!")
        print(f"📂 Archivo original: {xlsx_file}")
        print(f"📄 Archivo CSV:      {csv_file}")
        
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        raise

if __name__ == "__main__":
    main()