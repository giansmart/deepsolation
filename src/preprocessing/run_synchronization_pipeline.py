#!/usr/bin/env python
"""
Pipeline completo de sincronización de señales S1-S2.

Este script ejecuta el pipeline end-to-end:
1. Detección de offsets temporales
2. Aplicación de correcciones
3. Validación de resultados

Uso:
    python -m src.preprocessing.run_synchronization_pipeline
    python -m src.preprocessing.run_synchronization_pipeline --detect-only
    python -m src.preprocessing.run_synchronization_pipeline --correct-only

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

import sys
from pathlib import Path

from .detect_timestamp_offsets import detect_offsets
from .apply_timestamp_correction import apply_corrections


def run_pipeline(
    signals_dir: str = 'data/Signals_Raw/',
    labels_csv: str = 'data/nivel_damage.csv',
    offsets_csv: str = 'data/processed/timestamp_offsets.csv',
    output_dir: str = 'data/processed/synchronized/',
    detect_only: bool = False,
    correct_only: bool = False
):
    """
    Ejecuta el pipeline completo de sincronización.

    Args:
        signals_dir: Directorio raíz de señales RAW
        labels_csv: CSV con etiquetas
        offsets_csv: CSV de offsets (salida de paso 1, entrada de paso 2)
        output_dir: Directorio de salida para señales sincronizadas
        detect_only: Si True, solo ejecuta detección
        correct_only: Si True, solo ejecuta corrección

    Returns:
        Dict con resultados del pipeline
    """
    results = {}

    # PASO 1: Detección de offsets
    if not correct_only:
        print("\n" + "="*70)
        print("PASO 1: DETECCIÓN DE OFFSETS TEMPORALES")
        print("="*70 + "\n")

        offsets_df = detect_offsets(
            signals_dir=signals_dir,
            labels_csv=labels_csv
        )

        # Guardar tabla de offsets
        offsets_path = Path(offsets_csv)
        offsets_path.parent.mkdir(parents=True, exist_ok=True)
        offsets_df.to_csv(offsets_path, index=False)

        results['offsets_detected'] = len(offsets_df)
        results['offsets_file'] = str(offsets_path)

        print(f"\n✅ Tabla de offsets guardada en: {offsets_path}")
        print(f"   {len(offsets_df)} registros escritos.\n")

        if detect_only:
            return results

    # PASO 2: Aplicación de correcciones
    if not detect_only:
        print("\n" + "="*70)
        print("PASO 2: APLICACIÓN DE CORRECCIONES")
        print("="*70 + "\n")

        stats = apply_corrections(
            signals_dir=signals_dir,
            offsets_csv=offsets_csv,
            output_dir=output_dir,
            method='shift_indices'
        )

        results['correction_stats'] = stats
        results['output_dir'] = output_dir

        print(f"\n✅ Señales sincronizadas guardadas en: {output_dir}")

    # Reporte final
    if not detect_only and not correct_only:
        print("\n" + "="*70)
        print("📊 REPORTE FINAL DEL PIPELINE")
        print("="*70)
        print(f"\n✅ Pipeline completado exitosamente\n")

        print(f"   📂 Archivos generados:")
        print(f"      Tabla de offsets: {results['offsets_file']}")
        print(f"      Señales sincronizadas: {results['output_dir']}\n")

        print(f"   📈 Estadísticas:")
        print(f"      Total mediciones: {results['offsets_detected']}")
        print(f"      Señales corregidas: {stats['corrected']}")
        print(f"      Ya sincronizadas: {stats['already_synced']}")
        print(f"      Errores: {len(stats['errors'])}\n")

        print(f"   🔍 Validación:")
        print(f"      Validaciones exitosas: {stats['validation_passed']}")
        print(f"      Validaciones fallidas: {stats['validation_failed']}")

        if stats['validation_failed'] > 0:
            total_val = stats['validation_passed'] + stats['validation_failed']
            pct_success = (stats['validation_passed'] / total_val) * 100
            print(f"      Tasa de éxito: {pct_success:.1f}%")

        print("\n" + "="*70 + "\n")

        # Advertencias
        if len(stats['errors']) > 0:
            print("⚠️  ADVERTENCIA: Se encontraron errores durante el procesamiento:")
            for error in stats['errors']:
                print(f"   - {error['specimen_id']}: {error['error']}")
            print()  # Línea en blanco al final

        if stats['validation_failed'] > 0:
            print(f"⚠️  ADVERTENCIA: {stats['validation_failed']} señales con sincronización subóptima.")
            print("   Revisar archivos metadata.json en el directorio de salida.\n")

    return results


def main():
    """
    Función principal para ejecución desde línea de comandos.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline completo de sincronización de señales S1-S2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar pipeline completo
  python scripts/run_synchronization_pipeline.py

  # Solo detectar offsets
  python scripts/run_synchronization_pipeline.py --detect-only

  # Solo aplicar correcciones (requiere que exista timestamp_offsets.csv)
  python scripts/run_synchronization_pipeline.py --correct-only

  # Especificar rutas personalizadas
  python scripts/run_synchronization_pipeline.py \\
      --signals-dir data/Signals_Raw/ \\
      --output-dir data/processed/synchronized_custom/
        """
    )

    parser.add_argument(
        '--signals-dir',
        default='data/Signals_Raw/',
        help='Directorio raíz de señales RAW'
    )
    parser.add_argument(
        '--labels-csv',
        default='data/nivel_damage.csv',
        help='CSV con etiquetas'
    )
    parser.add_argument(
        '--offsets-csv',
        default='data/processed/timestamp_offsets.csv',
        help='CSV de offsets'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/synchronized/',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help='Solo ejecutar detección de offsets'
    )
    parser.add_argument(
        '--correct-only',
        action='store_true',
        help='Solo ejecutar corrección de señales'
    )

    args = parser.parse_args()

    # Validar argumentos
    if args.detect_only and args.correct_only:
        parser.error("No se puede usar --detect-only y --correct-only simultáneamente")

    # Ejecutar pipeline
    try:
        results = run_pipeline(
            signals_dir=args.signals_dir,
            labels_csv=args.labels_csv,
            offsets_csv=args.offsets_csv,
            output_dir=args.output_dir,
            detect_only=args.detect_only,
            correct_only=args.correct_only
        )

        print("✅ Pipeline ejecutado exitosamente.\n")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
