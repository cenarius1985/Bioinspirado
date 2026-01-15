import os
import pandas as pd
import re

def unir_resultados(run_dir):
    """
    Busca todos los archivos Excel de resultados en el directorio de ejecución,
    los concatena en un único DataFrame agregando columnas de metadatos,
    y guarda el resultado consolidado en la raíz de run_dir.
    """
    print(f"\n{'='*50}")
    print(f"Iniciando Consolidación de Resultados en: {run_dir}")
    print(f"{'='*50}")

    all_dataframes = []
    
    # Patrón para extraer info del nombre del archivo: Resultados_{obj}_{dim}dim.xlsx
    # Ejemplo: Resultados_Kapur_3dim.xlsx
    filename_pattern = re.compile(r'Resultados_([a-zA-Z]+)_(\d+)dim\.xlsx')

    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("Resultados_") and file.endswith(".xlsx") and "Consolidado" not in file:
                file_path = os.path.join(root, file)
                
                # Intentar extraer metadatos del nombre del archivo
                match = filename_pattern.match(file)
                if match:
                    obj_func = match.group(1)
                    dim = match.group(2) + "D" # Formato "3D"
                else:
                    # Fallback: intentar inferir de la estructura de carpetas
                    # Estructura esperada: .../Kapur/3D/Resultados_...
                    try:
                        path_parts = os.path.normpath(file_path).split(os.sep)
                        # Asumiendo que las dos carpetas padres son Dimension y ObjFunc
                        dim = path_parts[-2] # "3D"
                        obj_func = path_parts[-3] # "Kapur"
                    except:
                        obj_func = "Desconocido"
                        dim = "Desconocido"
                
                print(f"  Procesando: {file} | Obj: {obj_func}, Dim: {dim}")
                
                try:
                    df = pd.read_excel(file_path)
                    
                    # Insertar columnas de metadatos al principio
                    df.insert(0, 'Dimension', dim)
                    df.insert(0, 'Funcion_Objetivo', obj_func)
                    
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"    Error leyendo {file}: {e}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        output_path = os.path.join(run_dir, "Resultados_Consolidados.xlsx")
        combined_df.to_excel(output_path, index=False)
        print(f"\n>>> Archivo consolidado guardado exitosamente en: {output_path}")
        print(f">>> Total de registros consolidados: {len(combined_df)}")
    else:
        print("\nNo se encontraron archivos de resultados para consolidar.")

if __name__ == "__main__":
    # Para pruebas manuales, se puede ejecutar este script apuntando a una carpeta específica
    import sys
    if len(sys.argv) > 1:
        unir_resultados(sys.argv[1])
    else:
        print("Uso: python unir_resultado_excel.py <ruta_del_directorio_de_resultados>")
