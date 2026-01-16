"""
Módulo para conversión de archivos DICOM a PNG manteniendo propiedades de 16-bit
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

# Importación para DICOM
try:
    import pydicom
    DICOM_DISPONIBLE = True
    print("📋 pydicom disponible - Conversión DICOM→PNG activada")
except ImportError:
    DICOM_DISPONIBLE = False
    print("⚠️ pydicom no disponible - Para soporte DICOM instalar: pip install pydicom")

class ConvertidorDICOM:
    """
    Clase para convertir archivos DICOM a PNG manteniendo las propiedades de 16-bit
    """
    
    def __init__(self, carpeta_origen: str, carpeta_destino: str = None):
        """
        Inicializa el convertidor DICOM
        
        Args:
            carpeta_origen: Carpeta que contiene archivos DICOM
            carpeta_destino: Carpeta donde guardar PNGs (por defecto: carpeta_origen/dcm_data_png)
        """
        self.carpeta_origen = Path(carpeta_origen)
        
        if carpeta_destino:
            self.carpeta_destino = Path(carpeta_destino)
        else:
            self.carpeta_destino = self.carpeta_origen / "dcm_data_png"
        
        # Crear carpeta destino si no existe
        self.carpeta_destino.mkdir(exist_ok=True)
        
        print(f"🔄 Convertidor DICOM inicializado")
        print(f"   📁 Origen: {self.carpeta_origen}")
        print(f"   💾 Destino: {self.carpeta_destino}")

    def save_as_png(self, image: np.ndarray, output_path: str) -> bool:
        """
        Guarda la imagen como archivo PNG en escala de grises de 16 bits REAL
        
        Args:
            image (numpy.ndarray): Imagen a guardar (debe ser uint16)
            output_path (str): Ruta donde guardar la imagen
            
        Returns:
            bool: True si se guardó exitosamente, False en caso contrario
        """
        try:
            # Verificar que la imagen sea uint16 para garantizar 16 bits
            if image.dtype != np.uint16:
                print(f"    ⚠️ Imagen no es uint16, convirtiendo desde {image.dtype}")
                # Normalizar y convertir a uint16
                img_min = np.min(image)
                img_max = np.max(image)
                
                if img_min == img_max:
                    image_16bit = np.zeros_like(image, dtype=np.uint16)
                    print(f"    ⚠️ Imagen constante (min=max={img_min}), guardando como ceros")
                else:
                    image_16bit = ((image - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
                    print(f"    ✅ Convertida a uint16: [{img_min:.1f}, {img_max:.1f}] → [0, 65535]")
            else:
                image_16bit = image
                print(f"    ✅ Imagen ya es uint16, rango: [{np.min(image)}, {np.max(image)}]")
            
            # Guardar usando OpenCV que sí preserva 16 bits en PNG
            success = cv2.imwrite(output_path, image_16bit)
            
            if success:
                # Verificar que se guardó correctamente
                imagen_verificacion = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
                if imagen_verificacion is not None:
                    print(f"    💾 PNG 16-bit guardado: {Path(output_path).name}")
                    print(f"    🔍 Verificación: dtype={imagen_verificacion.dtype}, shape={imagen_verificacion.shape}")
                    print(f"    🔢 Rango final: [{np.min(imagen_verificacion)}, {np.max(imagen_verificacion)}]")
                    return True
                else:
                    print(f"    ❌ Error: no se puede leer el archivo guardado")
                    return False
            else:
                print(f"    ❌ cv2.imwrite falló")
                return False
            
        except Exception as e:
            print(f"    ❌ Error al guardar PNG: {str(e)}")
            return False

    def validar_conversion_png(self, archivo_png: Path) -> bool:
        """
        Valida que el archivo PNG se haya generado correctamente como 16-bit.
        
        Args:
            archivo_png: Ruta al archivo PNG generado
            
        Returns:
            bool: True si la conversión es válida, False en caso contrario
        """
        try:
            # Verificar que el archivo existe
            if not archivo_png.exists():
                print(f"    ❌ El archivo PNG no existe: {archivo_png}")
                return False
            
            # Verificar tamaño del archivo (muy pequeño indicaría problema)
            tamaño_kb = archivo_png.stat().st_size / 1024
            if tamaño_kb < 1:  # Menos de 1KB probablemente sea un error
                print(f"    ⚠️ Archivo PNG muy pequeño ({tamaño_kb:.1f}KB): {archivo_png.name}")
                return False
            
            # Cargar la imagen SIN conversión automática para verificar tipo real
            imagen_test = cv2.imread(str(archivo_png), cv2.IMREAD_UNCHANGED)
            if imagen_test is None:
                print(f"    ❌ No se puede leer el PNG generado: {archivo_png.name}")
                return False
            
            # VERIFICACIÓN CRÍTICA: debe ser uint16 para 16 bits
            if imagen_test.dtype != np.uint16:
                print(f"    ❌ PNG NO es 16-bit! dtype={imagen_test.dtype}: {archivo_png.name}")
                print(f"        💡 Se guardó como {imagen_test.dtype} en lugar de uint16")
                return False
            
            # Verificar que la imagen no sea completamente negra
            valores_unicos = np.unique(imagen_test)
            if len(valores_unicos) == 1 and valores_unicos[0] == 0:
                print(f"    ⚠️ La imagen PNG está completamente negra: {archivo_png.name}")
                return False
            
            # Verificar que use el rango de 16 bits
            min_val = np.min(imagen_test)
            max_val = np.max(imagen_test)
            rango_usado = max_val - min_val
            
            print(f"    ✅ PNG 16-bit validado: {archivo_png.name} ({tamaño_kb:.1f}KB)")
            print(f"    📊 Tipo: {imagen_test.dtype}, Shape: {imagen_test.shape}")
            print(f"    📈 Rango: [{min_val}, {max_val}] (dinámico: {rango_usado})")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error validando PNG {archivo_png.name}: {str(e)}")
            return False

    def diagnosticar_profundidad_bits(self, archivo_png: Path) -> dict:
        """
        Diagnóstica la profundidad de bits real de un archivo PNG
        
        Args:
            archivo_png: Ruta al archivo PNG a diagnosticar
            
        Returns:
            dict: Información detallada sobre la profundidad de bits
        """
        diagnostico = {
            'archivo': archivo_png.name,
            'existe': False,
            'es_16bit': False,
            'dtype': None,
            'shape': None,
            'rango': None,
            'tamaño_kb': 0
        }
        
        try:
            if not archivo_png.exists():
                print(f"📋 Archivo no existe: {archivo_png}")
                return diagnostico
            
            diagnostico['existe'] = True
            diagnostico['tamaño_kb'] = archivo_png.stat().st_size / 1024
            
            # Cargar con diferentes métodos para comparar
            
            # Método 1: OpenCV sin conversión (UNCHANGED preserva tipo original)
            img_cv_unchanged = cv2.imread(str(archivo_png), cv2.IMREAD_UNCHANGED)
            
            # Método 2: OpenCV en escala de grises (podría convertir a 8-bit)
            img_cv_gray = cv2.imread(str(archivo_png), cv2.IMREAD_GRAYSCALE)
            
            if img_cv_unchanged is not None:
                diagnostico['dtype'] = str(img_cv_unchanged.dtype)
                diagnostico['shape'] = img_cv_unchanged.shape
                diagnostico['es_16bit'] = img_cv_unchanged.dtype == np.uint16
                diagnostico['rango'] = [int(np.min(img_cv_unchanged)), int(np.max(img_cv_unchanged))]
                
                print(f"📊 DIAGNÓSTICO: {archivo_png.name}")
                print(f"   📁 Tamaño: {diagnostico['tamaño_kb']:.1f} KB")
                print(f"   🔢 OpenCV UNCHANGED: {img_cv_unchanged.dtype}, shape={img_cv_unchanged.shape}")
                print(f"   📈 Rango de valores: {diagnostico['rango']}")
                
                if img_cv_gray is not None:
                    print(f"   🔄 OpenCV GRAYSCALE: {img_cv_gray.dtype}, shape={img_cv_gray.shape}")
                    print(f"   📈 Rango grayscale: [{np.min(img_cv_gray)}, {np.max(img_cv_gray)}]")
                
                if diagnostico['es_16bit']:
                    print(f"   ✅ CONFIRMADO: Es PNG de 16 bits")
                else:
                    print(f"   ❌ PROBLEMA: Es PNG de {img_cv_unchanged.dtype} (no 16-bit)")
                    
            else:
                print(f"   ❌ No se puede leer la imagen")
                
        except Exception as e:
            print(f"   ❌ Error en diagnóstico: {e}")
            
        return diagnostico

    def extraer_primera_imagen_2d(self, imagen_original):
        """
        Extrae la primera imagen 2D de un array que puede ser multi-slice
        
        Args:
            imagen_original: Array numpy con datos de imagen DICOM
            
        Returns:
            Tuple: (imagen_2d, info_extraccion)
        """
        info_extraccion = {
            'shape_original': imagen_original.shape,
            'dimensiones_originales': len(imagen_original.shape),
            'tipo_original': str(imagen_original.dtype)
        }
        
        # Si es una imagen multi-dimensional, tomar la primera slice
        if len(imagen_original.shape) == 2:
            # Ya es 2D
            imagen_2d = imagen_original
            info_extraccion['metodo'] = 'ya_2d'
            print(f"    📋 Imagen ya es 2D: {imagen_original.shape}")
            
        elif len(imagen_original.shape) == 3:
            # 3D: tomar primera slice
            imagen_2d = imagen_original[0]
            info_extraccion['metodo'] = '3d_primera_slice'
            print(f"    📋 Imagen 3D detectada, usando primera slice: {imagen_original.shape} → {imagen_2d.shape}")
            
        elif len(imagen_original.shape) == 4:
            # 4D: tomar primera slice del primer volumen
            imagen_2d = imagen_original[0, 0]
            info_extraccion['metodo'] = '4d_primera_slice_primer_volumen'
            print(f"    📋 Imagen 4D detectada, usando primera slice del primer volumen: {imagen_original.shape} → {imagen_2d.shape}")
            
        else:
            # Para dimensiones mayores, tomar los primeros índices
            indices = tuple([0] * (len(imagen_original.shape) - 2))
            imagen_2d = imagen_original[indices]
            info_extraccion['metodo'] = f'{len(imagen_original.shape)}d_primeros_indices'
            print(f"    📋 Imagen {len(imagen_original.shape)}D detectada, usando primeros índices: {imagen_original.shape} → {imagen_2d.shape}")
        
        # Verificar que el resultado final es 2D
        if len(imagen_2d.shape) != 2:
            raise ValueError(f"No se pudo extraer imagen 2D. Resultado: {imagen_2d.shape}")
        
        info_extraccion['shape_final'] = imagen_2d.shape
        return imagen_2d, info_extraccion

    def preparar_imagen_16bit_para_png(self, imagen: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Prepara la imagen DICOM manteniendo las propiedades de 16 bits
        
        Args:
            imagen: Array numpy con datos de imagen DICOM
            
        Returns:
            Tuple: (imagen_uint16, info_conversion)
        """
        # Primero extraer imagen 2D (siempre la primera slice)
        imagen_2d, info_extraccion = self.extraer_primera_imagen_2d(imagen)
        
        info_conversion = {
            'extraccion_info': info_extraccion,
            'tipo_original': str(imagen_2d.dtype),
            'shape_original': imagen_2d.shape,
            'min_original': float(np.min(imagen_2d)),
            'max_original': float(np.max(imagen_2d))
        }
        
        # Convertir a uint16 para mantener la información de 16 bits
        if imagen_2d.dtype == np.int16:
            # Si es signed int16, convertir a unsigned int16
            # Desplazar el rango de [-32768, 32767] a [0, 65535]
            imagen_uint16 = (imagen_2d.astype(np.int32) + 32768).astype(np.uint16)
            print(f"    🔄 Convertido de int16 a uint16 (desplazamiento +32768)")
            info_conversion['metodo'] = 'int16_shift'
            
        elif imagen_2d.dtype == np.uint16:
            # Ya es uint16, usar directamente
            imagen_uint16 = imagen_2d
            print(f"    ✅ Manteniendo formato uint16 original")
            info_conversion['metodo'] = 'uint16_direct'
            
        else:
            # Para otros tipos, convertir manteniendo el rango
            min_val = np.min(imagen_2d)
            max_val = np.max(imagen_2d)
            
            if max_val > min_val:
                # Escalar al rango completo de uint16 [0, 65535]
                imagen_normalizada = 65535 * (imagen_2d.astype(np.float64) - min_val) / (max_val - min_val)
                imagen_uint16 = imagen_normalizada.astype(np.uint16)
                print(f"    🔄 Escalado de {imagen_2d.dtype} a uint16 [0-65535]")
                info_conversion['metodo'] = 'scaled_to_uint16'
            else:
                imagen_uint16 = np.zeros_like(imagen_2d, dtype=np.uint16)
                print(f"    ⚠️ Imagen constante, convertida a uint16 zeros")
                info_conversion['metodo'] = 'constant_zero'
        
        info_conversion.update({
            'shape_final': imagen_uint16.shape,
            'min_final': float(np.min(imagen_uint16)),
            'max_final': float(np.max(imagen_uint16)),
            'dtype_final': str(imagen_uint16.dtype)
        })
        
        return imagen_uint16, info_conversion

    def extraer_te_desde_dicom(self, dicom_data) -> Optional[float]:
        """
        Extrae el valor de TE desde los metadatos DICOM
        
        Args:
            dicom_data: Objeto pydicom con datos DICOM
            
        Returns:
            Valor de TE en microsegundos o None si no se encuentra
        """
        te_valor = None
        
        try:
            if hasattr(dicom_data, 'EchoTime'):
                te_valor = float(dicom_data.EchoTime) * 1000  # Convertir de ms a µs
                print(f"    ✅ TE extraído del DICOM: {te_valor:.1f} µs")
            elif hasattr(dicom_data, 'RepetitionTime'):
                # Algunos DICOM tienen TR en lugar de TE
                te_valor = float(dicom_data.RepetitionTime) * 1000
                print(f"    ℹ️ Usando TR como TE: {te_valor:.1f} µs")
            else:
                print(f"    ⚠️ No se encontró TE/TR en metadatos DICOM")
        except Exception as e:
            print(f"    ❌ Error extrayendo TE del DICOM: {e}")
        
        return te_valor

    def convertir_dicom_a_png(self, archivo_dicom: Path, conservar_metadatos: bool = True) -> Optional[Path]:
        """
        Convierte un archivo DICOM a PNG manteniendo propiedades 16-bit
        
        Args:
            archivo_dicom: Ruta al archivo DICOM
            conservar_metadatos: Si guardar metadatos en archivo JSON
            
        Returns:
            Ruta del archivo PNG generado o None si falla
        """
        if not DICOM_DISPONIBLE:
            print(f"❌ pydicom no disponible para convertir {archivo_dicom.name}")
            return None
        
        try:
            print(f"🔄 Procesando: {archivo_dicom.name}")
            
            # Leer archivo DICOM
            dicom_data = pydicom.dcmread(str(archivo_dicom))
            
            if not hasattr(dicom_data, 'pixel_array'):
                print(f"    ❌ DICOM no contiene pixel_array: {archivo_dicom.name}")
                return None
            
            # Obtener imagen
            imagen_original = dicom_data.pixel_array
            
            # Preparar imagen para PNG
            imagen_png, info_conversion = self.preparar_imagen_16bit_para_png(imagen_original)
            
            # Generar nombre de archivo PNG (conservar nombre original)
            nombre_base = archivo_dicom.stem
            archivo_png = self.carpeta_destino / f"{nombre_base}.png"
            
            # Guardar como PNG usando la función mejorada
            success = self.save_as_png(imagen_png, str(archivo_png))
            
            if success:
                # Validar que la conversión sea correcta
                if self.validar_conversion_png(archivo_png):
                    print(f"    ✅ PNG generado y validado exitosamente: {archivo_png.name}")
                    
                    # Guardar metadatos si se solicita
                    if conservar_metadatos:
                        self.guardar_metadatos_dicom(archivo_dicom, dicom_data, info_conversion)
                    
                    return archivo_png
                else:
                    print(f"    ⚠️ PNG generado pero falló la validación: {archivo_png.name}")
                    return None
            else:
                print(f"    ❌ Error guardando PNG: {archivo_png}")
                return None
                
        except Exception as e:
            print(f"    ❌ Error procesando {archivo_dicom.name}: {str(e)}")
            return None

    def guardar_metadatos_dicom(self, archivo_dicom: Path, dicom_data, info_conversion: dict):
        """
        Guarda metadatos DICOM relevantes en archivo JSON
        
        Args:
            archivo_dicom: Archivo DICOM original
            dicom_data: Datos DICOM parseados
            info_conversion: Información de la conversión
        """
        try:
            import json
            
            metadatos = {
                'archivo_original': archivo_dicom.name,
                'conversion_info': info_conversion,
                'dicom_metadata': {}
            }
            
            # Extraer metadatos relevantes
            campos_relevantes = [
                'PatientName', 'StudyDescription', 'SeriesDescription',
                'EchoTime', 'RepetitionTime', 'PixelSpacing', 'SliceThickness',
                'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit'
            ]
            
            for campo in campos_relevantes:
                if hasattr(dicom_data, campo):
                    valor = getattr(dicom_data, campo)
                    # Convertir tipos no serializables
                    if hasattr(valor, 'value'):
                        valor = valor.value
                    metadatos['dicom_metadata'][campo] = str(valor)
            
            # Extraer TE
            te_valor = self.extraer_te_desde_dicom(dicom_data)
            if te_valor:
                metadatos['te_microsegundos'] = te_valor
            
            # Guardar archivo JSON
            archivo_json = self.carpeta_destino / f"{archivo_dicom.stem}_metadata.json"
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(metadatos, f, indent=2, ensure_ascii=False)
            
            print(f"    📄 Metadatos guardados: {archivo_json.name}")
            
        except Exception as e:
            print(f"    ⚠️ Error guardando metadatos: {e}")

    def convertir_carpeta_completa(self, conservar_metadatos: bool = True) -> List[Path]:
        """
        Convierte todos los archivos DICOM de la carpeta a PNG
        
        Args:
            conservar_metadatos: Si guardar metadatos DICOM
            
        Returns:
            Lista de archivos PNG generados
        """
        # Buscar archivos DICOM
        extensiones_dicom = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']
        archivos_dicom = []
        
        for ext in extensiones_dicom:
            archivos_dicom.extend(list(self.carpeta_origen.glob(ext)))
        
        if not archivos_dicom:
            print(f"❌ No se encontraron archivos DICOM en: {self.carpeta_origen}")
            return []
        
        print(f"📋 Encontrados {len(archivos_dicom)} archivos DICOM")
        print(f"🚀 Iniciando conversión DICOM → PNG...")
        
        archivos_png_generados = []
        errores = 0
        
        for archivo_dicom in sorted(archivos_dicom):
            archivo_png = self.convertir_dicom_a_png(archivo_dicom, conservar_metadatos)
            if archivo_png:
                archivos_png_generados.append(archivo_png)
            else:
                errores += 1
        
        print(f"\n📊 RESUMEN DE CONVERSIÓN:")
        print(f"   ✅ Exitosos: {len(archivos_png_generados)}")
        print(f"   ❌ Errores: {errores}")
        print(f"   📁 Archivos PNG en: {self.carpeta_destino}")
        
        return archivos_png_generados

def verificar_tipo_archivos(carpeta: str) -> Tuple[bool, bool, int, int]:
    """
    Verifica qué tipos de archivos hay en la carpeta
    
    Args:
        carpeta: Ruta de la carpeta a verificar
        
    Returns:
        Tuple: (tiene_png, tiene_dicom, count_png, count_dicom)
    """
    carpeta_path = Path(carpeta)
    
    if not carpeta_path.exists():
        return False, False, 0, 0
    
    # Buscar PNG
    archivos_png = list(carpeta_path.glob("*.png"))
    archivos_png.extend(list(carpeta_path.glob("*.PNG")))
    
    # Buscar DICOM
    extensiones_dicom = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']
    archivos_dicom = []
    for ext in extensiones_dicom:
        archivos_dicom.extend(list(carpeta_path.glob(ext)))
    
    tiene_png = len(archivos_png) > 0
    tiene_dicom = len(archivos_dicom) > 0
    
    return tiene_png, tiene_dicom, len(archivos_png), len(archivos_dicom)

def diagnosticar_carpeta_png(carpeta: str) -> dict:
    """
    Función de conveniencia para diagnosticar todos los PNG de una carpeta
    
    Args:
        carpeta: Ruta de la carpeta a diagnosticar
        
    Returns:
        dict: Resumen del diagnóstico
    """
    carpeta_path = Path(carpeta)
    
    if not carpeta_path.exists():
        print(f"❌ Carpeta no existe: {carpeta}")
        return {'error': 'carpeta_no_existe'}
    
    # Buscar todos los PNG
    archivos_png = list(carpeta_path.glob("*.png"))
    archivos_png.extend(list(carpeta_path.glob("*.PNG")))
    
    if not archivos_png:
        print(f"📋 No se encontraron archivos PNG en: {carpeta}")
        return {'total': 0, 'archivos': []}
    
    print(f"🔍 DIAGNÓSTICO DE PROFUNDIDAD DE BITS")
    print(f"📁 Carpeta: {carpeta}")
    print(f"📊 Total de PNG encontrados: {len(archivos_png)}")
    print(f"{'='*60}")
    
    # Crear una instancia temporal del convertidor para usar el método
    convertidor = ConvertidorDICOM(str(carpeta_path))
    
    resultados = {
        'total': len(archivos_png),
        'png_16bit': 0,
        'png_8bit': 0,
        'errores': 0,
        'archivos': []
    }
    
    for archivo_png in sorted(archivos_png):
        diagnostico = convertidor.diagnosticar_profundidad_bits(archivo_png)
        resultados['archivos'].append(diagnostico)
        
        if diagnostico['existe']:
            if diagnostico['es_16bit']:
                resultados['png_16bit'] += 1
            else:
                resultados['png_8bit'] += 1
        else:
            resultados['errores'] += 1
        
        print()  # Línea en blanco entre archivos
    
    print(f"{'='*60}")
    print(f"📈 RESUMEN:")
    print(f"   ✅ PNG de 16 bits: {resultados['png_16bit']}")
    print(f"   ⚠️  PNG de 8 bits: {resultados['png_8bit']}")
    print(f"   ❌ Errores: {resultados['errores']}")
    
    if resultados['png_8bit'] > 0:
        print(f"")
        print(f"🚨 ATENCIÓN: Se encontraron {resultados['png_8bit']} archivos PNG de 8 bits")
        print(f"   Esto indica que la conversión DICOM→PNG no está preservando 16 bits")
        print(f"   Recomendación: Reconvertir usando el convertidor actualizado")
    
    return resultados

def convertir_dicom_rapido(carpeta_origen: str, carpeta_destino: str = None) -> str:
    """
    Función de conveniencia para conversión rápida DICOM → PNG
    
    Args:
        carpeta_origen: Carpeta con archivos DICOM
        carpeta_destino: Carpeta destino (opcional)
        
    Returns:
        Ruta de la carpeta con archivos PNG
    """
    convertidor = ConvertidorDICOM(carpeta_origen, carpeta_destino)
    archivos_generados = convertidor.convertir_carpeta_completa()
    
    if archivos_generados:
        print(f"🎉 Conversión completada: {len(archivos_generados)} archivos PNG")
        return str(convertidor.carpeta_destino)
    else:
        raise ValueError("No se pudieron convertir archivos DICOM")

# Función principal para testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python dicom_converter.py <carpeta_dicom>")
        sys.exit(1)
    
    carpeta = sys.argv[1]
    try:
        carpeta_png = convertir_dicom_rapido(carpeta)
        print(f"✅ Archivos PNG disponibles en: {carpeta_png}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)