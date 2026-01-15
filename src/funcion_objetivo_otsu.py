import numpy as np

def otsu_objective_function(thresholds, image):
    # Asegúrate de que los umbrales sean enteros y estén ordenados
    thresholds = sorted([int(t) for t in thresholds])
    
    # Calcular el histograma y el mean total de la imagen
    histogram, _ = np.histogram(image, bins=np.arange(257), density=True)
    pixel_values = np.arange(0, 256)
    total_mean = np.dot(histogram, pixel_values)
    
    # Calcular la varianza entre clases para la segmentación multiclase
    # Otsu maximiza la varianza entre clases (between-class variance).
    # Aquí calculamos esa varianza.
    total_variance = 0
    for i, threshold in enumerate(thresholds + [256], start=1):
        lower = 0 if i == 1 else thresholds[i-2]
        upper = threshold
        
        # Validar rangos
        if lower >= upper:
            continue
            
        prob = histogram[lower:upper]
        weight = prob.sum()
        
        if weight > 0:
            mean = np.dot(prob, np.arange(lower, upper)) / weight
            variance = weight * ((mean - total_mean) ** 2)
            total_variance += variance
            
    return total_variance
