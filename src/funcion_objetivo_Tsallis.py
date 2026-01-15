import numpy as np

def tsallis_objective_function(thresholds, image):
    # Histogram of the image
    histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    thresholds = [int(round(t)) for t in thresholds]
    thresholds = sorted(thresholds)

    q = 1.5  # Parámetro de no extensividad (puedes ajustar este valor)

    p_regions = []
    w_regions = []
    S_regions = []
    J = 0

    total_sum = np.sum(histogram)

    for region_idx in range(len(thresholds) + 1):
        start = 0 if region_idx == 0 else thresholds[region_idx - 1]
        end = thresholds[region_idx] if region_idx < len(thresholds) else 255
        
        if start >= end:
             S_regions.append(1e-8)
             w_regions.append(0) # Keep w_regions aligned with S_regions
             continue

        p_region = histogram[start:end]
        p_regions.append(p_region)
        w_region = np.sum(p_region)
        w_regions.append(w_region)

        # Calcular la entropía de Tsallis directamente
        if w_region > 0:
            probabilities = p_region / total_sum
            # Normalize probabilities within the region? 
            # Usually Tsallis uses normalized probabilities p_i / w_i. 
            # But the provided code used p_region / total_sum (which are global probabilities).
            # Let's check the original code logic again.
            # Original: probabilities = p_region / total_sum
            # This looks like global probability.
            # Standard Tsallis often uses renormalized probabilities for each class.
            # However, I should stick to the user's logic if possible, or correct it if it looks definitely wrong.
            # Let's stick to the user's implementation structure but fix the scope.
            # Wait, p_region comes from histogram density=True? No, density=True makes sum=1.
            # So histogram is p_i.
            # w_region is sum(p_i) for the region.
            # User code: probabilities = p_region / total_sum. total_sum of density=True histogram is 1.
            # So probabilities = p_region.
            # S_region = (1/(q-1)) * (1 - sum(probabilities^q)).
            # This looks like the entropy of the sub-distribution without renormalization?
            # Standard Kapur/Tsallis for segmentation usually renormalizes: P_i / w_k.
            # Let's look at the Kapur code: p / w_regions[region_idx]. That is renormalized.
            # In the Tsallis code provided: `probabilities = p_region / total_sum`. That is NOT renormalized.
            # It also calculates `J += w_region / total_sum * S_region`.
            # This looks like a weighted sum of entropies?
            # Or maybe `J += (1-q) * prod(S_regions)` (at the end).
            # The line 35 `J += (1 - q) * np.prod(S_regions)` suggests this is maximizing the pseudo-additive entropy.
            # For independent systems A and B: S(A+B) = S(A) + S(B) + (1-q)S(A)S(B).
            # So for multiple thresholds, it's likely using that property.
            # If so, S_region should be the entropy of that region *renormalized*.
            # If I don't renormalize, S_region will be very small and not represent the "class entropy".
            # However, I will strictly follow the code I read in the Read tool to avoid changing logic unless necessary.
            # The Read tool output showed:
            # probabilities = p_region / total_sum
            # S_region = (1 / (q - 1)) * (1 - np.sum(probabilities ** q))
            # I will preserve this logic.
            
            probabilities = p_region / total_sum
            S_region = (1 / (q - 1)) * (1 - np.sum(probabilities ** q))
        else:
            S_region = 1e-8  # Asignar un valor pequeño y no nulo

        S_regions.append(S_region)
        # The original code had this loop accumulation for J?
        # J += w_region / total_sum * S_region
        # But then it also returns J += (1 - q) * np.prod(S_regions) ??
        # The read code:
        # 33: J += w_region / total_sum * S_region
        # 35: J += (1 - q) * np.prod(S_regions)
        # This looks weird (mixing additive and pseudo-additive), but I will copy it exactly.
        # Wait, if I look closely at the snippet:
        # J is initialized to 0.
        # Loop adds weighted S_region.
        # End adds product term.
        # This seems to be a specific implementation.
        
    # Re-implementing the loop logic from the snippet
    for i, s in enumerate(S_regions):
         # The loop in the snippet calculated J inside.
         pass 
         
    # Re-calculating J to match snippet exactly
    J = 0
    for i in range(len(S_regions)):
        w_val = w_regions[i]
        s_val = S_regions[i]
        J += (w_val / total_sum) * s_val
        
    J += (1 - q) * np.prod(S_regions)

    return J
