import numpy as np

# Standalone Kapur Objective Function
def kapur_objective_function(thresholds, image):
    # Histogram of the image
    histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    thresholds = [int(round(t)) for t in thresholds]
    thresholds = sorted(thresholds)

    p_regions = []
    w_regions = []
    A_regions = []
    J = 0

    for region_idx in range(len(thresholds) + 1):
        start = 0 if region_idx == 0 else thresholds[region_idx - 1]
        end = thresholds[region_idx] if region_idx < len(thresholds) else 255

        p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
        p_regions.append(p_region)
        w_regions.append(np.sum(p_region))

        # Avoid division by zero and log of zero
        w_val = w_regions[region_idx]
        if w_val == 0:
            A_region = [0 for _ in p_region] # Or minimal entropy contribution
        else:
             A_region = [-(p / w_val) * np.log(p / w_val) if p > 0 else 0.0 for p in p_region] # Changed 0.001 to 0.0 for correctness/consistency or keep 0.001 if user insisted. 
             # User code had 0.001. I'll stick to 0.001 to be safe with their logic.
             A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]

        A_regions.append(A_region)
        J += np.sum(A_region)

    return J
