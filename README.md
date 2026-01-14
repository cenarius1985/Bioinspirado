<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<div align="center">
    <h1>🧬 Segmentación de Imágenes Médicas con Algoritmos Bio-inspirados 🧬</h1>
    <p>
        <strong>Implementación de técnicas de umbralización multinivel (7, 8 y 9 dimensiones) utilizando funciones objetivo de Kapur, Otsu y Tsallis.</strong>
    </p>
</div>

<hr>

<h2>📖 Descripción del Proyecto</h2>
<p>
    Este proyecto explora la aplicación de algoritmos de optimización metaheurísticos (bio-inspirados) para resolver problemas de segmentación de imágenes médicas mediante umbralización multinivel. Se evalúa el rendimiento de diversos algoritmos en dimensiones altas (7, 8 y 9 umbrales) utilizando métricas de calidad como <strong>PSNR</strong>, <strong>SSIM</strong> y el análisis de <strong>convergencia</strong>.
</p>

<h3>🤖 Algoritmos Bio-inspirados Utilizados</h3>
<ul>
    <li><strong>RSA</strong>: Reptile Search Algorithm</li>
    <li><strong>HBA</strong>: Honey Badger Algorithm</li>
    <li><strong>OPA</strong>: Osprey Prediction Algorithm</li>
    <li><strong>BES</strong>: Bald Eagle Search</li>
    <li><strong>GWO</strong>: Grey Wolf Optimizer</li>
    <li><strong>CSA</strong>: Crow Search Algorithm</li>
    <li><strong>HHO</strong>: Harris Hawks Optimization</li>
    <li><strong>TSO</strong>: Tuna Swarm Optimization</li>
</ul>

<hr>

<h2>⚙️ Funciones Objetivo</h2>

<p>A continuación se detallan las implementaciones en Python de las funciones objetivo utilizadas para evaluar la calidad de la segmentación.</p>

### 1. Entropía de Kapur

La entropía de Kapur busca maximizar la entropía de las clases separadas por los umbrales. Para una imagen con $L$ niveles de gris y $k$ umbrales $[t_1, t_2, \dots, t_k]$, la función objetivo se define como:

$$
J(t_1, \dots, t_k) = \sum_{j=0}^{k} H_j
$$

Donde las entropías parciales $H_j$ se calculan como:

$$
H_j = -\sum_{i=t_j}^{t_{j+1}-1} \frac{p_i}{\omega_j} \ln \left( \frac{p_i}{\omega_j} \right)
$$

Con $\omega_j$ siendo la probabilidad acumulada de la clase $j$:

$$
\omega_j = \sum_{i=t_j}^{t_{j+1}-1} p_i
$$

### 2. Método de Otsu

El método de Otsu busca maximizar la varianza entre clases ($\sigma_B^2$), lo que equivale a encontrar los umbrales que mejor separan las distribuciones de intensidad de los píxeles.

$$
\sigma_B^2(t_1, \dots, t_k) = \sum_{j=0}^{k} \omega_j (\mu_j - \mu_T)^2
$$

Donde:
* $\omega_j$: Probabilidad acumulada de la clase $j$.
* $\mu_j$: Media de intensidad de la clase $j$.
* $\mu_T$: Media de intensidad global de la imagen.

### 3. Entropía de Tsallis

La entropía de Tsallis es una generalización de la entropía de Shannon para sistemas no extensivos. La función objetivo busca maximizar la entropía total del sistema particionado:

$$
S_q(t_1, \dots, t_k) = \sum_{j=0}^{k} S_q^j + (1-q) \sum_{j \neq l} S_q^j S_q^l + \dots
$$

Donde la entropía de cada clase $S_q^j$ se define por el parámetro entrópico $q$:

$$
S_q^j = \frac{1}{q-1} \left( 1 - \sum_{i \in C_j} \left( \frac{p_i}{\omega_j} \right)^q \right)
$$

<hr>

<h2>💾 Almacenamiento de Resultados</h2>
<p>
    El sistema guarda automáticamente los vectores de umbrales, métricas PSNR, SSIM y curvas de convergencia en carpetas organizadas por métrica y dimensión.
</p>

<pre><code>import numpy as np
import os

# Create the folder "Metrica_8_bits_7_dim_otsu" if it doesn't exist
os.makedirs(carpeta, exist_ok=True)

# Save the arrays in the "Metrica_8_bits_7_dim_otsu" folder
np.save(os.path.join(carpeta, 'RSA_vec.npy'), RSA_vec)
np.save(os.path.join(carpeta, 'HBA_vec.npy'), HBA_vec)
# ... (y así sucesivamente para todos los algoritmos y métricas)
</code></pre>

<hr>

<h2>🖼️ Dataset de Imágenes Médicas</h2>
<p>
    A continuación se muestra una galería de las imágenes médicas utilizadas para las pruebas de segmentación.
</p>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
    <img src="img/92 (27).png" width="150" alt="92 (27)">
    <img src="img/92 (29).png" width="150" alt="92 (29)">
    <img src="img/94 (31).png" width="150" alt="94 (31)">
    <img src="img/IM000001.png" width="150" alt="IM000001">
    <img src="img/IM000003.png" width="150" alt="IM000003">
    <img src="img/IM000004.png" width="150" alt="IM000004">
    <img src="img/IM000016.png" width="150" alt="IM000016">
    <img src="img/IM000017.png" width="150" alt="IM000017">
    <img src="img/IM000018.png" width="150" alt="IM000018">
    <img src="img/Te-gl_0028.png" width="150" alt="Te-gl_0028">
    <img src="img/Te-gl_0072.png" width="150" alt="Te-gl_0072">
    <img src="img/Te-gl_0241.png" width="150" alt="Te-gl_0241">
    <img src="img/Te-gl_0277.png" width="150" alt="Te-gl_0277">
    <img src="img/Te-me_0043.png" width="150" alt="Te-me_0043">
    <img src="img/Te-me_0147.png" width="150" alt="Te-me_0147">
    <img src="img/Te-me_0155.png" width="150" alt="Te-me_0155">
    <img src="img/Te-me_0239.png" width="150" alt="Te-me_0239">
    <img src="img/Te-piTr_0008.png" width="150" alt="Te-piTr_0008">
    <img src="img/Te-pi_0025.png" width="150" alt="Te-pi_0025">
    <img src="img/Te-pi_0242.png" width="150" alt="Te-pi_0242">
</div>

</body>
</html>
