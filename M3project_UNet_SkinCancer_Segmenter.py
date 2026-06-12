# -*- coding: utf-8 -*-
"""Segmentación de Cáncer de Piel con U-Net - Optimizado con guardado de resultados (sin clasificación de subtipos)"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN
work_dir = r"C:\Users\diegm\OneDrive\Documentos\Tec\Procesamiento_imagenes"  
images_dir = os.path.join(work_dir, "Images")
masks_dir  = os.path.join(work_dir, "Masks")
output_dir = os.path.join(work_dir, "Resultados")
os.makedirs(output_dir, exist_ok=True)

# Parámetros de rendimiento
IMG_SIZE = (128, 128)          # Reduce tamaño para acelerar (puedes cambiarlo a 256 si tienes GPU)
BATCH_SIZE = 8
EPOCHS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Función de pérdida (elige una: 'bce', 'dice', 'focal', 'combined') ---
LOSS_TYPE = 'combined'   # Recomendado para manejar desbalance de clases

# 1. CARGAR DATOS OPTIMIZADA
def load_dataset_optimized(images_dir, masks_dir, img_size=IMG_SIZE):
    """Carga imágenes y máscaras, devuelve arrays numpy y lista de nombres de archivo"""
    image_paths, mask_paths, names = [], [], []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                img_full = os.path.join(root, f)
                mask_name = os.path.splitext(f)[0] + ".png"
                mask_full = None
                for m_root, _, m_files in os.walk(masks_dir):
                    if mask_name in m_files:
                        mask_full = os.path.join(m_root, mask_name)
                        break
                if mask_full and os.path.exists(mask_full):
                    image_paths.append(img_full)
                    mask_paths.append(mask_full)
                    names.append(os.path.splitext(f)[0])
    print(f" Encontrados {len(image_paths)} pares imagen-máscara.")
    if len(image_paths) == 0:
        return np.array([]), np.array([]), []

    images, masks = [], []
    for imp, map in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Cargando datos"):
        img = cv2.imread(imp)
        if img is None:
            print(f" No se pudo leer: {imp}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size) / 255.0

        mask = cv2.imread(map, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f" No se pudo leer: {map}")
            continue
        mask = cv2.resize(mask, img_size)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, -1)

        images.append(img)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32), names

print(" Cargando dataset...")
X, y, img_names = load_dataset_optimized(images_dir, masks_dir)
if X.size == 0:
    raise RuntimeError("No se cargaron datos. Revisa rutas y archivos.")

# Dividir entrenamiento/validación conservando los nombres
X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
    X, y, img_names, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# 2. FUNCIONES DE PÉRDIDA AVANZADAS
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1 + tf.keras.backend.epsilon()) +
                                      (1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Selección de pérdida
if LOSS_TYPE == 'bce':
    loss_fn = 'binary_crossentropy'
elif LOSS_TYPE == 'dice':
    loss_fn = dice_loss
elif LOSS_TYPE == 'focal':
    loss_fn = focal_loss()
elif LOSS_TYPE == 'combined':
    loss_fn = combined_loss
else:
    loss_fn = 'binary_crossentropy'

print(f"🔧 Función de pérdida utilizada: {LOSS_TYPE}")

# 3. MODELO U-Net LIGERO
def unet_light(input_size=(IMG_SIZE[0], IMG_SIZE[1], 3)):
    inputs = keras.Input(shape=input_size)
    # Contracting path (filtros reducidos para acelerar)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(c5)

    # Expansive path
    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([c4, u6])
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([c3, u7])
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([c2, u8])
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([c1, u9])
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss_fn,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    return model

# 4. ENTRENAMIENTO
model = unet_light()
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_model.keras'), save_best_only=True)
]

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1)

# 5. MÉTRICAS DETALLADAS Y GUARDADO EN EXCEL
y_pred_prob = model.predict(X_val)
y_pred_bin = (y_pred_prob > 0.5).astype(np.float32)

def per_class_iou(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    # Clase tumor (1)
    inter1 = np.logical_and(y_true_flat == 1, y_pred_flat == 1).sum()
    union1 = np.logical_or(y_true_flat == 1, y_pred_flat == 1).sum()
    iou_tumor = inter1 / (union1 + 1e-6)
    # Clase fondo (0)
    inter0 = np.logical_and(y_true_flat == 0, y_pred_flat == 0).sum()
    union0 = np.logical_or(y_true_flat == 0, y_pred_flat == 0).sum()
    iou_bg = inter0 / (union0 + 1e-6)
    return iou_tumor, iou_bg

# Calcular métricas por imagen
results = []
for i in range(len(y_val)):
    yt = y_val[i].squeeze()
    yp = y_pred_bin[i].squeeze()
    iou_t, iou_b = per_class_iou(yt, yp)
    dice = 2 * (yt * yp).sum() / (yt.sum() + yp.sum() + 1e-6)
    area_px = np.sum(yp)
    results.append({
        'imagen': names_val[i],
        'IoU_tumor': iou_t,
        'IoU_fondo': iou_b,
        'Dice': dice,
        'area_tumor_px': area_px
        # Si conoces la resolución (μm/pixel), añade una columna: 'area_tumor_um2': area_px * escala**2
    })

df_results = pd.DataFrame(results)
excel_path = os.path.join(output_dir, f'resultados_segmentacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
df_results.to_excel(excel_path, index=False)
print(f"\n Resultados guardados en: {excel_path}")

# Estadísticas globales
mean_iou_tumor = df_results['IoU_tumor'].mean()
std_iou_tumor = df_results['IoU_tumor'].std()
mean_dice = df_results['Dice'].mean()
mean_area = df_results['area_tumor_px'].mean()

print("\n" + "="*60)
print(" MÉTRICAS DE SEGMENTACIÓN (Objetivo 1)")
print(f"  IoU promedio (tumor):   {mean_iou_tumor:.4f} ± {std_iou_tumor:.4f}")
print(f"  IoU promedio (fondo):   {df_results['IoU_fondo'].mean():.4f} ± {df_results['IoU_fondo'].std():.4f}")
print(f"  Dice promedio:          {mean_dice:.4f}")
print(f"  Área promedio del tumor: {mean_area:.1f} píxeles")
print("="*60)

# Guardar historial de entrenamiento
history_df = pd.DataFrame(history.history)
history_csv = os.path.join(output_dir, f'historial_entrenamiento_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
history_df.to_csv(history_csv, index=False)
print(f" Historial de entrenamiento guardado en: {history_csv}")

# 6. GUARDAR FIGURAS DE PREDICCIONES
n_samples = min(5, len(X_val))
indices = np.random.choice(len(X_val), n_samples, replace=False)

for idx in indices:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(X_val[idx])
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(y_val[idx].squeeze(), cmap='gray')
    axes[1].set_title('Máscara real')
    axes[1].axis('off')
    axes[2].imshow((y_pred_bin[idx].squeeze() > 0.5).astype(np.uint8), cmap='gray')
    axes[2].set_title('Predicción')
    axes[2].axis('off')
    plt.tight_layout()
    fig_name = f"pred_{names_val[idx]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" Figura guardada: {fig_path}")

print("\n Análisis completado. Todos los resultados se encuentran en:")
print(f"   {output_dir}")