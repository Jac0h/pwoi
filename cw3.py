import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.color import rgb2gray
from skimage.feature.texture import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# === 1. Extract patches ===
def extract_patches(input_dir, output_dir, patch_size=(128, 128)):
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        save_path = os.path.join(output_dir, category)
        os.makedirs(save_path, exist_ok=True)

        for idx, file_name in enumerate(os.listdir(category_path)):
            img_path = os.path.join(category_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            for y in range(0, h - patch_size[1], patch_size[1]):
                for x in range(0, w - patch_size[0], patch_size[0]):
                    patch = img[y:y+patch_size[1], x:x+patch_size[0]]
                    patch_name = f"{category}_{idx}_{x}_{y}.png"
                    cv2.imwrite(os.path.join(save_path, patch_name), patch)

# === 2. Extract features using GLCM ===
def extract_features_from_patches(patches_dir):
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features_list = []

    for category in os.listdir(patches_dir):
        cat_path = os.path.join(patches_dir, category)
        if not os.path.isdir(cat_path):
            continue

        for file_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = rgb2gray(img)
            gray = (gray * 63).astype(np.uint8)  # 5-bit = 64 levels

            glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

            feats = []
            for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
                val = graycoprops(glcm, prop).flatten()
                feats.extend(val)

            feats.append(category)
            features_list.append(feats)

    col_names = [f"{prop}_{d}_{int(np.rad2deg(a))}"
                 for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
                 for d in distances for a in angles]
    col_names.append("label")

    return pd.DataFrame(features_list, columns=col_names)

# === 3. Classify features ===
def classify_texture_features(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1)
    y = LabelEncoder().fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'Dokładność klasyfikatora: {acc:.2%}')


extract_patches('zdj_cw3', 'patches')
df = extract_features_from_patches('patches')
df.to_csv('features.csv', index=False)
classify_texture_features('features.csv')
