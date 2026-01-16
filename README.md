# Left Supraclinoid Internal Carotid Artery Aneurysm Detection

## 1. Project Overview & Clinical Necessity

Intracranial aneurysms are focal dilations of cerebral arteries that pose a significant health risk. Rupture of these aneurysms results in subarachnoid hemorrhage (SAH), a catastrophic event with high mortality and morbidity rates.

The **Left Supraclinoid Internal Carotid Artery (ICA)** is a critical anatomical location for aneurysm formation. Located near the optic nerve and other vital structures, aneurysms here can be asymptomatic until rupture or cause compression symptoms (e.g., visual deficits).

**Why an Automated Detection Model is Necessary:**
* **High Miss Rate:** Small, unruptured aneurysms are often incidental findings and can be easily overlooked on routine scans due to their size and complex vascular geometry.
* **Workload Efficiency:** Radiologists face increasing imaging volumes. An automated pre-screening tool can prioritize high-risk scans, reducing time-to-diagnosis.
* **Proactive Intervention:** Early detection allows for elective endovascular or surgical intervention (coiling/clipping) before a fatal rupture occurs.

## 2. Dataset Description

This project utilizes data from the **RSNA Intracranial Aneurysm Detection** competition hosted on Kaggle. The dataset is one of the largest publicly available collections of annotated intracranial imaging.

* **Source:** Radiological Society of North America (RSNA), ASNR, SNIS, and ESNR.
* **Modalities:** The dataset comprises thousands of 3D imaging series, primarily **Computed Tomography Angiography (CTA)** and **Magnetic Resonance Angiography (MRA)**.
* **Structure:**
    * **DICOM Series:** Volumetric scan data containing multiple 2D slices per patient.
    * **Labels:** Binary classifications for aneurysm presence across 13 specific anatomical locations.
    * **Localizers:** A `train_localized.csv` file provides **$x, y$ coordinates** pinpointing aneurysms within specific slices. Note that while $z$-axis depth can be inferred from frame numbers, explicit $z$ coordinates were not provided in the label file and were not utilized for model input coordinates.

## 3. Work Done & Technical Analysis

The project workflow encompasses Exploratory Data Analysis (EDA), volumetric preprocessing, and a Transfer Learning approach for binary classification.

### **A. Distribution Analysis**
* **Target Identification:** Statistical analysis of metadata identified the **Left Supraclinoid Internal Carotid Artery** as a high-frequency target (approx. 330 positive cases), validating it as a priority for model training.

### **B. Volumetric Normalization**
* **Depth Profiling:** Developed a normalization routine to link `SOPInstanceUID` with `SeriesInstanceUID`.
* **Heuristic Location:** Calculated the normalized depth ratio of aneurysm centroids, determining that relevant vascular features cluster around the 48th percentile (0.48 ratio) of the 3D stack. This heuristic guided Region-of-Interest (ROI) extraction.

### **C. Radiological Visualization**
* **Windowing:** Implemented a DICOM pipeline using `pydicom` to extract multi-slice windows (e.g., $\pm 2$ frames) around calculated anatomical centers, effectively visualizing healthy vs. pathological vascular architecture.

### **D. Model Architecture & Training**
The core detection system is built upon a **Transfer Learning** framework using **MobileNetV2**.

* **Architecture:**
    * **Backbone:** MobileNetV2 pre-trained on ImageNet weights, initially frozen to serve as a feature extractor.
    * **Input Shape:** $(160, 160, 3)$ RGB images.
    * **Augmentation Pipeline:** Integrated `tf.keras.layers` for `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, and `RandomBrightness` to mitigate overfitting on the medical imaging data.
    * **Classification Head:** A custom top-end consisting of a `GlobalAveragePooling2D` layer, followed by a `Dropout` layer (0.2) for regularization, and a final `Dense` layer (1 unit) for binary classification.
* **Training Strategy:**
    * **Optimizer:** Adam optimizer starting with a learning rate of $1e^{-4}$.
    * **Loss Function:** `BinaryCrossentropy` (from logits).
    * **Callbacks:** Implemented `EarlyStopping` (monitoring `val_loss`, patience=5) to prevent overfitting and restore best weights.
    * **Fine-Tuning:** A secondary training phase involved unfreezing the top layers of MobileNetV2 (from layer 100 onwards) and re-training with a reduced learning rate ($1e^{-5}$) to adapt high-level features specifically to intracranial vascular structures.
