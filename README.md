# TB Chest Xray Classification

# Tuberculosis Detection from Chest X-rays using Deep Learning

## 📌 Overview
This project aims to classify chest X-ray images as Tuberculosis (TB) or Normal using deep learning techniques.

## 🎯 Objective
To develop an AI-based system for early TB detection to support clinical diagnosis.

## 🧠 Methodology

### Data
- Chest X-ray images (TB vs Normal)

### Model
- Convolutional Neural Network (CNN)
- Transfer learning (ResNet18)

### Workflow
1. Data loading and preprocessing
2. Model training
3. Evaluation using accuracy and confusion matrix

## 📊 Results
The ResNet18 model was trained for 10 epochs. To maintain clinical relevance, I prioritized Sensitivity (Recall) to ensure minimal false negatives in TB detection.
- Check the results section for the images. 

## 🧠 Discussion & Limitations
While the ResNet18 model shows high accuracy, medical deployment requires further validation. Limitations include:
- Dataset Bias: The model may perform differently on X-rays from different hardware manufacturers.
- Clinical Integration: Future work should focus on integrating this into a DICOM viewer for real-time radiologist assistance.
- Next Steps: Implementing Radiomics-based feature selection (LASSO) to compare deep learning features with hand-crafted texture features.

## 👤 Author
Michael Opoku Mbroh

## 🌍 Vision
To deploy AI-powered diagnostic tools in Africa to improve healthcare access.
