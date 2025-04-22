import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
import pickle
import json
import os

# Define constants
MODEL_PATH = 'melanoma_detection_finetuned_model.h5'
CLASS_NAMES = ['Benign', 'Malignant']
SAVE_DIR = 'evaluation_results'

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = keras.models.load_model(MODEL_PATH)

# Load validation data
# Replace this with your code to load validation data
def load_validation_data():
    """Load validation data and compute predictions."""
    # Example code - replace with your validation data loading
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = datagen.flow_from_directory(
        './002/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    # Get ground truth labels
    y_true = val_generator.classes
    
    # Get predictions
    y_pred_proba = model.predict(val_generator)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return y_true, y_pred, y_pred_proba

# Plot and save training history
def plot_and_save_history(history):
    """Plot training history metrics and save figures."""
    # Convert to dict if it's a keras History object
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy') 
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'accuracy_plot.png'), dpi=300)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_plot.png'), dpi=300)
    
    # Plot precision and recall
    plt.figure(figsize=(10, 6))
    plt.plot(history['precision'], label='Training Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.plot(history['recall'], label='Training Recall')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.title('Precision and Recall')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'precision_recall_plot.png'), dpi=300)
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(history['auc'], label='Training AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('Area Under Curve (AUC)')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'auc_plot.png'), dpi=300)
    
    # Save final metrics
    final_metrics = {
        "accuracy": float(history['val_accuracy'].values[-1]),
        "precision": float(history['val_precision'].values[-1]),
        "recall": float(history['val_recall'].values[-1]),
        "auc": float(history['val_auc'].values[-1]),
        "loss": float(history['val_loss'].values[-1])
    }
    
    with open(os.path.join(SAVE_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    return final_metrics

# Plot and save precision-recall curve
def plot_and_save_pr_curve(y_true, y_pred_proba):
    """Generate and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b', label=f'PR AUC = {pr_auc:.3f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'precision_recall_curve.png'), dpi=300)
    
    # Save PR curve data
    pr_data = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist() if thresholds.size > 0 else [],
        "pr_auc": float(pr_auc)
    }
    
    with open(os.path.join(SAVE_DIR, 'pr_curve_data.json'), 'w') as f:
        json.dump(pr_data, f, indent=4)
    
    return pr_auc

# Plot and save ROC curve
def plot_and_save_roc_curve(y_true, y_pred_proba):
    """Generate and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b', label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'), dpi=300)
    
    # Save ROC curve data
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(roc_auc)
    }
    
    with open(os.path.join(SAVE_DIR, 'roc_curve_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=4)
    
    return roc_auc

# Plot and save confusion matrix
def plot_and_save_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = [0, 1]
    plt.xticks(tick_marks, CLASS_NAMES)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    
    # Save confusion matrix data
    cm_data = {
        "matrix": cm.tolist(),
        "class_names": CLASS_NAMES
    }
    
    with open(os.path.join(SAVE_DIR, 'confusion_matrix_data.json'), 'w') as f:
        json.dump(cm_data, f, indent=4)
    
    # Generate and save classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    with open(os.path.join(SAVE_DIR, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return cm, report

# Main execution
def main():
    # Load history from pickle file if available, otherwise create dummy history
    try:
        with open('model_history.pkl', 'rb') as f:
            history = pickle.load(f)
        print("Loaded training history from file.")
    except FileNotFoundError:
        print("History file not found. Using placeholder data.")
        # Create dummy history with the keys you mentioned
        history = {
            'accuracy': [0.7, 0.8, 0.85],
            'precision': [0.7, 0.75, 0.8],
            'recall': [0.65, 0.7, 0.75],
            'auc': [0.75, 0.8, 0.85],
            'loss': [0.5, 0.4, 0.3],
            'val_accuracy': [0.65, 0.7, 0.75],
            'val_precision': [0.65, 0.7, 0.75],
            'val_recall': [0.6, 0.65, 0.7],
            'val_auc': [0.7, 0.75, 0.8],
            'val_loss': [0.55, 0.5, 0.45]
        }
    
    # Plot and save training history
    final_metrics = plot_and_save_history(history)
    print("Training history plots saved.")
    
    # Load validation data and get predictions
    try:
        print("Loading validation data and generating predictions...")
        y_true, y_pred, y_pred_proba = load_validation_data()
        
        # Plot and save precision-recall curve
        pr_auc = plot_and_save_pr_curve(y_true, y_pred_proba)
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        
        # Plot and save ROC curve
        roc_auc = plot_and_save_roc_curve(y_true, y_pred_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Plot and save confusion matrix
        cm, report = plot_and_save_confusion_matrix(y_true, y_pred)
        print("Confusion matrix and classification report saved.")
        
        # Print summary of results
        print("\nEvaluation Results Summary:")
        print(f"Final Validation Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final Validation Precision: {final_metrics['precision']:.4f}")
        print(f"Final Validation Recall: {final_metrics['recall']:.4f}")
        print(f"Final Validation AUC: {final_metrics['auc']:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"Error during validation data evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()