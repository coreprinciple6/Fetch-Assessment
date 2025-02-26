import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os, random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataset, num_epochs=2, batch_size=32, 
                learning_rate=1e-4, device=DEVICE, random_seed=42,
                freeze_transformer=True, freeze_taskA=False, freeze_taskB=False,
                alpha=0.6, beta=0.8):
    """
    Train the multitask sentence transformer model.
    Args:   
        model (nn.Module): The multitask model to train.
        dataset (Dataset): The dataset providing sentences and labels.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to run training on
        random_seed (int): Random seed for reproducibility.
        freeze_transformer (bool): Whether to freeze the transformer weights.
        freeze_taskA (bool): Whether to freeze the taskA classifier weights.
        freeze_taskB (bool): Whether to freeze the taskB classifier weights.
        alpha (float): Weight for task A loss in the combined loss.
        beta (float): Weight for task B loss in the combined loss.
    """

    # Freeze components based on parameters
    if freeze_transformer: # Freeze the transformer weights
        for param in model.transformer.parameters():
            param.requires_grad = False
    
    if freeze_taskA: # Freeze the taskA classifier weights
        for param in model.taskA_classifier.parameters():
            param.requires_grad = False
        alpha = alpha//2 # Reduce the weight of taskA loss
    
    if freeze_taskB: # Freeze the taskB classifier weights
        for param in model.taskB_classifier.parameters():
            param.requires_grad = False
        beta = beta//3 # Reduce the weight of taskB loss

    random.seed(random_seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    #  Necessary components for training
    criterion = nn.CrossEntropyLoss() #Loss function for both tasks: CrossEntropyLoss.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    metrics = {
        'loss': [],
        'taskA': {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': []
        },
        'taskB': {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': []
        }
    }
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_samples = 0
        
        all_taskA_preds = []
        all_taskA_labels = []
        all_taskB_preds = []
        all_taskB_labels = []
        
        for batch in dataloader:
            sentences = batch["sentence"]
            taskA_labels = batch["taskA_label"].to(device)
            taskB_labels = batch["taskB_label"].to(device)
            
            # Tokenize the batch using the model's tokenizer
            encoded_input = model.tokenizer(sentences,
                                          padding=True,
                                          truncation=True,
                                          return_tensors='pt')
                      
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            optimizer.zero_grad()
            # Forward pass with multitask output: embeddings, taskA logits, taskB logits.
            _, taskA_logits, taskB_logits = model(input_ids, attention_mask, multitask=True)
            
            # Compute losses for both tasks
            lossA = criterion(taskA_logits, taskA_labels)
            lossB = criterion(taskB_logits, taskB_labels)
            loss = alpha * lossA + beta * lossB # Combined loss
            
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and predictions for metrics calculation
            batch_size_actual = input_ids.size(0)
            epoch_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # Store predictions and ground truth labels for metrics calculation
            taskA_preds = torch.argmax(taskA_logits, dim=1)
            taskB_preds = torch.argmax(taskB_logits, dim=1)
            
            all_taskA_preds.extend(taskA_preds.cpu().numpy())
            all_taskA_labels.extend(taskA_labels.cpu().numpy())
            all_taskB_preds.extend(taskB_preds.cpu().numpy())
            all_taskB_labels.extend(taskB_labels.cpu().numpy())
        
        # Calculate metrics for the epoch
        avg_loss = epoch_loss / total_samples
        
        # Task A metrics
        taskA_precision, taskA_recall, taskA_f1, _ = precision_recall_fscore_support(
            all_taskA_labels, all_taskA_preds, average='weighted',zero_division=0)
        taskA_acc = sum(1 for pred, label in zip(all_taskA_preds, all_taskA_labels) if pred == label) / len(all_taskA_labels)
        taskA_cm = confusion_matrix(all_taskA_labels, all_taskA_preds)
        
        # Task B metrics
        taskB_precision, taskB_recall, taskB_f1, _ = precision_recall_fscore_support(
            all_taskB_labels, all_taskB_preds, average='weighted', zero_division=0)
        taskB_acc = sum(1 for pred, label in zip(all_taskB_preds, all_taskB_labels) if pred == label) / len(all_taskB_labels)
        taskB_cm = confusion_matrix(all_taskB_labels, all_taskB_preds)
        
        # Store metrics
        metrics['loss'].append(avg_loss)
        metrics['taskA']['accuracy'].append(taskA_acc)
        metrics['taskA']['precision'].append(taskA_precision)
        metrics['taskA']['recall'].append(taskA_recall)
        metrics['taskA']['f1'].append(taskA_f1)
        metrics['taskA']['confusion_matrix'].append(taskA_cm)
        
        metrics['taskB']['accuracy'].append(taskB_acc)
        metrics['taskB']['precision'].append(taskB_precision)
        metrics['taskB']['recall'].append(taskB_recall)
        metrics['taskB']['f1'].append(taskB_f1)
        metrics['taskB']['confusion_matrix'].append(taskB_cm)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Task A - Acc: {taskA_acc:.4f}, Prec: {taskA_precision:.4f}, Rec: {taskA_recall:.4f}, F1: {taskA_f1:.4f}")
        print(f"  Task B - Acc: {taskB_acc:.4f}, Prec: {taskB_precision:.4f}, Rec: {taskB_recall:.4f}, F1: {taskB_f1:.4f}")
    
    return metrics

def visualize_metrics(metrics, save_dir='./graphs',type='train'):
    """
    metrics visualizations.
    Args:
        metrics (dict): Dictionary containing metrics
        save_dir (str): Directory to save the plots
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot loss
    if type == 'train':
        epochs = range(1, len(metrics['loss']) + 1)
    
        # 1. Plot the combined loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics['loss'], 'o-', linewidth=2, markersize=8)
        plt.title('Training Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{type}_combined_loss.png")
        plt.close()
    
        # 2. Plot accuracy for both tasks
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, metrics['taskA']['accuracy'], 'o-', label='Task A', linewidth=2, markersize=8)
        plt.plot(epochs, metrics['taskB']['accuracy'], 's-', label='Task B', linewidth=2, markersize=8)
        plt.title('Accuracy by Task', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{type}_accuracy_comparison.png")
        plt.close()
    
        # 3. Plot all metrics for Task A
        plt.figure(figsize=(14, 8))
        plt.plot(epochs, metrics['taskA']['accuracy'], 'o-', label='Accuracy', linewidth=2)
        plt.plot(epochs, metrics['taskA']['precision'], 's-', label='Precision', linewidth=2)
        plt.plot(epochs, metrics['taskA']['recall'], '^-', label='Recall', linewidth=2)
        plt.plot(epochs, metrics['taskA']['f1'], 'd-', label='F1 Score', linewidth=2)
        plt.title('Task A Metrics', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{type}_taskA_metrics.png")
        plt.close()
        
        # 4. Plot all metrics for Task B
        plt.figure(figsize=(14, 8))
        plt.plot(epochs, metrics['taskB']['accuracy'], 'o-', label='Accuracy', linewidth=2)
        plt.plot(epochs, metrics['taskB']['precision'], 's-', label='Precision', linewidth=2)
        plt.plot(epochs, metrics['taskB']['recall'], '^-', label='Recall', linewidth=2)
        plt.plot(epochs, metrics['taskB']['f1'], 'd-', label='F1 Score', linewidth=2)
        plt.title('Task B Metrics', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{type}_taskB_metrics.png")
        plt.close()

    if type == 'test':
        # 7. Plot ROC curves for both tasks
        plt.figure(figsize=(14, 8))
        # add a reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Reference', linewidth=2)
        plt.plot(metrics['taskA']['fpr'], metrics['taskA']['tpr'], 'o-', label='Task A', linewidth=4)
        plt.plot(metrics['taskB']['fpr'], metrics['taskB']['tpr'], 's-', label='Task B', linewidth=2)
        plt.title('ROC Curves', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{type}_roc_curves.png")
        plt.close()


def evaluate_model(model, test_dataset, batch_size=32, device=DEVICE):
    """
    Evaluate the multitask sentence transformer model on test data.
    Args:
        model (nn.Module): The trained multitask model to evaluate.
        test_dataset (Dataset): The test dataset providing sentences and labels.
        batch_size (int): Batch size for evaluation.
        device (str): Device to run evaluation on ('cpu' or 'cuda').
    """
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    
    all_taskA_preds = []
    all_taskA_labels = []
    all_taskA_probs = []
    all_taskB_preds = []
    all_taskB_labels = []
    all_taskB_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            sentences = batch["sentence"]
            taskA_labels = batch["taskA_label"].to(device)
            taskB_labels = batch["taskB_label"].to(device)
            
            # Tokenize the batch using the model's tokenizer
            encoded_input = model.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            # Forward pass with multitask output
            _, taskA_logits, taskB_logits = model(input_ids, attention_mask, multitask=True)
            
            # Get predictions and probabilities
            taskA_preds = torch.argmax(taskA_logits, dim=1)
            taskB_preds = torch.argmax(taskB_logits, dim=1)
            
            # Convert logits to probabilities using softmax
            taskA_probs = torch.nn.functional.softmax(taskA_logits, dim=1)
            taskB_probs = torch.nn.functional.softmax(taskB_logits, dim=1)
            
            # Store predictions, ground truth, and probabilities
            all_taskA_preds.extend(taskA_preds.cpu().numpy())
            all_taskA_labels.extend(taskA_labels.cpu().numpy())
            all_taskA_probs.append(taskA_probs.cpu().numpy())
            
            all_taskB_preds.extend(taskB_preds.cpu().numpy())
            all_taskB_labels.extend(taskB_labels.cpu().numpy())
            all_taskB_probs.append(taskB_probs.cpu().numpy())
    
    # Convert probability lists to numpy arrays
    all_taskA_probs = np.vstack(all_taskA_probs)
    all_taskB_probs = np.vstack(all_taskB_probs)
    
    # Convert label lists to numpy arrays for metric calculation
    all_taskA_labels = np.array(all_taskA_labels)
    all_taskB_labels = np.array(all_taskB_labels)
    all_taskA_preds = np.array(all_taskA_preds)
    all_taskB_preds = np.array(all_taskB_preds)
    
    # Get unique classes for both tasks
    taskA_classes = np.unique(all_taskA_labels)
    taskB_classes = np.unique(all_taskB_labels)

    # calculate roc curve for task A
    try:
        if len(taskA_classes) == 2:  # Binary classification
            taskA_fpr, taskA_tpr, threshold = roc_curve(all_taskA_labels, all_taskA_probs[:, 1])
            taskA_roc_auc = auc(taskA_fpr, taskA_tpr)
        else:  # Multiclass
            taskA_fpr, taskA_tpr, threshold = roc_curve(
                np.eye(len(taskA_classes))[all_taskA_labels],
                all_taskA_probs,
                multi_class='ovr'
            )
            taskA_roc_auc = auc(taskA_fpr, taskA_tpr)
    except:
        taskA_fpr = None
        taskA_tpr = None

    # calculate roc curve for task B
    try:
        if len(taskB_classes) == 2:  # Binary classification
            taskB_fpr, taskB_tpr, threshold = roc_curve(all_taskB_labels, all_taskB_probs[:, 1])
            taskB_roc_auc = auc(taskB_fpr, taskB_tpr)
        else:  # Multiclass
            taskB_fpr, taskB_tpr, threshold = roc_curve(
                np.eye(len(taskB_classes))[all_taskB_labels],
                all_taskB_probs,
                multi_class='ovr'
            )
            taskB_roc_auc = auc(taskB_fpr, taskB_tpr)
    except:
        taskB_fpr = None
        taskB_tpr = None

    
    # Task A metrics
    taskA_precision, taskA_recall, taskA_f1, _ = precision_recall_fscore_support(
        all_taskA_labels, all_taskA_preds, average='weighted', zero_division=0
    )

    taskA_acc = (all_taskA_preds == all_taskA_labels).mean()
    taskA_cm = confusion_matrix(all_taskA_labels, all_taskA_preds)

    # Task B metrics
    taskB_precision, taskB_recall, taskB_f1, _ = precision_recall_fscore_support(
        all_taskB_labels, all_taskB_preds, average='weighted', zero_division=0
    )
    taskB_acc = (all_taskB_preds == all_taskB_labels).mean()
    taskB_cm = confusion_matrix(all_taskB_labels, all_taskB_preds)

    
    # Store all metrics in a nested dictionary
    metrics = {
        'overall': {
            'taskA_accuracy': taskA_acc,
            'taskB_accuracy': taskB_acc,
            'combined_accuracy': (taskA_acc + taskB_acc) / 2
        },
        'taskA': {
            'accuracy': taskA_acc,
            'precision': taskA_precision,
            'recall': taskA_recall,
            'f1': taskA_f1,
            'auc': taskA_roc_auc,
            'fpr': taskA_fpr,
            'tpr': taskA_tpr,
            'confusion_matrix': taskA_cm,

        },
        'taskB': {
            'accuracy': taskB_acc,
            'precision': taskB_precision,
            'recall': taskB_recall,
            'f1': taskB_f1,
            'auc': taskB_roc_auc,
            'fpr': taskB_fpr,
            'tpr': taskB_tpr,
            'confusion_matrix': taskB_cm,
        }
    }
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Task A - Accuracy: {taskA_acc:.4f}, Precision: {taskA_precision:.4f}, Recall: {taskA_recall:.4f}, F1: {taskA_f1:.4f}")
    if taskA_roc_auc:
        print(f"Task A - AUC: {taskA_roc_auc:.4f}")
    print(f"Task B - Accuracy: {taskB_acc:.4f}, Precision: {taskB_precision:.4f}, Recall: {taskB_recall:.4f}, F1: {taskB_f1:.4f}")
    if taskB_roc_auc:
        print(f"Task B - AUC: {taskB_roc_auc:.4f}")
    print(f"Combined Accuracy: {(taskA_acc + taskB_acc) / 2:.4f}")
    
    return metrics
    

