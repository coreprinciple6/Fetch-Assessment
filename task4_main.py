import torch
from task2_model import MultiTaskSentenceTransformer
from task4_dataset import MultiTaskDataset, create_train_test_datasets
from task4_train import train_model, evaluate_model, visualize_metrics

# Hyperparameters
EMBED_DIM = 512
PRETRAINED_MODEL = 'distilbert-base-uncased' #'bert-base-uncased'
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 6
DATASET_SIZE = 1000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

print('Initializing the model...')
model = MultiTaskSentenceTransformer(model_name=PRETRAINED_MODEL, embedding_dim=EMBED_DIM)
model.to(DEVICE)

print('Preparing Dataset...')
train_dataset, test_dataset = create_train_test_datasets(total_samples=DATASET_SIZE, test_ratio=0.2)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

print("\nTrain class distribution:")
print(train_dataset.get_class_distribution())
print("\nTest class distribution:")
print(test_dataset.get_class_distribution())

# Example sentence from each class in the training set
class_examples = {} # (taskA, taskB) -> sentence
for i in range(len(train_dataset)):
    taskA = train_dataset.taskA_labels[i] # 0: Technology, 1: Entertainment
    taskB = train_dataset.taskB_labels[i] # 0: Negative, 1: Positive
    if (taskA, taskB) not in class_examples: # Only store the first example from each class
        class_examples[(taskA, taskB)] = train_dataset.sentences[i]
        if len(class_examples) == 4:
            break

print("\nExample sentences from training set:")
for (taskA, taskB), sentence in class_examples.items(): 
    domain = "Technology" if taskA == 0 else "Entertainment"
    sentiment = "Positive" if taskB == 1 else "Negative"
    print(f"{domain} & {sentiment}: {sentence}")


# Train the model
print('\nTraining the model...')
train_metrics = train_model(model, train_dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE, device=DEVICE,
            freeze_transformer=True, freeze_taskA=False, freeze_taskB=False)
print('Training complete!')
visualize_metrics(train_metrics, type='train')

# Evaluate the trained model
print('\nEvaluating the trained model...')
test_metrics = evaluate_model(model, test_dataset, batch_size=BATCH_SIZE, device=DEVICE)
print('Evaluation complete!')
visualize_metrics(test_metrics, type='test')

print(f"All plots saved to .graphs/")