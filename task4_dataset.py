import random
import torch
from torch.utils.data import Dataset
import numpy as np
import re

class MultiTaskDataset(Dataset):
    def __init__(self, num_samples=200, random_seed=42):
        """
        Create dataset for multitask training with balanced class distribution.
        Each sample contains:
            - A sentence (str)
            - A label for Task A (0: technology, 1: entertainment)
            - A label for Task B (0: negative, 1: positive)
        Args:
            num_samples (int): Total number of samples to generate (should be divisible by 4)
            random_seed (int): Random seed for reproducibility
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        if num_samples % 4 != 0:
            raise ValueError("num_samples must be divisible by 4 for balanced category assignment.")
            
        self.num_samples = num_samples
        samples_per_category = num_samples // 4
        
        self.sentences = []
        self.taskA_labels = []  # 0 for technology, 1 for entertainment
        self.taskB_labels = []  # 0 for negative, 1 for positive

        # Base templates for each category
        self.tech_positive_templates = [
            "The {tech_product} is incredibly {positive_adj} and {positive_adj}.",
            "This {adjective} {tech_product} is {positive_verb} the {tech_domain}.",
            "I am {positive_emotion} with the latest {tech_product}; it's {positive_adj}.",
            "The {tech_domain}-driven {tech_product} delivers {positive_adj} performance.",
            "The new {tech_product} is a {positive_noun} in the tech world."
        ]
        
        self.tech_negative_templates = [
            "The {tech_product} is {negative_adj} designed and {negative_adj} to use.",
            "The latest update {negative_verb} the {tech_product}'s performance.",
            "I am {negative_emotion} with the product's {negative_adj} technology.",
            "The {tech_product} frequently {negative_verb} and is very {negative_adj}.",
            "The new {tech_product} is a major {negative_noun} and waste of money."
        ]
        
        self.entertainment_positive_templates = [
            "The {media_type} was an absolute {positive_noun} with {positive_adj} {media_aspect}.",
            "I {positive_emotion} the {entertainment_event}; it was {positive_adj} and {positive_adj}.",
            "The {entertainment_event} left me {positive_emotion} and {positive_emotion}.",
            "The {media_type} is {positive_adv} {positive_adj} and {positive_adj}.",
            "The {media_type} was {positive_adj} and {positive_adv} {positive_verb}."
        ]
        
        self.entertainment_negative_templates = [
            "The {media_type} was {negative_adj} and lacked any {positive_adj} moments.",
            "I was not {positive_emotion} by the {negative_adj} {media_aspect}.",
            "The {entertainment_event} felt {negative_adj} and very {negative_adj}.",
            "The {media_type} was {negative_adj} and failed to {positive_verb} my attention.",
            "The {media_type} was a major {negative_noun} with {negative_adj} {media_aspect}."
        ]

        # Word banks for template filling
        self.word_banks = {
            "tech_product": ["smartphone", "laptop", "tablet", "smartwatch", "router", "fitness tracker", 
                            "smart speaker", "drone", "VR headset", "wireless earbuds", "gaming console",
                            "digital camera", "e-reader", "smart TV", "robot vacuum", "smart thermostat",
                            "graphics card", "CPU", "SSD drive", "AI assistant", "security system"],
            
            "tech_domain": ["AI", "cloud", "internet", "IoT", "blockchain", "fintech", "edtech", "healthtech", 
                           "robotics", "automation", "cybersecurity", "AR", "VR", "quantum computing", "big data",
                           "machine learning", "data science", "web development", "app development", "UX design"],
            
            "media_type": ["movie", "TV show", "documentary", "series", "film", "play", "musical", "concert", 
                          "album", "song", "podcast", "novel", "book", "short story", "art exhibition",
                          "comedy special", "theatrical production", "performance", "reality show", "game show"],
            
            "entertainment_event": ["performance", "concert", "show", "exhibition", "festival", "play", 
                                   "screening", "premiere", "tour", "recital", "opera", "ballet", "stand-up set",
                                   "live event", "broadcast", "interview", "award ceremony", "release party"],
            
            "media_aspect": ["visuals", "acting", "direction", "cinematography", "special effects", "soundtrack", 
                            "script", "choreography", "lighting", "set design", "character development", "plot", 
                            "pacing", "editing", "score", "dialogue", "production value", "performances", "narrative"],
            
            "positive_adj": ["innovative", "impressive", "outstanding", "exceptional", "incredible", "fantastic", 
                            "reliable", "intuitive", "powerful", "efficient", "versatile", "seamless", "stunning",
                            "responsive", "engaging", "immersive", "revolutionary", "groundbreaking", "elegant", "brilliant"],
            
            "negative_adj": ["disappointing", "frustrating", "unreliable", "buggy", "sluggish", "defective", 
                            "confusing", "cumbersome", "mediocre", "outdated", "overpriced", "flawed", "boring",
                            "predictable", "unoriginal", "tedious", "shallow", "disjointed", "chaotic", "lackluster"],
            
            "positive_verb": ["revolutionizing", "transforming", "enhancing", "improving", "elevating", 
                             "streamlining", "optimizing", "enriching", "captivating", "delighting",
                             "fascinating", "intriguing", "entertaining", "inspiring", "empowering"],
            
            "negative_verb": ["ruins", "destroys", "compromises", "undermines", "crashes", "freezes", 
                             "malfunctions", "fails", "disappoints", "bores", "frustrates", "irritates",
                             "annoys", "confuses", "overwhelms", "underwhelms", "drags", "deteriorates"],
            
            "positive_emotion": ["thrilled", "delighted", "impressed", "satisfied", "amazed", "excited", 
                                "pleased", "overjoyed", "captivated", "fascinated", "grateful", "enthusiastic",
                                "inspired", "moved", "enthralled", "enchanted", "engaged", "mesmerized"],
            
            "negative_emotion": ["disappointed", "frustrated", "annoyed", "irritated", "dissatisfied", 
                                "unimpressed", "bored", "tired", "displeased", "disenchanted", "concerned",
                                "confused", "distracted", "underwhelmed", "alienated", "disengaged"],
            
            "positive_noun": ["masterpiece", "breakthrough", "revelation", "game-changer", "innovation", 
                             "marvel", "success", "achievement", "triumph", "wonder", "delight", "gem",
                             "treasure", "standout", "highlight", "sensation", "phenomenon", "milestone"],
            
            "negative_noun": ["letdown", "disappointment", "failure", "disaster", "fiasco", "flop", 
                             "nightmare", "waste", "mess", "catastrophe", "headache", "hassle", "nuisance",
                             "eyesore", "blunder", "misstep", "mistake", "mishap", "setback"],
            
            "positive_adv": ["wonderfully", "incredibly", "remarkably", "exceptionally", "surprisingly", 
                            "genuinely", "thoroughly", "perfectly", "absolutely", "truly", "completely",
                            "consistently", "undeniably", "refreshingly", "delightfully", "impressively"],
            
            "adjective": ["cutting-edge", "innovative", "advanced", "revolutionary", "state-of-the-art", 
                          "next-generation", "groundbreaking", "futuristic", "high-tech", "leading-edge",
                          "top-of-the-line", "premium", "sophisticated", "sleek", "modern", "intelligent"]
        }

        # Generate sentences
        # Technology & Positive: Task A = 0, Task B = 1
        self._generate_category_samples(samples_per_category, 
                                        self.tech_positive_templates, 
                                        self.word_banks, 0, 1)
            
        # Technology & Negative: Task A = 0, Task B = 0
        self._generate_category_samples(samples_per_category, 
                                        self.tech_negative_templates, 
                                        self.word_banks, 0, 0)
            
        # Entertainment & Positive: Task A = 1, Task B = 1
        self._generate_category_samples(samples_per_category, 
                                        self.entertainment_positive_templates, 
                                        self.word_banks, 1, 1)
            
        # Entertainment & Negative: Task A = 1, Task B = 0
        self._generate_category_samples(samples_per_category, 
                                        self.entertainment_negative_templates, 
                                        self.word_banks, 1, 0)
        
        # Shuffle the dataset while maintaining label correspondence
        combined = list(zip(self.sentences, self.taskA_labels, self.taskB_labels))
        random.shuffle(combined)
        self.sentences, self.taskA_labels, self.taskB_labels = zip(*combined)
        
        # Convert back to lists for mutability
        self.sentences = list(self.sentences)
        self.taskA_labels = list(self.taskA_labels)
        self.taskB_labels = list(self.taskB_labels)

    def _generate_category_samples(self, num_samples, templates, word_banks, taskA_label, taskB_label):
        """Generate samples for a specific category using templates and word banks"""
        for i in range(num_samples):
            template = random.choice(templates)
            # Fill in the template with random words from appropriate word banks
            filled_template = template
            # Find all placeholders in the template
            placeholders = re.findall(r'\{([^}]+)\}', template)
            # Replace each placeholder with a random word from the corresponding word bank
            for placeholder in placeholders:
                if placeholder in word_banks:
                    replacement = random.choice(word_banks[placeholder])
                    filled_template = filled_template.replace(f"{{{placeholder}}}", replacement, 1)
            
            self.sentences.append(filled_template)
            self.taskA_labels.append(taskA_label)
            self.taskB_labels.append(taskB_label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "taskA_label": self.taskA_labels[idx],
            "taskB_label": self.taskB_labels[idx]
        }


def create_train_test_datasets(total_samples=1000, test_ratio=0.2, random_seed=42):
    """
    Create train and test datasets with balanced classes
    Args:
        total_samples (int): Total number of samples (must be divisible by 4)
        test_ratio (float): Ratio of data to use for testing (0.0-1.0)
        random_seed (int): Random seed for reproducibility
    """
    # Ensure total_samples is divisible by 4 for balanced classes
    if total_samples % 4 != 0:
        total_samples = total_samples + (4 - (total_samples % 4))
        print(f"Adjusted total_samples to {total_samples} to ensure balanced classes")
    
    # Create the full dataset
    full_dataset = MultiTaskDataset(num_samples=total_samples, random_seed=random_seed)
    
    # Calculate train/test split sizes
    test_size = int(total_samples * test_ratio)
    train_size = total_samples - test_size
    
    # Ensure test_size is divisible by 4 for balanced test set
    if test_size % 4 != 0:
        test_size = (test_size // 4) * 4
        train_size = total_samples - test_size
    
    # Create stratified split to maintain class balance
    train_indices = []
    test_indices = []
    
    # Group indices by their combined labels
    label_indices = {
        (0, 0): [],  # Tech & Negative
        (0, 1): [],  # Tech & Positive
        (1, 0): [],  # Entertainment & Negative
        (1, 1): []   # Entertainment & Positive
    }
    
    for idx in range(len(full_dataset)): # Iterate over all samples
        taskA = full_dataset.taskA_labels[idx]
        taskB = full_dataset.taskB_labels[idx]
        label_indices[(taskA, taskB)].append(idx)
    
    # Calculate how many samples of each class should be in the test set
    samples_per_class_test = test_size // 4
    
    # Split each class proportionally
    for label_combo, indices in label_indices.items():
        random.shuffle(indices)
        test_indices.extend(indices[:samples_per_class_test]) # Add the first samples to the test set
        train_indices.extend(indices[samples_per_class_test:]) # Add the rest to the train set
    
    # Create the train and test datasets using the indices
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    return train_dataset, test_dataset


class Subset(Dataset):
    """Custom Subset class for dataset at specified indices"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
        # Pre-calculate the class distribution for easier access
        self.taskA_labels = [dataset.taskA_labels[i] for i in indices]
        self.taskB_labels = [dataset.taskB_labels[i] for i in indices]
        self.sentences = [dataset.sentences[i] for i in indices]
        
    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "taskA_label": self.taskA_labels[idx],
            "taskB_label": self.taskB_labels[idx]
        }
        
    def __len__(self):
        return len(self.indices)
        
    def get_class_distribution(self):
        """Return the distribution of classes in this subset"""
        distribution = {
            ('Tech', 'Neg'): 0,  # Tech & Negative
            ('Tech', 'Pos'): 0,  # Tech & Positive
            ('Entertain', 'Neg'): 0,  # Entertainment & Negative
            ('Entertain', 'Pos'): 0   # Entertainment & Positive
        }
        
        for i in range(len(self)):
            taskA = self.taskA_labels[i]
            taskA = "Tech" if taskA == 0 else "Entertain"
            taskB = self.taskB_labels[i]
            taskB = "Neg" if taskB == 0 else "Pos"
            distribution[(taskA, taskB)] += 1
            
        return distribution
