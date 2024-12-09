from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import json
import os
from PIL import Image
import random
import torch

class VQADataset(Dataset):
    def __init__(self, annotations_path, questions_path, image_dir, dataset_type, transform=None):
        super().__init__()
        
        # Store parameters
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        
        # Set up transform with optional data augmentation
        self.transform = transform or transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.3), 
            transforms.ToTensor(),
        ])
        
        # Load annotations and questions
        self.annotations = self.load_json(annotations_path)['annotations']
        self.questions = self.load_json(questions_path)['questions']
        self.question_dict = {q['question_id']: q for q in self.questions}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        question_id = ann['question_id']
        image_id = ann['image_id']
        question = self.question_dict.get(question_id, {}).get('question', '')
        
        # Load the image
        prefix = 'COCO_train2014' if self.dataset_type == 'train' else 'COCO_val2014'
        image_path = os.path.join(self.image_dir, f"{prefix}_{image_id:012d}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get correct answers (typically there are multiple answers)
        correct_answers = [answer['answer'] for answer in ann['answers']]
        
        # Randomly sample half of the correct answers to keep as correct
        num_correct_to_keep = len(correct_answers) // 2
        correct_answers_kept = random.sample(correct_answers, num_correct_to_keep)

        # Generate incorrect answers by shuffling answers from other questions
        incorrect_answers = []
        while len(incorrect_answers) < num_correct_to_keep:
            incorrect_answer = self.get_random_incorrect_answer(correct_answers)
            # Ensure incorrect answer is not the same as any correct answer
            if incorrect_answer not in correct_answers:
                incorrect_answers.append(incorrect_answer)

        # Combine correct and incorrect answers into samples
        samples = []

        # Add correct answers
        for answer in correct_answers_kept:
            samples.append({
                'image': image,
                'question': question,
                'answer': answer,
                'label': 1,
                'image_id': image_id,
                'question_id': question_id
            })

        # Add incorrect answers
        for answer in incorrect_answers:
            samples.append({
                'image': image,
                'question': question,
                'answer': answer,
                'label': 0, 
                'image_id': image_id,
                'question_id': question_id
            })

        return samples

    def get_random_incorrect_answer(self, correct_answers):
        """Generate an incorrect answer by shuffling answers from other questions."""
        incorrect_answer = random.choice(correct_answers)
        while incorrect_answer in correct_answers:
            # Randomly select another annotation to fetch an answer
            random_idx = random.randint(0, len(self.annotations) - 1)
            random_ann = self.annotations[random_idx]
            random_answers = [answer['answer'] for answer in random_ann['answers']]
            incorrect_answer = random.choice(random_answers)
        
        return incorrect_answer

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def custom_collate_fn(batch):
        # Flatten the list of samples
        flat_batch = [item for sublist in batch for item in sublist]
        return {
            'image': torch.stack([item['image'] for item in flat_batch]),
            'question': [item['question'] for item in flat_batch],
            'answer': [item['answer'] for item in flat_batch],
            'label': torch.tensor([item['label'] for item in flat_batch], dtype=torch.long),
            'image_id': [item['image_id'] for item in flat_batch],
            'question_id': [item['question_id'] for item in flat_batch]
        }
