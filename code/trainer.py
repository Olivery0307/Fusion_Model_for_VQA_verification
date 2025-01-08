import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from data_loader import VQADataset

class Trainer:
    def __init__(self, 
                 training_image_dir, 
                 training_annotations_path, 
                 training_questions_path,
                 val_image_dir, 
                 val_annotations_path, 
                 val_questions_path,
                 model_class,
                 device=None,
                 batch_size=64,
                 learning_rate=1e-5,
                 weight_decay=1e-2):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_class = model_class

        self.training_dataset = VQADataset(
            image_dir=training_image_dir,
            annotations_path=training_annotations_path,
            questions_path=training_questions_path,
            dataset_type="train"
        )

        self.val_dataset = VQADataset(
            image_dir=val_image_dir,
            annotations_path=val_annotations_path,
            questions_path=val_questions_path,
            dataset_type="val"
        )
        
        self.training_data_loader = None
        self.val_data_loader = None

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def create_data_loaders(self, sample_size=None):
        if sample_size:
            self.training_dataset = Subset(self.training_dataset, range(min(sample_size, len(self.training_dataset))))
            self.val_dataset = Subset(self.val_dataset, range(min(sample_size, len(self.val_dataset))))

        self.training_data_loader = DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=VQADataset.custom_collate_fn
        )
        self.val_data_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=VQADataset.custom_collate_fn
        )

    def sample_batch(self,sample_size=None):
        if self.training_data_loader is None:
            self.create_data_loaders(sample_size)
        
        for batch in self.training_data_loader:
            # Extract the data for the first question in the batch
            question = batch['question'][0]
            image = batch['image'][0]
            image_id = batch['image_id'][0]
            question_id = batch['question_id'][0]
            
            # Convert the tensor image to PIL format for visualization
            image = transforms.ToPILImage()(image.cpu())
            
            # Visualize the image
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Image ID: {image_id}")
            plt.show()
            
            # Print out the question details
            print(f"Question ID: {question_id}")
            print(f"Question: {question}")
            
            # Loop through all answers in the batch that correspond to the selected question
            for i in range(len(batch['answer'])):
                if batch['question_id'][i] == question_id:
                    answer = batch['answer'][i]
                    label = batch['label'][i]
                    print(f"Answer {i + 1}: {answer} (Label: {'Correct' if label == 1 else 'Incorrect'})")
            
            # Break after showing the first question and its answers
            break

    def train(self, num_epochs=5, sample_size=None, showing_batch_size=None):
        if self.training_data_loader is None:
            self.create_data_loaders(sample_size)

        model = self.model_class().to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if showing_batch_size is None:
            showing_batch_size = max(1, len(self.training_data_loader) // 10)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            print(f"Starting Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(self.training_data_loader):
                images = batch['image'].to(self.device)
                questions = batch['question']
                answers = batch['answer']
                labels = batch['label'].float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = model(images, questions, answers)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs >= 0.5).long()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                if batch_idx % showing_batch_size == 0:
                    print(f"Batch [{batch_idx}/{len(self.training_data_loader)}], Loss: {loss.item():.4f}")

            epoch_train_loss = train_loss / len(self.training_data_loader)
            epoch_train_accuracy = train_correct / train_total
            self.train_losses.append(epoch_train_loss)
            self.train_accuracies.append(epoch_train_accuracy)

            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch in self.val_data_loader:
                    images = batch['image'].to(self.device)
                    questions = batch['question']
                    answers = batch['answer']
                    labels = batch['label'].float().unsqueeze(1).to(self.device)

                    outputs = model(images, questions, answers)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs >= 0.5).long()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / len(self.val_data_loader)
            epoch_val_accuracy = val_correct / val_total
            self.val_losses.append(epoch_val_loss)
            self.val_accuracies.append(epoch_val_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

        print("Training completed successfully.")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

    def plot_training_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Adjust x-axis for epoch (shift by 1)
        epochs = [i + 1 for i in range(len(self.train_losses))]
        
        # Plot Loss Graph
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        ax1.plot(epochs, self.val_losses, label='Validation Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot Accuracy Graph
        ax2.plot(epochs, self.train_accuracies, label='Train Accuracy')
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.5, 1) 
        ax2.legend()
        
        plt.tight_layout()
        plt.show()



    def compare_predictions(self, model):
        """
        Visualize two images with their questions, answers, labels (correct/incorrect), and model predictions.
        
        Args:
            model (nn.Module): Trained model to evaluate predictions.
        """
        if self.training_data_loader is None:
            self.create_data_loaders()
    
        # Switch model to evaluation mode
        model.eval()
    
        for batch in self.training_data_loader:
            # Extract the data for the first question in the batch
            question = batch['question'][0]
            image = batch['image'][0]
            image_id = batch['image_id'][0]
            question_id = batch['question_id'][0]
    
            # Convert the tensor image to PIL format for visualization
            image = transforms.ToPILImage()(image.cpu())
    
            # Visualize the image
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Image ID: {image_id}")
            plt.show()
    
            # Print out the question details
            print(f"Question ID: {question_id}")
            print(f"Question: {question}")
    
            # Loop through all answers in the batch that correspond to the selected question
            for i in range(len(batch['answer'])):
                if batch['question_id'][i] == question_id:
                    answer = batch['answer'][i]
                    label = batch['label'][i]
    
                    # Model Prediction
                    image_tensor = batch['image'][i].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        prediction = model(image_tensor, [question], [answer])
                        predicted_label = (torch.sigmoid(prediction) >= 0.5).long().item()
    
                    # Print the answer, label, and prediction
                    print(f"Answer {i + 1}: {answer}")
                    print(f"Label: {label}")
                    print(f"Prediction: {predicted_label}\n")
    
            # Break after processing the first question and its answers
            break



