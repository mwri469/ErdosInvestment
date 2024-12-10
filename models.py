import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from preprocess import *
from sequential_models import *


def main():
    # Placeholder preprocessing (replace with your actual data loading)

    X_train, y_train, X_val, y_val, X_oos, y_oos = preprocess_data()

    torch.set_float32_matmul_precision('high')

    pipeline = ModelPipeline()
    pipeline.train_ensemble(X_train, y_train, X_val, y_val)
    pipeline.save_models()

    y_hat_oos = pipeline.predict_ensemble(X_oos)
    mse = mean_squared_error(y_hat_oos, y_oos)
    print(f'Mean Squared Error: {mse}')

class ModelPipeline:
    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = config or {
            'num_models': 20,
            'learning_rate_range': (0.0001, 0.01),
            'epochs_range': (10, 50),
            'seq_models': [
                'simple',
                'medium',
                'complex',
                'light_dropout',
                'heavy_dropout',
                'complex_bidirectional'
            ]
        }
        self.models = []

        # Compile each model during training
        self._train_single_model = torch.compile(
            self._train_single_model, 
            mode='max-autotune'
        )

    @torch.compile
    def _train_single_model(self, train_loader, val_loader, input_size, output_size):
        # Randomly select model architecture and hyperparameters
        model_class = random.choice(self.config['seq_models'])
        learning_rate = random.uniform(*self.config['learning_rate_range'])
        epochs = random.randint(*self.config['epochs_range'])

        model = get_model(model_class, input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model_funcs.append({'class': model_class, 'input': input_size})

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Optional validation
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(model(x_val.to(self.device)), y_val.to(self.device))
                    for x_val, y_val in val_loader
                ) / len(val_loader)

        return model

    @torch.compile
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256)

        input_size = X_train.shape[2]
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1

        print('\nTraining ensemble models...')
        self.models = []
        self.model_funcs = []
        for _ in tqdm(range(self.config['num_models'])):
            self.models.append(self._train_single_model(train_loader, val_loader, input_size, output_size))

    def predict_ensemble(self, X_oos):
        test_dataset = torch.FloatTensor(X_oos).to(self.device)
        
        with torch.no_grad():
            predictions = torch.stack([
                model(test_dataset).cpu() 
                for model in self.models
            ])
        
        return predictions.mean(dim=0).numpy()

    def save_models(self, path='model_weights/ensemble_models.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump([model.state_dict() for model in self.models], f)

    def load_models(self, path='model_weights/ensemble_models.pkl', input_size=None, output_size=None):
        with open(path, 'rb') as f:
            state_dicts = pickle.load(f)
        
        if input_size is None or output_size is None:
            raise ValueError("Must provide input_size and output_size when loading models")
        
        self.models = []
        for state_dict in state_dicts:
            model = get_model
            model.load_state_dict(state_dict)
            self.models.append(model)

if __name__ == '__main__':
    main()
