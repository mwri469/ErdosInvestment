import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 16, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(16, 8, batch_first=True, num_layers=1)
        self.fc = nn.Linear(8, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])

class MediumLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 32, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(32, 16, batch_first=True, num_layers=1)
        self.lstm3 = nn.LSTM(16, 8, batch_first=True, num_layers=1)
        self.fc = nn.Linear(8, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return self.fc(x[:, -1, :])

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, num_layers=1)
        self.lstm3 = nn.LSTM(32, 16, batch_first=True, num_layers=1)
        self.fc = nn.Linear(16, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return self.fc(x[:, -1, :])

class LightDropoutLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(32, 16, batch_first=True, num_layers=1)
        self.dropout3 = nn.Dropout(dropout)
        self.fc = nn.Linear(16, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        return self.fc(x[:, -1, :])

class HeavyDropoutLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(32, 16, batch_first=True, num_layers=1)
        self.dropout3 = nn.Dropout(dropout - 0.2)
        self.fc = nn.Linear(16, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        return self.fc(x[:, -1, :])

class ComplexBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, past_steps=3, future_steps=1):
        super().__init__()
        # First bidirectional layer
        self.lstm1 = nn.LSTM(
            input_size, 128, 
            batch_first=True, 
            bidirectional=True, 
            num_layers=1
        )
        # Subsequent bidirectional layers
        self.lstm2 = nn.LSTM(
            256, 64, 
            batch_first=True, 
            bidirectional=True, 
            num_layers=1
        )
        self.lstm3 = nn.LSTM(
            128, 32, 
            batch_first=True, 
            bidirectional=True, 
            num_layers=1
        )
        self.lstm4 = nn.LSTM(
            64, 16, 
            batch_first=True, 
            bidirectional=True, 
            num_layers=1
        )
        self.fc = nn.Linear(32, future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)  # Bidirectional doubles the output feature dim
        x, _ = self.lstm2(x)  
        x, _ = self.lstm3(x)  
        x, _ = self.lstm4(x)  
        return self.fc(x[:, -1, :])

# Mapping function for easier instantiation
def get_model(model_type, input_size, past_steps=3, future_steps=1):
    model_map = {
        'simple': SimpleLSTM,
        'medium': MediumLSTM,
        'complex': ComplexLSTM,
        'light_dropout': LightDropoutLSTM,
        'heavy_dropout': HeavyDropoutLSTM,
        'complex_bidirectional': ComplexBidirectionalLSTM
    }
    return model_map[model_type](input_size, past_steps, future_steps)
