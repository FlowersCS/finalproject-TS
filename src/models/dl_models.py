import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_size=354, num_classes=6):
        super(MLPModel, self).__init__()
        # Hidden layers with dropout decrecientes
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.35)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
        
        # weights init
        self._initialize_weights()
        

    def forward(self, x):
        # negative_slope ayuda a mantener el flujo de gradientes
        x = F.selu(self.fc1(x)) 
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.selu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = F.selu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        return self.output(x)
    
    def predict_proba(self, x):
        """Método para compatibilidad con sklearn-style"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1).cpu().numpy()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)  # Inicialización ortogonal
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Pequeño bias inicial
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=354, num_classes=6, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Para que la entrada sea (batch, seq, feature)
            dropout=0.3 if num_classes > 1 else 0,  # Dropout entre capas LSTM
            #bidirectional=True  # Bidireccional para capturar mejor la temporalidad
        )
        
        # Capas de atencion
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Capas fully connected
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x debe tener forma (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Aplicar atención
        attn_weights = self.attention(lstm_out)
        attn_weights = attn_weights.squeeze(2) 
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalizar pesos de atención
        attn_output = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, hidden_size)
        
        # Capas fully connected
        x = F.relu(self.fc1(attn_output))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
        


class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=6, input_features=354):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Calcula el tamaño de entrada a fc1 automáticamente
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_features)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            self._to_linear = dummy.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1:
            x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x