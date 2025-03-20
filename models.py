from torch import nn


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
    

class TimeSeriesWaveletModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeSeriesWaveletModel, self).__init__()
        self.feature_combiner = nn.Linear(input_size, 1)
        
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        
        x = self.feature_combiner(x)  
        
        _, (h_n, _) = self.lstm(x)
        
        out = self.fc(h_n[-1])
        return out


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return out
    

class LSTMWaveletModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMWaveletModel, self).__init__()
        self.feature_combiner = nn.Linear(input_size, 1)
        
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        
        x = self.feature_combiner(x)  
        
        out, (h_n, _) = self.lstm(x)
        
        return out



