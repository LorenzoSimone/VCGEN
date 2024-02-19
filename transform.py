import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

class TextEmbedder:
    def __init__(self):
        self.tokenizer = {"bioBert": AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")}

    def embed(self, model, text):
        return self.tokenizer[model].encode(text, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
    

class VCGTransform:
    def __init__(self, ecg_series):
        self.ecg_series = ecg_series
        self.weights = {
            "kors": np.array([[0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
                              [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
                              [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.311]]),
            "inverse_dower": np.array([[0.156, -0.010, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194],
                                       [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],
                                       [0.022, 0.102, -0.229, -0.310, -0.246, -0.063, 0.055, 0.108]]),
            "qlsv": np.array([[0.199, -0.018, -0.147, -0.058, 0.037, 0.139, 0.232, 0.226],
                              [-0.164, 0.503, 0.023, -0.085, -0.003, 0.033, 0.060, 0.104],
                              [0.085, -0.130, -0.184, -0.163, -0.193, -0.119, -0.023, 0.043]])
        }
    
    def ecg2vcg(self, method):
        result = []
        for ecg in self.ecg_series:
            i, ii, v1, v2, v3, v4, v5, v6 = ecg[:, 0], ecg[:, 1], ecg[:, 6], ecg[:, 7], ecg[:, 8], ecg[:, 9], ecg[:, 10], ecg[:, 11]
            result.append(np.array(self.weights[method] @ np.array([i, ii, v1, v2, v3, v4, v5, v6])).T)
        return result
    
    def ecg2vcg(self, method):
        result = []
        for ecg in self.ecg_series:
            i, ii, v1, v2, v3, v4, v5, v6 = ecg[:, 0], ecg[:, 1], ecg[:, 6], ecg[:, 7], ecg[:, 8], ecg[:, 9], ecg[:, 10], ecg[:, 11]
            result.append(np.array(self.weights[method] @ np.array([i, ii, v1, v2, v3, v4, v5, v6])).T)
        return result
    
