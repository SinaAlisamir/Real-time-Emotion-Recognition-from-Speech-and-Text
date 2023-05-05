import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import pickle

class SER_Module(object):
    """
    SER module to predict emotion based on features.
        
    """
    def __init__(self, device="cpu", model_path="", model_path_v="", model_path_l=""):
        self.device = device
        self.model_path = model_path

        if model_path_l != "":
            self.model_l = pickle.load(open(model_path_l, 'rb'))
            if model_path != "":
                self.model_a = pickle.load(open(model_path, 'rb'))
                self.model_v = pickle.load(open(model_path_v, 'rb'))
        else:
            self.model = torch.load(model_path, map_location=device)

        self.datasets_labels = {
            "IEMOCAP": ["happiness", "sadness", "anger", "neutral"],
            "CMU_MOSEI": ["negative", "positive"],
            # "GEMEP": ['amusement', 'colère', 'désespoir', 'fierté', 'inquiétude', 'intérêt', 'irritation', 'joie', 'peur', 'plaisir', 'soulagement', 'tristesse'],
            "GEMEP": ['fun', 'anger', 'despair', 'pride', 'worry', 'interest', 'irritation', 'joy', 'fear', 'pleasure', 'relief', 'sadness'],
            "RAVDESS": ['neutral', 'calmness', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise'],
            "CaFE": ['anger', 'disgust', 'joy', 'neutral', 'fear', 'surprise', 'sadness'],
            "EmoDB": ['anger', 'boredom', 'disgust', 'anxiety', 'happiness', 'sadness', 'neutral'],
        }

    def predict(self, feats_A, feats_L=None, dataset="IEMOCAP", multi_choice=0):
        # for cmu+iemocap: outputs = self.modules.main_model(x_A=feats_A, x_L=feats_L, taskID=taskID)
        if dataset in ["GEMEP", "RAVDESS", "CaFE", "EmoDB"]:
            output_rnn, _ = self.model.rnn(feats_A)
            output_rnn = output_rnn[:, -1, :].unsqueeze(1)
            output = self.model.linears[multi_choice](output_rnn)
        elif dataset in ["IEMOCAP", "CMU_MOSEI"]:
            output = self.model(x_A=feats_A, x_L=feats_L, taskID=multi_choice)
        elif dataset == "IEMOCAP-Allociné":
            arousal = self.model_a.predict(feats_A)[0]
            arousal = round(arousal, 3)
            valence = self.model_v.predict(feats_A)[0]
            valence = round(valence, 3)
            sentiment = self.model_l.predict_proba(feats_L)[0][1]
            sentiment = round(sentiment, 3)
            result = {}
            result["probs-3"] = f"arousal-audio:{arousal} - valence-audio:{valence} - sentiment-text:{sentiment}"
            result["probs"] = {"arousal":arousal, "valence":valence, "sentiment":sentiment}
            if sentiment < 0.2:
                if arousal > .65:
                    result["pred"] = "angry"
                else:
                    result["pred"] = "sad"
            elif sentiment > 0.6:
                if arousal > .65:
                    result["pred"] = "happy"
                else:
                    result["pred"] = "content"
            else: 
                result["pred"] = "neutral"
            self.result = result
            return result
        elif dataset == "Allociné":
            sentiment = self.model_l.predict_proba(feats_L)[0][1]
            sentiment = round(sentiment, 3)
            result = {}
            result["probs-3"] = f"sentiment:{sentiment}"
            result["probs"] = {"sentiment":sentiment, "arousal":0.5, "valence":sentiment}
            if sentiment < 0.2:
                result["pred"] = "sad"
            elif sentiment > 0.6 and sentiment <= 0.9:
                result["pred"] = "content"
            elif sentiment > 0.9:
                result["pred"] = "happy"
            else: 
                result["pred"] = "neutral"
            self.result = result
            return result

        outs = torch.nn.functional.softmax(output.squeeze().squeeze())
        outs = outs.cpu().detach().numpy()

        labels = self.datasets_labels[dataset]
        ranks = np.argsort(outs) # worst to best
        ranks = np.flipud(ranks) # best to worst
        result = {}
        result["pred"] = labels[np.argmax(outs)]
        result["probs"] = {}
        for out, label in zip(outs, labels):
            result["probs"][label] = out
        result["probs-3"] = ""
        max_num = min(3, len(ranks))
        for i in range(max_num):
            idx = ranks[i]
            label = labels[idx]
            out = int(outs[idx]*100)
            if i == 0:
                result["probs-3"] = f"{label}: {out}%"
            else:
                result["probs-3"] += " - " + f"{label}: {out}%"
        self.result = result
        return result
