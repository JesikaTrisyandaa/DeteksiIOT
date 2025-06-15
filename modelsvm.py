import numpy as np

class SVM_Light:
    def __init__(self, bobot, bias):
        self.bobot = bobot
        self.bias = bias

    def predict(self, fitur_baru):
        hasil = np.dot(fitur_baru, self.bobot.T) + self.bias
        return np.where(hasil >= 0, 1, -1)
