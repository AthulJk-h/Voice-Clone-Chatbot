import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    model = None
    _mapping = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None


    def predict(self, file_path):

        MFCCs = self.preprocess(file_path)

        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
     
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":

    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()
    assert kss is kss1
    keyword1 = kss.predict("test/down.wav")   
    print("Prediction")
    print(keyword1)