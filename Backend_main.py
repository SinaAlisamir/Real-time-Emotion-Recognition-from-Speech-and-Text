'''
Audio stream -> Buffer -> VAD -> SER -> Emotion
'''
from datetime import datetime
import os, sys, time, argparse, glob, logging
import speechbrain as sb
import pyaudio
import numpy as np
import json
import time
# from pydub import AudioSegment
# import soundfile as sf
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from VAD_Module import VAD_Module
from SER_Module import SER_Module
from Feats_Module import Feats_Module
from threading import Thread
import speech_recognition
import torch
from scipy.io import wavfile
from pydub import AudioSegment

from tkinter import *
from tkinter import ttk
from settings import Settings


class Main():
    def __init__(self):
        super().__init__()
        self.turn_counter = 1
        self.text_counter = 1
        self.analysed_items = []
        self.init_settings()
        self.init_all_models()
        self.prep_to_listen()
        T=Thread(target=self.main_loop,args=())
        T.start()
        # self.main_loop()

    def init_settings(self):
        self.device = Settings.device
        self.VAD_model_id = "MFB_GRU"
        self.SER_model_id = "AL_IEMOCAP_Allociné"
        self.TER_model_id = "L_allociné"
        self.VAD_model = Settings.Models.VAD_models[self.VAD_model_id]
        self.SER_model = Settings.Models.SER_models[self.SER_model_id]
        self.TER_model = Settings.Models.TER_models[self.TER_model_id]

    def change_VAD_withTitle(self, title):
        for key, value in Settings.Models.VAD_models.items():
            if value["title"] == title: self.VAD_model_id = key
        self.VAD_model = Settings.Models.VAD_models[self.VAD_model_id]
        self.init_VAD(skipFeats=True)
    
    def change_SER_withTitle(self, title):
        for key, value in Settings.Models.SER_models.items():
            if value["title"] == title: self.SER_model_id = key
        self.SER_model = Settings.Models.SER_models[self.SER_model_id]
        self.init_SER(skipFeats=True)

    def change_TER_withTitle(self, title):
        for key, value in Settings.Models.TER_models.items():
            if value["title"] == title: self.TER_model_id = key
        self.TER_model = Settings.Models.TER_models[self.TER_model_id]
        self.init_TER(skipFeats=True)
        
    def add_text_analysis(self):
        item = {
            "number": len(self.analysed_items)+1, # +1 to count from one not zero
            "modality": "L",
            "trs": self.txtInp,
            "probs": self.txtOut["probs-3"],
        }
        self.analysed_items.append(item)
        
    def add_speech_analysis(self):
        transcription = self.get_ASR_trans()
        item = {
            "number": len(self.analysed_items)+1, # +1 to count from one not zero
            "modality": "A",
            "file_path": self.segment_save_path,
            "trs": transcription,
            "probs": self.labels["probs-3"],
        }
        if transcription != "":
            self.turn_counter = self.turn_counter + 1
            self.analysed_items.append(item)

    def get_ASR_trans(self):
        transcription = self.get_google_ASR()
        # transcription = self.get_VOSK_ASR()
        return transcription

    def get_VOSK_ASR(self):
        try:
            text_path = "temp_vosk.srt"
            os.system(f"vosk-transcriber -l fr -i {self.segment_save_path} -t srt -o {text_path}")
            with open(text_path) as f:
                lines = f.readlines()
            lines_accepted = []
            for i, line in enumerate(lines):
                if (i+2)%4 == 0:
                    lines_accepted.append(line.replace("\n",""))
            transcription = " ".join(lines_accepted)
            os.remove(text_path)
            return transcription
        except:
            return ""

    def get_google_ASR(self):
        r = speech_recognition.Recognizer() 
        transcription = ""
        with speech_recognition.AudioFile(self.segment_save_path) as source: 
            audio = r.record(source) 
            try:
                transcription = r.recognize_google(audio, language=self.SER_model["lang"]) # "en-US" "fr-FR"
            except: 
                transcription = ""
        return transcription

    def read_interface_info(self):
        with open(Settings.interface_info_path, 'r') as fp:
            self.data = json.load(fp)

    def main_loop(self):
        time.sleep(.1)
        self.read_interface_info()
        if self.data["listen"]:
            self.listen_loop()
        else:
            self.main_loop()

    def init_VAD(self, skipFeats=False):
        if skipFeats:
            if self.VAD_model["feat_type"] == self.VADFeatsModule.feat_type:
                skipFeats = True
            else:
                skipFeats = False
        if not skipFeats:
            self.VADFeatsModule = Feats_Module(device=self.device, feat_type=self.VAD_model["feat_type"], norm=self.VAD_model["feat_norm"])
        self.VADModule = VAD_Module(device=self.device, model_path=self.VAD_model["model_path"])

    def init_SER(self, skipFeats=False):
        if skipFeats:
            if self.SER_model["feat_type"] == self.SERFeatsModule.feat_type:
                skipFeats = True
            else:
                skipFeats = False
        if not skipFeats:
            self.SERFeatsModule = Feats_Module(device=self.device, feat_type=self.SER_model["feat_type"], norm=self.SER_model["feat_norm"])
        if self.SER_model["feat_type_text"] != "":
            self.SER_TERFeatsModule = Feats_Module(device=self.device, feat_type=self.SER_model["feat_type_text"], norm=self.SER_model["feat_norm_text"])
        
        if "tfidf" in self.SER_model["feat_type_text"]:
            self.SER_TERFeatsModule = Feats_Module(device=self.device, 
                feat_type=self.SER_model["feat_type_text"], 
                norm=self.SER_model["feat_norm_text"],
                vectorizer_path=self.SER_model["vectorizer_path"])
            self.SERModule = SER_Module(device=self.device, model_path=self.SER_model["model_path"],
                model_path_v=self.SER_model["model_path_v"], 
                model_path_l=self.SER_model["model_path_trs"])
        else:
            self.SERModule = SER_Module(device=self.device, model_path=self.SER_model["model_path"])

    def init_TER(self, skipFeats=False):
        if skipFeats:
            if self.TER_model["feat_type"] == self.TERFeatsModule.feat_type:
                skipFeats = True
            else:
                skipFeats = False
        if not skipFeats:
            self.TERFeatsModule = Feats_Module(device=self.device, feat_type=self.TER_model["feat_type"], norm=self.TER_model["feat_norm"])
        if "tfidf" in self.SER_model["feat_type_text"]:
            self.TERFeatsModule = Feats_Module(device=self.device, 
                feat_type=self.TER_model["feat_type"], 
                norm=self.TER_model["feat_norm"],
                vectorizer_path=self.TER_model["vectorizer_path"])
            self.TERModule = SER_Module(device=self.device, model_path_l=self.TER_model["model_path"])
        else:
            self.TERModule = SER_Module(device=self.device, model_path=self.TER_model["model_path"])

    def init_all_models(self):
        self.init_VAD()
        self.init_SER()
        self.init_TER()
        print("All models loaded")

    def prep_to_listen(self):
        self.rate = Settings.frame_rate # in Hertz
        p=pyaudio.PyAudio()
        self.CHUNK = int(Settings.chunk_sec*self.rate)#2**15
        self.stream=p.open(format=pyaudio.paInt16,channels=1,rate=self.rate, input=True, frames_per_buffer=self.CHUNK)
        self.num_of_chunks = Settings.max_time_sec/Settings.chunk_sec
        print("prep_to_listen done")

    def analyse_text(self, txt):
        self.txtInp = txt
        inp = [txt]
        feats_L = self.TERFeatsModule.extract_feats(inp)
        classifier = self.TER_model["classifier"]
        out = self.TERModule.predict(feats_A=None, feats_L=feats_L, dataset=classifier["dataset"], multi_choice=classifier["index"])
        self.txtOut = out
        self.labels = out
        self.add_text_analysis()
        return out

    def stop_listening(self):
        self.segment_save_path = os.path.join(self.data["log_dir"], f"Turn_{self.turn_counter}.wav")
        segment_int16 = (np.iinfo(np.int16).max * (self.segment/np.abs(self.segment).max())).astype(np.int16)
        wavfile.write(self.segment_save_path, self.rate, segment_int16)
        self.get_emotion(self.segment)
        self.add_speech_analysis()

    def get_emotion(self, segment):
        self.feats_A = self.SERFeatsModule.extract_feats(segment)
        self.feats_L = None
        if self.SER_model["feat_type_text"] != "":
            trans = self.get_ASR_trans()
            if trans != "":
                self.feats_L = self.SER_TERFeatsModule.extract_feats([trans])
        classifier = self.SER_model["classifier"]
        self.labels = self.SERModule.predict(self.feats_A, feats_L=self.feats_L, dataset=classifier["dataset"], multi_choice=classifier["index"])
        return self.labels

    def get_emotion_from_file(self, file_path):
        self.segment_save_path = file_path
        audio_file = AudioSegment.from_wav(file_path)
        segment = np.array(audio_file.get_array_of_samples()) / 32767.0
        self.get_emotion(segment)
        self.add_speech_analysis()

    def listen_loop(self):
        self.buffer = np.array([])
        for i in range(int(self.num_of_chunks)):
            print("Listening", i)
            self.read_interface_info()
            audioChunk = np.fromstring(self.stream.read(self.CHUNK, exception_on_overflow = False),dtype=np.int16)
            sig = audioChunk / 32767.0
            self.buffer = np.concatenate((self.buffer, sig), 0)
            if len(self.buffer) > Settings.buffer_max: 
                self.buffer = self.buffer[-Settings.buffer_max:]

            segment_detected = False
            self.segment = self.buffer
            # in case the user has not activated the VAD and just presses start/stop listening
            if not self.data["listen"] and not self.data["VAD"]["active"]: 
                segment_detected=True
            # print("segment_detected", segment_detected)
            if self.data["VAD"]["active"]:
                # print("vad active")
                self.feats_VAD = self.VADFeatsModule.extract_feats(self.buffer)
                feats_fs = (len(self.buffer)/Settings.frame_rate)/self.feats_VAD.size()[1]
                preds = self.VADModule.predict(self.feats_VAD)
                pp_sets = self.VAD_model["post_process"]
                # print("preds", np.mean(preds))
                times = self.VADModule.post_process(feats_fs=feats_fs, 
                                                    hys_top=pp_sets["hys_top"], hys_bottom=pp_sets["hys_bottom"], 
                                                    cutWin_sec=pp_sets["cutWin_sec"], mergeWin_sec=pp_sets["mergeWin_sec"])
                if len(times) > 0: 
                    start = int(len(self.buffer) * times[0][0]) - int(0.5*Settings.frame_rate)
                    stop  = int(len(self.buffer) * times[-1][-1])
                    # print(times, stop, len(self.buffer))
                    # print("stop", stop, len(self.buffer) - len(sig))
                    coeff = Settings.wait_chunks_for_VAD
                    checkTime = len(self.buffer) - coeff*len(sig)
                    if not stop > checkTime: # wait a chunk to decide!
                        self.segment = self.buffer[start:-1]
                        segment_detected = True
                # print("times", times)

            if segment_detected:
                # print("lens", len(self.segment), Settings.buffer_max)
                T=Thread(target=self.stop_listening,args=())
                T.start()
                self.main_loop()
                break
            
            if not self.data["listen"]:
                self.main_loop()
                break

if __name__== "__main__":
    main = Main()

