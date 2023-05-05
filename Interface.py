# from tkinter import *
import tkinter
from tkinter import messagebox
from tkinter import Tk
from tkinter import ttk
import numpy as np
# from PIL import ImageTk, Image
import os, sys
import json
from datetime import datetime
from settings import Settings
from Backend_main import Main as backend_main
import torch
from pydub import AudioSegment
from pydub.playback import play
# from VAD_plotter import VAD_plotter

class Visualise():
    def __init__(self):
        super().__init__()
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        sys.setrecursionlimit(50000)
        self.backend = backend_main()
        self.init_logs()
        self.init_settings()
        self.init_tk()
        self.add_elements_to_canvas()
        # self.vad_plotter = VAD_plotter(self.backend)
        self.show_canvas()

    def init_tk(self):
        self.window_size = Settings.Interface.window_size
        self.root = Tk()
        self.root.title(string=Settings.Interface.title)
        self.root.geometry(self.window_size)
        # self.root.state('zoomed')
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()

    def init_logs(self):
        self.runTime = datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        self.main_save_dir = os.path.join(Settings.logs_dir, self.runTime)
        if not os.path.exists(self.main_save_dir):
            os.makedirs(self.main_save_dir)
        
    def init_settings(self):
        self.analysed_items = []
        self.data = {
            "log_dir": self.main_save_dir,
            "listen": False,
            "VAD": {"active":False},
            "SER": {}
        }
        self.save_settings(init=True)
    
    def load_backend_data(self):
        with open(Settings.interface_info_path, 'r') as fp:
            data = json.load(fp)

    def save_settings(self, init=False):
        if not init: self.load_backend_data() # to avoid overwritting
        with open(Settings.interface_info_path, 'w') as fp:
            json.dump(self.data, fp, ensure_ascii=False)

    def show_canvas(self):
        self.root.after(100, self.updateLoop)
        self.root.mainloop()

    def add_elements_to_canvas(self):
        
        # imageScale = 1/3
        # img = Image.open('Loading_2.gif').resize((int(198*imageScale),int(198*imageScale)))
        # self.tkimage = ImageTk.PhotoImage(img)
        # self.label_img = tkinter.Label(self.root, image=self.tkimage)
        # self.label_img.place(relx=.1, rely=.2, anchor="center")

        txtW = .25
        btnW = .2
        btnH = .05

        txt_rly = .05
        sph_rly = .2

        self.label_text = tkinter.Label(self.root, text= "Text:")
        self.label_text.place(relx=.05, rely=txt_rly, relwidth=btnW, relheight=btnH, anchor="center")
        self.textBox = tkinter.Entry(self.root, highlightcolor="#46011D", highlightbackground="#46011D")
        self.textBox.insert("0", Settings.Interface.TER_init_text)
        self.textBox.bind('<Return>', self.analyse_text)
        self.textBox.place(relx=.3, rely=txt_rly, relwidth=txtW, relheight=btnH, anchor="center")
        self.activate_textBox = tkinter.Button(self.root, text ="Analyse text", command = self.analyse_text)
        self.activate_textBox.place(relx=.55, rely=txt_rly, relwidth=btnW, relheight=btnH, anchor="center")

        self.label_speech = tkinter.Label(self.root, text= "Speech:")
        self.label_speech.place(relx=.05, rely=sph_rly, relwidth=btnW, relheight=btnH, anchor="center")
        self.vad_button = tkinter.Button(self.root, text ="VAD not active", command = self.VAD_activate_btn)
        self.vad_button.place(relx=.2, rely=sph_rly, relwidth=btnW, relheight=btnH, anchor="center")
        # self.vad_button_plt = tkinter.Button(self.root, text ="Plot VAD", command = self.VAD_plot_btn)
        # self.vad_button_plt.place(relx=.2, rely=sph_rly+btnH, relwidth=btnW/2, relheight=btnH, anchor="center")
        self.listen_button = tkinter.Button(self.root, text ="Start listening", command = self.listen_or_not)
        self.listen_button.place(relx=.4, rely=sph_rly, relwidth=btnW, relheight=btnH, anchor="center")
        self.load_file_button = tkinter.Button(self.root, text ="Load audio from file:", command = self.load_audio_file)
        self.load_file_button.place(relx=.6, rely=sph_rly, relwidth=btnW, relheight=btnH, anchor="center")
        self.textBox_file = tkinter.Entry(self.root, highlightcolor="#46011D", highlightbackground="#46011D")
        self.textBox_file.insert("0", '/path/file-16bInt-16kHz.wav')
        self.textBox_file.bind('<Return>', self.load_audio_file)
        self.textBox_file.place(relx=.83, rely=sph_rly, relwidth=txtW, relheight=btnH, anchor="center")

        self.add_optionMenus()
        self.add_resultsList()

    def add_optionMenus(self):
        self.VAD_option_Menu = tkinter.StringVar()
        self.VAD_options = []
        for key, value in Settings.Models.VAD_models.items():
            self.VAD_options.append(value["title"])
        self.VAD_menu = tkinter.OptionMenu(self.root, self.VAD_option_Menu, *self.VAD_options, command=self.VADmenu_callback)
        self.VAD_menu.place(relx=.2, rely=.26, relwidth=.15, relheight=.05, anchor="center")
        self.VAD_option_Menu.set(self.backend.VAD_model["title"])

        self.TER_option_Menu = tkinter.StringVar()
        self.TER_options = []
        for key, value in Settings.Models.TER_models.items():
            self.TER_options.append(value["title"])
        self.TER_menu = tkinter.OptionMenu(self.root, self.TER_option_Menu, *self.TER_options, command=self.TERmenu_callback)
        self.TER_menu.place(relx=.5, rely=.11, relwidth=.4, relheight=.05, anchor="center")
        self.TER_option_Menu.set(self.backend.TER_model["title"])

        self.SER_option_Menu = tkinter.StringVar()
        self.SER_options = []
        for key, value in Settings.Models.SER_models.items():
            self.SER_options.append(value["title"])
        self.SER_menu = tkinter.OptionMenu(self.root, self.SER_option_Menu, *self.SER_options, command=self.SERmenu_callback)
        self.SER_menu.place(relx=.5, rely=.26, relwidth=.4, relheight=.05, anchor="center")
        self.SER_option_Menu.set(self.backend.SER_model["title"])

    def VADmenu_callback(self, value):
        print("VADmenu_callback", value)
        self.backend.change_VAD_withTitle(value)

    def TERmenu_callback(self, value):
        print("TERmenu_callback", value)
        self.backend.change_TER_withTitle(value)

    def SERmenu_callback(self, value):
        print("SERmenu_callback", value)
        self.backend.change_SER_withTitle(value)

    def add_resultsList(self):
        rlx = .5
        rly = .6
        rlW = 0.8
        sbW = 0.02
        self.label_results = tkinter.Label(self.root, text= "Analysis:")
        self.label_results.place(relx=.05, rely=rly, relwidth=.2, relheight=.2, anchor="center")
        self.resultsList = tkinter.Text(self.root, highlightcolor="#46011D", highlightbackground="#46011D")
        self.resultsList.place(relx=rlx, rely=rly, relwidth=rlW, relheight=0.5, anchor="center")
        self.sb = tkinter.Scrollbar(self.root, command=self.resultsList.yview)
        self.sb.place(relx=rlx+(rlW-sbW)/2, rely=rly, relwidth=sbW, relheight=0.5, anchor="center")
        self.resultsList.configure(yscrollcommand=self.sb.set)
        self.resultsList.configure(state="disabled")

    def add_outputsCell_speech(self, item, itemDir="1.0"):
        width = int(self.resultsList.winfo_width() / 7.25)
        height = int(self.resultsList.winfo_height() / 50)
        textTemp = tkinter.Text(self.resultsList, height=height, width=width)
        textTemp.config(state='disabled')
        textTemp.bind_all("<MouseWheel>", self._on_mousewheel)
        
        label_title = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 18, "bold"), 
                                    text= "Item "+str(item["number"]))
        textTemp.window_create("end", window=label_title)
        label_title.place(relx=0.0, rely=0.0, relwidth=0.15, relheight=0.4, anchor='nw')

        button_play = tkinter.Button(textTemp, text="▷", command = lambda: self.play_sound(item["file_path"]), 
                                highlightbackground='red') #,height=3, width=1
        textTemp.window_create("end", window=button_play)
        button_play.place(relx=0.0, rely=0.42, relwidth=0.15, relheight=0.5, anchor='nw')
        # textTemp.insert("end", "\n")
        label_trs = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 14), 
                                    text= item["trs"])
        textTemp.window_create("end", window=label_trs)
        label_trs.place(relx=0.15, rely=0.0, relwidth=1.0, relheight=0.4, anchor='nw')
        # textTemp.insert("end", "\n")
        label_pred = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 14), 
                                    text= str(item["probs"]))
        textTemp.window_create("end", window=label_pred)
        label_pred.place(relx=0.15, rely=0.4, relwidth=1.0, relheight=0.4, anchor='nw')
        self.resultsList.window_create(itemDir, window=textTemp)
        # self.resultsList.update()
    
    def _on_mousewheel(self, event):
        # print(event.delta)
        self.resultsList.yview_scroll(-1*event.delta, "units")

    def add_outputsCell_text(self, item, itemDir="1.0"):
        self.resultsList.update()
        width = int(self.resultsList.winfo_width() / 7.25)
        height = int(self.resultsList.winfo_height() / 50)
        textTemp = tkinter.Text(self.resultsList, height=height, width=width) # , height=5, width=87
        # textTemp.place(relx=0.0, rely=0.0, relwidth=0.05, relheight=0.01, anchor='center')
        textTemp.config(state='disabled')
        textTemp.bind_all("<MouseWheel>", self._on_mousewheel)
        label_title = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 18, "bold"), 
                                    text= "Item "+str(item["number"]))
        textTemp.window_create("end", window=label_title)
        label_title.place(relx=0.0, rely=0.0, relwidth=0.15, relheight=0.4, anchor='nw')
        # textTemp.insert("end", "\n")
        label_trs = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 14), 
                                    text= item["trs"])
        textTemp.window_create("end", window=label_trs)
        label_trs.place(relx=0.15, rely=0.0, relwidth=1.0, relheight=0.4, anchor='nw')
        # textTemp.insert("end", "\n")
        label_pred = tkinter.Label(textTemp, bg='white', font=("Comic Sans MS", 14), 
                                    text= str(item["probs"]))
        textTemp.window_create("end", window=label_pred)
        label_pred.place(relx=0.15, rely=0.4, relwidth=1.0, relheight=0.4, anchor='nw')
        self.resultsList.window_create(itemDir, window=textTemp)
        # self.text.insert("1.0", "\n")

    def play_sound(self, file_path):
        print("play_sound")
        audio_file = AudioSegment.from_wav(file_path)
        play(audio_file)

    def analyse_text(self, event=None):
        print("analyse_text")
        out = self.backend.analyse_text(self.textBox.get())
        print("out", out)

    def load_audio_file(self, event=None):
        print("load_audio_file")
        # messagebox.showinfo(title="Warning", message="Method not yet implemented")
        file_path = self.textBox_file.get()
        self.backend.get_emotion_from_file(file_path)

    def listen_or_not(self):
        if self.listen_button['text'] == 'Stop listening':
            self.listen_button['text'] = 'Start listening'
            self.data["listen"] = False
            self.save_settings()
        elif self.listen_button['text'] == 'Start listening':
            self.listen_button['text'] = 'Stop listening'
            self.data["listen"] = True
            self.save_settings()

    def VAD_activate_btn(self):
        if self.vad_button['text'] == 'VAD not active':
            self.vad_button['text'] = 'VAD activated'
            self.activate_VAD()
        elif self.vad_button['text'] == 'VAD activated':
            self.vad_button['text'] = 'VAD not active'
            self.deactivate_VAD()

    def VAD_plot_btn(self):
        print("VAD_plot_btn")
        messagebox.showinfo(title="Warning", message="Method not yet implemented")
        # self.vad_plotter.plot()
        
    def activate_VAD(self):
        self.data["VAD"]["active"] = True
        self.save_settings()

    def deactivate_VAD(self):
        self.data["VAD"]["active"] = False
        self.save_settings()

    def check_update_results(self):
        should_update = False
        try:
            if not self.win_width == self.resultsList.winfo_width(): should_update = True
            if not self.win_height == self.resultsList.winfo_height(): should_update = True
        except:
            pass
        self.win_width  = self.resultsList.winfo_width()
        self.win_height = self.resultsList.winfo_height()
        if not len(self.analysed_items) == len(self.backend.analysed_items): should_update = True
        if should_update:
            # diff = len(self.backend.analysed_items) - len(self.analysed_items)
            # new_items = []
            # if diff == 1: new_items = [self.backend.analysed_items[-1]]
            # if diff > 1:  new_items = self.backend.analysed_items[:-diff]
            # for item in new_items:
            #     if item["modality"] == "L":
            #         self.add_outputsCell_text(item)
            #     else:
            #         self.add_outputsCell_speech(item)
            self.fill_results_list()
            self.analysed_items = self.backend.analysed_items.copy()
    
    def fill_results_list(self):
        # self.add_resultsList()
        for widget in self.resultsList.winfo_children():
            widget.destroy()
        for item in reversed(self.backend.analysed_items):
            if item["modality"] == "L":
                self.add_outputsCell_text(item, itemDir="end")
            else:
                self.add_outputsCell_speech(item, itemDir="end")

    def updateLoop(self):
        self.check_update_results()
        self.root.after(100, self.updateLoop)

if __name__== "__main__":
    visualise = Visualise()


