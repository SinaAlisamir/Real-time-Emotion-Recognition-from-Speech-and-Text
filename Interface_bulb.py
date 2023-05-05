# from tkinter import *
import tkinter
from tkinter import messagebox
from tkinter import Tk
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk
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
import asyncio
# from pywizlight import wizlight, PilotBuilder, discovery
import requests
# from huesdk import Hue
from phue import Bridge

class Visualise():
    def __init__(self):
        super().__init__()
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        sys.setrecursionlimit(50000)
        self.bulb_ips = []
        self.hue_lights = []
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

        self.add_gifShow()
        self.add_optionMenus()
        self.add_resultsList()
        self.add_bulb_interface()

    def add_bulb_interface(self):
        self.bulb_checked = tkinter.IntVar()
        self.bulb_checkbox = tkinter.Checkbutton(self.root, text="Connect to smart bulb", variable=self.bulb_checked)
        self.bulb_checkbox.place(relx=.14, rely=.34, relwidth=.2, relheight=.05, anchor="center")

    def add_gifShow(self):
        self.gif_show = False
        self.gif_counter = 0
        self.total_counter = 0
        self.frameCnt = 15
        file_dir = os.path.dirname(__file__)
        self.gif_frames_happy = [PhotoImage(file=os.path.join(file_dir,"Pics", "laugh.gif"),format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
        self.gif_frames_content = [PhotoImage(file=os.path.join(file_dir,"Pics", "smile4.gif"),format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
        self.gif_frames_neutral = [PhotoImage(file=os.path.join(file_dir,"Pics", "neutral.gif"),format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
        self.gif_frames_sad = [PhotoImage(file=os.path.join(file_dir,"Pics", "sad.gif"),format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
        self.gif_frames_angry = [PhotoImage(file=os.path.join(file_dir,"Pics", "angry2.gif"),format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
        self.label_gif = tkinter.Label(self.root)
        # frame = self.gif_frames[0]
        # # image = Image.open('Pics/smily.png')
        # # image = image.resize((85,85), Image.ANTIALIAS)
        # # frame = ImageTk.PhotoImage(image)
        # self.label_gif.image = frame
        # self.label_gif.configure(image=frame)
        # self.label_gif.place(relx=.87, rely=.08, relwidth=.15, relheight=.15, anchor="center")

    def update_gif(self):    
        if self.gif_show:
            if self.backend.labels["pred"] == "happy": 
                self.gif_frames = self.gif_frames_happy
            if self.backend.labels["pred"] == "content": 
                self.gif_frames = self.gif_frames_content
            if self.backend.labels["pred"] == "neutral": 
                self.gif_frames = self.gif_frames_neutral
            if self.backend.labels["pred"] == "sad": 
                self.gif_frames = self.gif_frames_sad
            if self.backend.labels["pred"] == "angry": 
                self.gif_frames = self.gif_frames_angry
            self.total_counter += 1
            self.gif_counter += 1
            if self.gif_counter >= self.frameCnt: self.gif_counter = 0
            frame = self.gif_frames[self.gif_counter].subsample(5)
            self.label_gif.image = frame
            self.label_gif.configure(image=frame)
            self.label_gif.place(relx=.87, rely=.08, relwidth=.15, relheight=.15, anchor="center")
            if self.total_counter > 50:
                self.total_counter = 0
                self.gif_show = False
                self.label_gif.image.blank()
                self.label_gif.image = None
            
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
        rly = .65
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
        self.gif_show = True
        print("bulb activated?", self.bulb_checked.get())
        if self.bulb_checked.get() == 1:
            loop = asyncio.get_event_loop()
            # loop.run_until_complete(self.bulb_func_wiz())
            loop.run_until_complete(self.bulb_func_hue())

    async def bulb_func_hue(self):#async 
        if self.hue_lights == []:
            bridge_ip = "192.168.14.1"
            print("bridge_ip", bridge_ip)
            b = Bridge(bridge_ip)
            print("bridge username:", b.username)
            self.bridge = b
            print("bridge lights", b.lights)
            light_ids_dict = b.get_light_objects('id')
            self.hue_lights = list(light_ids_dict.keys())
            print("bridge light ids", self.hue_lights)
            # b.set_light(17, 'hue', 65535//2)

        for hue_light_id in self.hue_lights:
            arousal = self.backend.labels["probs"]["arousal"]
            valence = self.backend.labels["probs"]["valence"]
            sentiment = self.backend.labels["probs"]["sentiment"]
            brightness = min(250, int(250*arousal)) # max is 254
            # hue_light.set_brightness(brightness)# await 
            self.bridge.set_light(hue_light_id, 'bri', brightness)
            if sentiment > 0.8:
                self.bridge.set_light(hue_light_id, 'hue', 65535//3)
            elif sentiment < 0.2:
                self.bridge.set_light(hue_light_id, 'hue', 0)
            else:
                self.bridge.set_light(hue_light_id, 'hue', 65535//6)

    def bulb_func_hue_old(self):#async 
        if self.hue_lights == []:
            page = requests.get("http://discovery.meethue.com/")
            page_content_str = page.content
            # print("page_content_str1", page_content_str)#10.0.1.2
            # page_content_str = "[{\"id\":\"xxx\",\"internalipaddress\":\"192.168.1.10\",\"port\":443}]\n"
            page_content_str = page_content_str.decode("utf-8").replace("\n", "")
            # print("page_content_str2", page_content_str)
            res = json.loads(page_content_str)
            bridge_ip = res[0]["internalipaddress"]
            # bridge_ip = "10.0.1.2"
            # print("bridge_ip", bridge_ip)
            try:
                username = Hue.connect(bridge_ip=bridge_ip)
            except :
                pass
            print("Hue bridge username", username)
            self.hue = Hue(bridge_ip=bridge_ip, username=username)
            self.hue_lights = self.hue.get_lights()
            print("number of hue lights deteceted:", len(self.hue_lights))
        for hue_light in self.hue_lights:
            arousal = self.backend.labels["probs"]["arousal"]
            valence = self.backend.labels["probs"]["valence"]
            sentiment = self.backend.labels["probs"]["sentiment"]
            brightness = min(200, int(200*arousal)) # max is 254
            hue_light.set_brightness(brightness)# await 
            if sentiment > 0.8:
                hex_col = "#" + hex(180)[2:] + hex(243)[2:] + hex(12)[2:]
                hue_light.set_color(hexa=hex_col)# await 
            elif sentiment < 0.2:
                hex_col = "#" + hex(243)[2:] + hex(43)[2:] + hex(12)[2:]
                hue_light.set_color(hexa=hex_col)# await 
            else:
                hex_col = "#" + hex(125)[2:] + hex(125)[2:] + hex(125)[2:]
                hue_light.set_color(hexa=hex_col)# await 
            

    async def bulb_func_wiz(self):
        if self.bulb_ips == []:
            bulbs = await discovery.discover_lights(broadcast_space="192.168.1.255")
            print("Bulb detected?", bulbs)
            print(f"Bulb IP address: {bulbs[0].ip}")
            self.bulb_ips = [bulbs[0].ip]
            # self.bulb_ips = ["10.0.1.2"]
        for bulb_ip in self.bulb_ips:
            self.myBulb = wizlight(bulb_ip)
            arousal = self.backend.labels["probs"]["arousal"]
            valence = self.backend.labels["probs"]["valence"]
            sentiment = self.backend.labels["probs"]["sentiment"]
            brightness = min(100, int(100*arousal))
            if sentiment > 0.8:
                await self.myBulb.turn_on(PilotBuilder(rgb = (180, 243, 12), brightness = brightness))
            elif sentiment < 0.2:
                await self.myBulb.turn_on(PilotBuilder(rgb = (243, 43, 12), brightness = brightness))
            else:
                await self.myBulb.turn_on(PilotBuilder(rgb = (125, 125, 125), brightness = brightness))

    def updateLoop(self):
        self.check_update_results()
        self.update_gif()
        self.root.after(100, self.updateLoop)

if __name__== "__main__":
    # import urllib3
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    visualise = Visualise()


