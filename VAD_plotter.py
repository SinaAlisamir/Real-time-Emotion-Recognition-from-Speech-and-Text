import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from settings import Settings

class VAD_plotter():
    def __init__(self, backend, update_sec=2.25, plot_freq=10, max_time_sec=45*60):
        super().__init__()
        self.backend = backend
        self.duration = Settings.buffer_max
        self.duration_sec = Settings.buffer_sec
        self.plot_freq = plot_freq
        self.plot_duration = self.plot_freq * self.duration_sec
        self.update_sec = update_sec
        self.interval = self.update_sec*1000
        self.max_time_sec = max_time_sec
        self.max_time = int(max_time_sec/update_sec)
        self.init_plot()

    def get_plot_values(self):
        # self.audio_sig = np.random.rand(Settings.buffer_max//2)
        # self.audio_sig = np.append(np.zeros(Settings.buffer_max//4),np.random.rand(Settings.buffer_max//4))
        # self.vad_preds = np.random.rand(Settings.buffer_max//2)
        # self.vad_outs  = np.random.rand(Settings.buffer_max//2)
        self.audio_sig = self.backend.buffer
        self.vad_preds = self.backend.VADFeatsModule.outs
        self.vad_outs  = self.backend.VADFeatsModule.outs_pp

    def init_plot(self):
        self.fig = plt.figure()
        ax1 = plt.axes(xlim=(-self.duration_sec, 0), ylim=(-1.1,1.1))
        
        plotlays, plotcols = [3], ["black", "red", "green"]
        self.lines = []
        for index in range(3):
            lobj = ax1.plot([],[],lw=2,color=plotcols[index])[0]
            self.lines.append(lobj)

        x1,y1 = [],[]

        plt.legend(["Audio Input", "VAD Preds", "VAD Output"])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    def init(self):
        for line in self.lines:
            line.set_data([],[])
        return self.lines
        
    def PlotBufferAnim(self, i):
        # print("here1.5")
        try:
            self.get_plot_values()
            duration = self.plot_duration#len(self.audio_sig)
            # vadOuts = loadVAD(VADFolder)
            # print(buffer.shape, vadOuts.shape)
            self.audio_sig = np.interp(np.linspace(0, self.audio_sig.shape[0], duration), np.arange(0, len(self.audio_sig), 1), self.audio_sig)
            self.vad_preds = np.interp(np.linspace(0, self.vad_preds.shape[0], duration), np.arange(0, len(self.vad_preds), 1), self.vad_preds)
            self.vad_outs = np.interp(np.linspace(0, self.vad_outs.shape[0], duration), np.arange(0, len(self.vad_outs), 1), self.vad_outs)
            # print(buffer.shape, vadOuts.shape)
            # X = list(range(size))
            X = np.linspace(-self.duration_sec, 0, len(self.audio_sig))

            xlist = [X, X, X]
            ylist = [self.audio_sig, self.vad_preds, self.vad_outs] #vadOuts
            for lnum,line in enumerate(self.lines): line.set_data(xlist[lnum], ylist[lnum])
            # print("here3")
        except:
            pass
        return self.lines

    def pad_if_smaller(self, arr, duration):
        new_arr = arr.copy()
        diff = duration - len(new_arr)
        if diff > 0:
            new_arr = np.append(np.zeros(diff), new_arr)
        return new_arr

    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.PlotBufferAnim, 
                                        self.max_time, init_func=self.init,
                                        interval=self.interval, blit=False)
        plt.show()

# plotter = VAD_plotter()
# plotter.plot()