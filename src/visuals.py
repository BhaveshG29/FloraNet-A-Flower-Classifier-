import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import json
import numpy as np

# Loads the metric.json file from cache folder.
def data_loader(path:str):
    with open(path, "r") as f:
        data = json.load(f)
        
    loss_hist = data["Loss"]
    acc_hist = data["Accuracy"]
    f1_hist = data["F1 Score"]
    
    return [loss_hist, acc_hist, f1_hist]


class visuals:
    def __init__(self, load_path:str = "../cache/metric.json") -> None:
        self.metrics = data_loader(path=load_path)
    
    #Plots the All the Metric Curves in one Figure
    def curve_plotter(self, show_plot=True, save_plot=True):
        fig = plt.figure(figsize=(14,7))

        ax1 = plt.subplot(1,3,1) 
        y1 = np.array(self.metrics[0])[:-1]
        x1 = np.arange(len(y1))
        val_y1 = self.metrics[0][-1]

        ax1.plot(x1, y1, c="r", label="Loss Curve", lw=2)
        ax1.scatter(len(y1)+1, val_y1, c="orange", marker="*", label="Validating Loss", zorder=5, s=50)

        ax1.set_title("Loss", fontweight="bold", fontsize=15)
        ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
        ax1.set_xlabel("Iterations", fontsize=12)
        ax1.grid(True)
        ax1.legend()


        ax2 = plt.subplot(1,3,2)
        y2 = np.array(self.metrics[1])[:-1]
        x2 = np.arange(len(y2))
        val_y2 = self.metrics[1][-1]

        ax2.plot(x2, y2, c="g", lw=2, label="Accuracy Curve")
        ax2.scatter(len(y2)+1, val_y2, c="b", marker="*", label="Validating Accuracy", s=50, zorder=5)

        ax2.set_title("Accuracy", fontweight="bold", fontsize=15)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_xlabel("Iterations", fontsize=12)
        ax2.grid(True)
        ax2.legend()


        ax3 = plt.subplot(1,3,3)
        y3 = np.array(self.metrics[2])[:-1]
        x3 = np.arange(len(y3))
        val_y3 = self.metrics[2][-1]

        ax3.plot(x3, y3, c="g", lw=2, label="F1-Score Curve")
        ax3.scatter(len(y3)+1, val_y3, c="b", marker="*", label="Validating F1-Score", s=50, zorder=5)

        ax3.set_title("F1-Score", fontweight="bold", fontsize=15)
        ax3.set_ylabel("F1-Score", fontsize=12)
        ax3.set_xlabel("Iterations", fontsize=12)
        ax3.grid(True)
        ax3.legend()

        if save_plot:
            plt.savefig("../visuals/metrics.png", dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

    @staticmethod
    def update_frames(frame, line, data):
        x = np.arange(frame)
        y = data[:frame]
        line.set_data(x, y)
        return line,
    
    #Wrapper Function for Animating the Metrics
    def _animate_curve(self, data:List, title: str, x_label: str, y_label: str, save_path: str, color:str|None = "g", show_animation = True, save_animation=True):
        fig, ax = plt.subplots()
        (line,) = ax.plot([], [], lw=2, c=color)

        ax.set_title(title, fontweight="bold", fontsize=25)
        ax.set_xlabel(x_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)

        ax.set_xlim(0, len(data))

        y_min, y_max = min(data), max(data)
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        ax.set_ylim(y_min, y_max)

        anim = FuncAnimation(
            fig=fig,
            func=visuals.update_frames,
            frames=len(data),
            fargs=(line, data),
            interval=50,
            blit=True
        )

        if save_animation:
            anim.save(save_path, fps=15)

        if show_animation:
            plt.show()

        plt.close(fig)

    #Loss Animation
    def animate_loss(self, show_animation=True,save_animation=True):
        loss = self.metrics[0]
        loss.pop(-1)
        self._animate_curve(data=loss, title="Training Loss", x_label="Iterations", y_label="Cross Entropy Loss", save_path="../visuals/loss.gif", color="r", show_animation=show_animation, save_animation=save_animation)
    
    #Accuracy Animation
    def animate_accuracy(self,show_animation=True,save_animation=True):
        accuracy = self.metrics[1]
        accuracy.pop(-1)
        self._animate_curve(data=accuracy, title="Training Accuracy", x_label="Iterations", y_label="Accuracy", save_path="../visuals/accuracy.gif", show_animation=show_animation, save_animation=save_animation)
    
    #F1-Score Animation
    def animate_f1_score(self, show_animation=True,save_animation=True):
        f1_score = self.metrics[2]
        f1_score.pop(-1)
        self._animate_curve(data=f1_score, title="Training F1 Score", x_label="Iterations", y_label="F1 Score", save_path="../visuals/f1_score.gif", show_animation=show_animation, save_animation=save_animation)





if __name__ == "__main__":
    #Manually Plotting by running this file, if required
    v = visuals()

    v.curve_plotter()
    v.animate_loss(show_animation=False)
    v.animate_accuracy(show_animation=False)
    v.animate_f1_score(show_animation=False)







































