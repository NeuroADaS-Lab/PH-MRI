import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

from config import *

class PH_MRI:
    def __init__(self, data, target) -> None:
        self.diagrams = None
        self.X = None
        self.homology_dimensions = [0, 1, 2]
        self.data = data
        self.target = target

    ### Persistent Homology
    def compute_PH(self):
        VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=self.homology_dimensions, n_jobs=-1)
        self.diagrams = VR.fit_transform(self.data)
        print(f"diagrams.shape: {self.diagrams.shape} ({self.diagrams.shape[1]} topological features)")

        # plot diagram
        plot_diagram(self.diagrams[0], plotly_params={"layout": {"width": 1000, "height": 750, "margin": {"l": 10, "t": 10, "r": 10, "b": 10}, "font": {"size": 40}}})

    def compute_Betti_curves(self):
        ### Betti curves
        b = BettiCurve()
        self.X = b.fit_transform(self.diagrams)
        print("X shape is {}".format(self.X.shape))

        # plot Betti curves
        b.plot(self.X, sample=1)

        self.th_0 = b.samplings_[0]
        self.th_1 = b.samplings_[1]
        self.th_2 = b.samplings_[2]

        self.th_0_min = np.argwhere(self.th_0 >= 0.0).min()
        self.th_0_max = np.argwhere(self.th_0 <= 1.0).max()

        self.th_1_min = np.argwhere(self.th_1 >= 0.0).min()
        self.th_1_max = np.argwhere(self.th_1 <= 1.0).max()

        self.th_2_min = np.argwhere(self.th_2 >= 0.0).min()
        self.th_2_max = np.argwhere(self.th_2 <= 1.0).max()

    def plot_Betti_curves_one_subject(self, subject:int):
        # sample plot of subject 0
        plt.plot(self.th_0[self.th_0_min:self.th_0_max], self.X[subject, 0, self.th_0_min:self.th_0_max], label="H0", color="#EF553B")
        plt.plot(self.th_1[self.th_1_min:self.th_1_max], self.X[subject, 1, self.th_1_min:self.th_1_max], label="H1", color="#00CC96")
        plt.plot(self.th_2[self.th_2_min:self.th_2_max], self.X[subject, 2, self.th_2_min:self.th_2_max], label="H2", color="#AB63FA")
        plt.title("Subject {}".format(subject))
        plt.xlabel("Filtration parameter")
        plt.ylabel("Betti number")
        plt.legend()
        plt.show()

    def plot_Betti_curves_all_subjects(self, dimension):
        # dimension
        if dimension == 0:
            th = self.th_0
            th_min = self.th_0_min
            th_max = self.th_0_max
            title = "H0"
        elif dimension == 1:
            th = self.th_1
            th_min = self.th_1_min
            th_max = self.th_1_max
            title = "H1"
        elif dimension == 2:
            th = self.th_2
            th_min = self.th_2_min
            th_max = self.th_2_max
            title = "H2"
        else:
            print("Error: dimension = {}".format(dimension))
            sys.exit(0)

        # plot for each subject
        for i in range(self.X.shape[0]):
            plt.plot(th[th_min:th_max], self.X[i, 0, th_min:th_max], c="blue" if self.target[i] else "red")

        plt.title(title)
        plt.show()

    def plot_Betti_curves_mean(self, dimension, connectivity_name):
        X_0 = self.X[self.target == 0, :, :]
        X_1 = self.X[self.target == 1, :, :]

        if dimension == 0:
            th = self.th_0[self.th_0_min:self.th_0_max]

            mean_X0 = X_0[:, 0, self.th_0_min:self.th_0_max].mean(axis=0)
            std_X0 = X_0[:, 0, self.th_0_min:self.th_0_max].std(axis=0)
            se_X0 = 1.96*std_X0 / np.sqrt(X_0.shape[0])

            mean_X1 = X_1[:, 0, self.th_0_min:self.th_0_max].mean(axis=0)
            std_X1 = X_1[:, 0, self.th_0_min:self.th_0_max].std(axis=0)
            se_X1 = 1.96*std_X1 / np.sqrt(X_1.shape[0])

        elif dimension == 1:
            th = self.th_1[self.th_1_min:self.th_1_max]

            mean_X0 = X_0[:, 1, self.th_1_min:self.th_1_max].mean(axis=0)
            std_X0 = X_0[:, 1, self.th_1_min:self.th_1_max].std(axis=0)
            se_X0 = 1.96*std_X0 / np.sqrt(X_0.shape[0])

            mean_X1 = X_1[:, 1, self.th_1_min:self.th_1_max].mean(axis=0)
            std_X1 = X_1[:, 1, self.th_1_min:self.th_1_max].std(axis=0)
            se_X1 = 1.96*std_X1 / np.sqrt(X_1.shape[0])

        elif dimension == 2:
            th = self.th_2[self.th_2_min:self.th_2_max]

            mean_X0 = X_0[:, 2, self.th_2_min:self.th_2_max].mean(axis=0)
            std_X0 = X_0[:, 2, self.th_2_min:self.th_2_max].std(axis=0)
            se_X0 = 1.96*std_X0 / np.sqrt(X_0.shape[0])

            mean_X1 = X_1[:, 2, self.th_2_min:self.th_2_max].mean(axis=0)
            std_X1 = X_1[:, 2, self.th_2_min:self.th_2_max].std(axis=0)
            se_X1 = 1.96*std_X1 / np.sqrt(X_1.shape[0])
            
        else:
            print("Error: dimension = {}".format(dimension))
            sys.exit(0)

        # Mean and STD for each dimension
        plt.plot(th, mean_X0, c="red", label="MS")
        plt.fill_between(th, mean_X0 - se_X0, mean_X0 + se_X0, color="red", alpha=.1)

        plt.plot(th, mean_X1, c="blue", label="Healthy")
        plt.fill_between(th, mean_X1 - se_X1, mean_X1 + se_X1, color="blue", alpha=.1)

        plt.title("{} - H{}".format(connectivity_name, dimension))
        plt.xlim([0.0, 1.0])

        plt.legend()

        fname = "{}_H{}.png".format(connectivity_name, dimension)
        plt.savefig(os.path.join(output_basepath, fname), dpi=300)
        plt.show()
