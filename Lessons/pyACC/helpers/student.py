import os
import re
from urllib import request

from .logger import Logger

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size" : 13})
from astropy.cosmology import FlatLambdaCDM, wCDM

class Student:

    def __init__(self, name, branch_name, folder_name):
        """
        Constructor of the class Student.

        Parameters
        ----------
        name : str
            Name of the student.
        branch_name : str
            Name of the branch.
        folder_name : str
            Name to the folder where the student's files are located.
        """

        self.name = name
        self.logger = Logger(self.name)

        self.branch_name = branch_name
        self.remote_folder_name = folder_name
        self.local_folder_name = re.sub('[^a-zA-Z0-9]', '', self.name)
        self.raw_git_url = f"https://raw.githubusercontent.com/GenovaCosmology/AstroCosmoComputational/{self.branch_name}/Students/{self.remote_folder_name}/"

    def get_file(self, local_path, file_path, file_name, overwrite=False):
        """
        This method downloads the file from the student's repository.

        Parameters
        ----------
        local_path : str
            Local path to save the file.
        file_path : str
            Path to the file in the student's repository.
        file_name : str
            Name of the file.
        overwrite : bool, optional
        """
        local_path = local_path+f"/{self.local_folder_name}/"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            self.logger(f"The folder {local_path} has been created.")

        success = False

        if not os.path.exists(local_path+file_name) or overwrite:
            try:
                request.urlretrieve(self.raw_git_url+file_path+"/"+file_name, local_path+file_name)
                success = True
            except:
                self.logger.error(f"The file {file_name} could not be downloaded from {self.raw_git_url+file_path}/{file_name}.")
        else:
            self.logger.warning(f"The file {file_name} already exists. Manually remove it to download it again.")

        if success:
            self.logger(f"The file {file_name} has been downloaded from {self.raw_git_url+file_path}/{file_name}.")

        return local_path+file_name, success
    

class ComovingDistanceChallenge:

    def __init__(self, student_list):
        """ 
        Initialize the class

        Parameters:
        -----------
        student_list : dict
            Dictionary containing the students' names and their branch names on git repo AstroCosmoComputational on GitHub
        """
        # Save the student list
        self.student_list = student_list
        
        # Compute reference quantities
        self.z_ref = np.arange(0, 2, 0.01) + 0.01/2
        self.dc_LCDM_ref = FlatLambdaCDM(H0=67, Om0=0.319).comoving_distance(self.z_ref).value
        self.dc_wCDM_ref = wCDM(H0=67, Om0=0.319, Ode0=1-0.319, w0=-1.1).comoving_distance(self.z_ref).value

        # Prepare private object 
        self.results = {}

    def download_data(self):
        """ 
        Download the data from the students' repositories
        """
        for student in self.student_list:
            self.results[student] = [*Student(student, self.student_list[student][0], self.student_list[student][1]).get_file("data", "Lesson2/Exercise3", "comoving_distances.txt", overwrite=True)]
            print()

        self.logger = Logger("Comoving Distance Challenge")

    def read_files(self):
        """
        Read the files and store the results in the private object
        """

        for student in self.student_list:
            if not self.results[student][1]:
                self.logger.error(f"No file to read for {student} :((")
                continue

            z, dc_LCDM, dc_wCDM = np.genfromtxt(self.results[student][0], unpack=True)

            if not np.allclose(z, self.z_ref, 1.e-5):
                self.logger.error(f"z values for {student} are not correct :(")
                continue

            if np.sum(np.isnan(dc_LCDM))!=0 or np.sum(np.isnan(dc_wCDM))!=0:
                self.logger.error(f"Distances for {student} contains NaNs :(")
                continue

            self.results[student] += [dc_LCDM, dc_wCDM]

    def plot(self):
        """ 
        Plot the results

        Returns:
        --------
        ax : list of matplotlib axes
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        for student in self.student_list:
            if len(self.results[student])==2:
                continue

            ax[0].plot(self.z_ref, self.results[student][2], label=student)
            ax[1].plot(self.z_ref, self.results[student][3], label=student)

        for i in range(2):
            ax[i].grid(True, ls="--", alpha=0.7)
            ax[i].set_xlabel(r"$z$")
            ax[i].set_ylabel(r"$D_C$ [Mpc]")
            ax[i].legend(loc="upper left")

        ax[0].set_title(r"$\Lambda$CDM")
        ax[1].set_title(r"wCDM")

        return ax

    def print_residuals(self):
        """
        Print the residuals for each student
        """
        for student in self.student_list:
            if len(self.results[student])==2:
                continue

            res_lcdm = np.mean(np.abs(self.results[student][2]/self.dc_LCDM_ref-1)*100)
            res_wcdm = np.mean(np.abs(self.results[student][3]/self.dc_wCDM_ref-1)*100)

            msg = "Residuals for student "+student+":\n"
            msg += f"LCDM: %.2f percent - wCDM: %.2f percent"%(res_lcdm, res_wcdm)
            self.logger(msg)

    def plot_residuals(self):
        """ 
        Plot the results

        Returns:
        --------
        ax : list of matplotlib axes
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        for student in self.student_list:
            if len(self.results[student])==2:
                continue

            ax[0].plot(self.z_ref, np.abs(self.results[student][2]/self.dc_LCDM_ref-1)*100, label=student)
            ax[1].plot(self.z_ref, np.abs(self.results[student][3]/self.dc_wCDM_ref-1)*100, label=student)

        for i in range(2):
            ax[i].grid(True, ls="--", alpha=0.7)
            ax[i].set_yscale("log")
            ax[i].set_xlabel(r"$z$")
            ax[i].set_ylabel(r"$|\Delta D_C/D_C|$ [\%]")
            ax[i].legend(loc="upper left")

        ax[0].set_title(r"$\Lambda$CDM")
        ax[1].set_title(r"wCDM")

        return ax