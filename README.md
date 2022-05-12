# NNImageProcessing
This is a program that can be utilized to segment and measure fission yeast cells.

# Creating a virtual environment
1. If python is not installed, install python from https://www.python.org/
2. Open up terminal (if using a Windows machine, donwload the terminal from https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701?hl=en-us&gl=US)
3. Create a new folder to house the virtual environment and change the directory of the terminal to the folder.
4. Enter in this command in the terminal: py -m venv env (env = name of virtual environment. Change it to suit one's needs)
5. Enter in this command to activate the env: .\env\Scripts\activate (change env to name of environment that one set up in step 4)
6. To deactivate the environment simply enter in this command: deactivate

# Instructions
1. Create a python virtual environment with the libraries from the [requirements.txt]
2. Activate the environment and set the terminal path to the directory where all of the .tif microscope images will take place.
3. Enter in the command: "python process_images.py *.tif" (You can pass other arguments in, check the code description for more information) (*.tif command only works on MacOS. Check the Windows instruction text file)
4. Wait for the csv to pop up from the photos and take those files to the R scripts in NNImageProcessing_Stats repository
