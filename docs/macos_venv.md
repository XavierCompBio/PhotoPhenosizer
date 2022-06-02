# Creating MacOS python virtual environment

## Installation Guide
1. Make sure that python is downloaded from: https://www.python.org
2. Open up terminal and set its directory to the folder that you want to store your virtual environments:
```
cd "[path of directory here]"
```
4. Enter the following commands:
```
python3 -m venv [name of environment]
```
5. This should create a folder of the name of the environment in the directory that you specified in step 2
6. To activate the environment use this command:
```
source [name of environment]/bin/activate
```
7. Should see the command prompt input line change to the correct environment.

## Reference
* [venv documentation](https://docs.python.org/3/library/venv.html)
