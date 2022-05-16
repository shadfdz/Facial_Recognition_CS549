# PythonProject
[Github](https://github.com/shadfdz/Facial_Recognition_CS549.git)

# Environment setup for project (use pip 22.0.4):
- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
  - Might require chmod +x to get permission
- `pip3 install --upgrade pip=22.0.4`
- Install pip-tools `pip3 install pip-tools`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install requirements `pip3 install -r requirements.txt`

## Update versions
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Project structure:
- Create an output folder and dataset folder
- Store video files in datasetfolder

# Project Workflow
- run python3 emotion_classifier.py to detect faces and classify the emotion
- run pyhotn3 emotion_analysis.py to output analysis plots
