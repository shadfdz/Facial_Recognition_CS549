# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
  - Might require chmod +x to get permission
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install requirements `pip3 install -r requirements.txt`

## Update versions
`pip-compile --output-file=requirements.txt requirements.in --upgrade`
