
# Project Setup Instructions

To replicate our experiments or to analyze our results, please ensure to fill in the necessary API keys and other configurations by creating a .env file (see .sample.env) - the .env is ignored in .gitignore for security.

Setup the python environment using either venv or pyenv or your favourite python environment amanger. Call the environment  anything you like.

Follow the steps below to set up the venv environment for the project:

## 1. Create the Environment
Navigate to your project directory using the `cd` command:

```bash
cd to/your/path
```

Then create a new environment with Python 3.11:

```bash
conda create -p ./venv python=3.11 -y
```

Once the virtual environment has been created, activate it:

```bash
conda activate ./venv
```

## 2. Modify Your `.env` File
Copy or modify your `.env` file to match the structure and variables provided in `.env.example`.

## 3. Install the Required Libraries
Use `pip` to install all the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 4. Select Your Kernel in Jupyter Notebook
When working in Jupyter Notebook, ensure that you select the Python kernel from the environment you just created. The path to the kernel will be something like:

```
/your/path/to/venv/bin/python
```
