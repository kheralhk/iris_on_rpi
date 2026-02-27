# Iris Recognition CLI

A command-line tool for iris code generation, comparison, and management using OpenCV and custom iris recognition algorithms.

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd <repo_directory>
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venvScriptsactivate
pip install -r requirements.txt
```

3. Compile wahet from [here](https://github.com/ngoclamvt123/usit-v2.2.0)

4. Put the wahet executable into your path variable or in the current directory

## Usage

The CLI provides several commands for iris recognition tasks.

### Generate Iris Code from a Single Image

```bash
python cli.py iris_gen <filename> [-o <output_file>] [-v]
```

- `<filename>`: Path to the input image.
- `-o`: Optional. Specify output `.npy` file for the iris code.
- `-v`: Verbose. Prints the iris code and mask code to the terminal.

Example:

```bash
python cli.py iris_gen images/eye1.jpg -o iris_code1.npy
```

### Generate Iris Codes from Multiple Images

```bash
python main.py iris_gens <filename1> <filename2> ...
```

- Generates iris codes for multiple images and saves them to `iriscodes.npy`.

Example:

```bash
python cli.py iris_gens images/*.png
```

### Compare Two Images

```bash
python cli.py compare <filename1> <filename2> [--rotation <rotation>]
```

- Compares two iris images and outputs the Hamming distance.
- Default rotation: 21

### Compare Image Against an Iris Code

```bash
python cli.py compare_iris_code <filename> <code_path> [--rotation <rotation>]
```

- Compares an iris image to a stored iris code.

### Find Best Match in Iris Database

```bash
python cli.py find <filename> <codes_path> [--rotation <rotation>] [--threshold <threshold>]
```

- Searches a database of iris codes and returns the best match index and score.
- Default rotation: 21
- Default threshold: 0.3

### Enroll a New Iris Code

```bash
python cli.py enroll <filename> <codes_path>
```

- Adds a new iris code to the existing database.

## Notes

- Input images must exist at the specified paths.
- Iris codes and masks are stored as NumPy boolean arrays.
- Ensure the `filters` and `iris` modules are correctly installed and accessible.

## Example Workflow

1. Generate iris codes for a dataset:

```bash
python cli.py iris_gens dataset/*.jpg
```

2. Compare a new image to the database:

```bash
python cli.py find new_image.jpg iriscodes.npy
```

3. Enroll a new iris:

```bash
python cli.py enroll new_image.jpg iriscodes.npy
```
