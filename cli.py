import click
import cv2 as cv
from iris import get_iris_band, IrisClassifier, hamming_distance
from filters import filters
import numpy as np

iris_classifier = IrisClassifier(filters)

@click.group()
def cli():
    pass
    
@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-o", type=click.Path(exists=False), help="Output file name")
@click.option("-v", default=False, show_default=True, help="Verbose", is_flag=True)
def iris_gen(filename, o, v):
    img = cv.imread(filename)
    iris, mask = get_iris_band(img)
    iris_code, mask_code, _ = iris_classifier.get_iris_code(iris, mask)
    
    filename = filename.rsplit('.', 1)[0]
    iris_code = np.array(iris_code, dtype=np.bool)
    mask_code = np.array(mask_code, dtype=np.bool)
    
    # Store iris code in (2, 1, 2048) shape
    code = np.stack((iris_code, mask_code), axis=0)[:, np.newaxis, :]
    if v:
        click.echo("".join([str(i) for i in iris_code.astype(int)]))
        click.echo("".join([str(i) for i in mask_code.astype(int)]))
    if o is not None:
        np.save(o, code)
        if not v:
            click.echo("Saved iris code to " + o)
    else:
        np.save(filename, code)
        if not v:
            click.echo("Saved iris code to " + filename + ".npy")

@cli.command()
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
def iris_gens(filenames):
    iris_codes = []
    mask_codes = []
    
    N = len(filenames)
    for i, file in enumerate(filenames):
        click.echo(str(int((i/N)*100))+"%\r", nl=False)
        img = cv.imread(file)
        iris, mask = get_iris_band(img)
        iris_code, mask_code, _ = iris_classifier.get_iris_code(iris, mask)
        iris_codes.append(iris_code)
        mask_codes.append(mask_code)
    
    iris_codes = np.array(iris_codes, dtype=np.bool)
    mask_codes = np.array(mask_codes, dtype=np.bool)
    codes = np.stack((iris_codes, mask_codes), axis=0)
    np.save("iriscodes.npy", codes)
    click.echo("saved to iriscodes.npy")

@cli.command()
@click.argument("filename1", type=click.Path(exists=True))
@click.argument("filename2", type=click.Path(exists=True))
@click.option("--rotation", type=click.INT, default=21, show_default=True)
def compare(filename1, filename2, rotation):
    img1 = cv.imread(filename1)
    img2 = cv.imread(filename2)
    iris1, mask1 = get_iris_band(img1)
    iris2, mask2 = get_iris_band(img2)
    score, _ = iris_classifier(iris1, iris2, mask1, mask2, rotation=rotation)
    click.echo(score)

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("code_path", type=click.Path(exists=True))
@click.option("--rotation", type=click.INT, default=21, show_default=True)
def compare_iris_code(filename, code_path, rotation=21):
    img = cv.imread(filename)
    iris, mask = get_iris_band(img)
    code = np.load(code_path)
    if code.ndim == 3 and code.shape[0] == 2:
        # (2, N, L) or (2, 1, L) => pick the first sample
        iris_code = code[0, 0]
        mask_code = code[1, 0]
    else:
        # (2, L)
        iris_code = code[0]
        mask_code = code[1]
    score, _ = iris_classifier.compare_iris_code_and_iris(iris, iris_code, mask, mask_code)
    click.echo(score)
    
@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("codes_path", type=click.Path(exists=True))
@click.option("--rotation", type=click.INT, default=21, show_default=True)
@click.option("--threshold", default=0.3, show_default=True, help="stops searching when lower then threshold")

def find(filename, codes_path, rotation, threshold):
    codes = np.load(codes_path)
    
    img = cv.imread(filename)
    iris, mask = get_iris_band(img)
    
    iris_code, mask_code, _ = iris_classifier.get_iris_code(iris, mask, offset=0)
    
    iris_codes = np.zeros((rotation, iris_code.shape[0]), dtype=np.bool)
    mask_codes = np.zeros((rotation, iris_code.shape[0]), dtype=np.bool)
    
    for i in range(rotation):
        iris_codes[i], mask_codes[i], _ = iris_classifier.get_iris_code(iris, mask, offset=i-rotation//2)
    
    best_match = None
    score = 1.0
    
    for j in range(codes.shape[1]):
        for i in range(rotation):
            curr_score = hamming_distance(iris_codes[i], codes[0,j], mask_codes[i], codes[1,j])
            if curr_score < score:
                score = curr_score
                best_match = j
        if score < threshold:
            break
    click.echo("idx: " + str(best_match))
    click.echo("Score: " + str(score))
            
@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("codes_path", type=click.Path(exists=False))
def enroll(filename, codes_path):
    # Read image and compute iris code
    img = cv.imread(filename)
    iris, mask = get_iris_band(img)
    iris_code, mask_code, _ = iris_classifier.get_iris_code(iris, mask)

    # Ensure boolean arrays and build (2, L)
    iris_code = np.array(iris_code, dtype=bool)
    mask_code = np.array(mask_code, dtype=bool)
    code = np.stack((iris_code, mask_code), axis=0)

    try:
        # Try to load an existing database (expected shape (2, N, L))
        codes = np.load(codes_path)
        # Append new sample as (2, 1, L)
        codes = np.concatenate((codes, code[:, np.newaxis, :]), axis=1)
        idx = codes.shape[1] - 1
        
    except FileNotFoundError:
        # Create a new database with the first sample as (2, 1, L)
        codes = code[:, np.newaxis, :]
        idx = 0

    np.save(codes_path, codes)
    click.echo("Enrolled iris into " + codes_path)
    click.echo("idx: " + str(idx))


if __name__ == '__main__':
    cli()
