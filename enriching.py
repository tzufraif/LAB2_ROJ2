import os
import random

import pandas as pd
import torch
import shutil
from tqdm import tqdm
from PIL import Image, ImageFilter


def generate_samples(num, num_samples=10000, sample_ratio=100):
    """
    Generate new images by applying transformations
    :param num: experiment number
    :param num_samples: number of samples to generate,default 10000
    :param sample_ratio: number of samples per class, default 100
    :return:
    """
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    assert num_samples % sample_ratio == 0

    # create a folder for train
    os.mkdir(os.path.join('data', f'train{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'train{num}', cls))
    # val folders by class
    os.mkdir(os.path.join('data', f'val{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'val{num}', cls))

    # --------------------------------------------------------------------------

    train_dir_read = os.path.join("data", "train")
    train_dir_write = os.path.join("data", f'train{num}')


    dir_read = train_dir_read
    dir_write = train_dir_write

    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(dir_read, cls)
        images = os.listdir(path)
        images = [x for x in images if 'png' in x]
        sampled_images = random.sample(images, sample_ratio)
        generated_counter = 0
        for img in sampled_images:  # loops for sample_ratio images
            img_path = os.path.join(path, img)

            # effect - 4 images
            for spread in range(20, 60, 10):
                imgObj = Image.open(img_path)
                imgObj.effect_spread(spread)
                img_name = f'{img}_effect_{spread}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1

            # box blur - 1 image

            imgObj = Image.open(img_path)
            imgObj.filter(filter=ImageFilter.BLUR)
            img_name = f'{img}_blur.png'
            imgObj.save(os.path.join(dir_write, cls, img_name))
            generated_counter += 1

            # rotation - 5
            angles = []
            used_angles = set()
            for _ in range(5):
                angle = random.randint(1, 359)
                while angle in used_angles:
                    angle = random.randint(1, 359)
                used_angles.add(angle)
                angles.append(angle)

            for angle in angles:
                imgObj = Image.open(img_path)
                imgObj = imgObj.rotate(angle=angle)
                img_name = f'{img}_rotate_{angle}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1


def generate_val_images(num, val_ratio=0.15):
    """
    Generates vals from train images
    :param num: experiment number
    :param val_ratio: ratio of the val images
    :return:
    """
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    # create folders

    train_dir_read = os.path.join("data", f'train{num}')
    val_dir_write = os.path.join("data", f'val{num}')

    moved = 0

    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(train_dir_read, cls)
        images = os.listdir(path)
        sampled_images = random.sample(images, int(len(images) * val_ratio))
        for file_name in sampled_images:
            moved += 1
            # move files from train to val folder
            shutil.move(os.path.join(path, file_name), os.path.join(val_dir_write, cls))

    print(f'{moved} images moved successfully!')


def count_images(dir):
    """
    Counts the number of images per class and export to csv
    :param dir: direction of the images
    """
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    lens = []
    for cls in classes:
        path = os.path.join('data', dir, cls)
        images = os.listdir(path)
        lens.append(len(images))
    df = pd.DataFrame({
        'class': classes,
        'num_images': lens
    })
    print(df)
    print(f'Total number of {dir} images: {sum(lens)}')
    df.to_csv(f'{dir}_img.csv')


if __name__ == '__main__':
    num = 4
    generate_samples(num=num, num_samples=10000)
    generate_val_images(num=num, val_ratio=0.15)
    count_images(dir=f'train{num}')
    count_images(dir=f'val{num}')
