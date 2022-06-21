import os
import random

import PIL.Image
import pandas as pd
import torch
import shutil
from tqdm import tqdm
from PIL import Image, ImageFilter


def generate_samples(num, num_samples=10000, sample_ratio=100):
    """
    Generates images for model training
    :param num: experiment number
    :param num_samples: number of samples to generate
    :param sample_ratio: number of samples per class
    :return:
    """
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    assert num_samples % sample_ratio == 0

    # create folders
    # train folders
    os.mkdir(os.path.join('data', f'train{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'train{num}', cls))
    # val folders
    os.mkdir(os.path.join('data', f'val{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'val{num}', cls))

    # --------------------------------------------------------------------------

    train_dir_read = os.path.join("data", "train")
    train_dir_write = os.path.join("data", f'train{num}')

    # imgs_per_label = num_samples // sample_ratio

    dir_read = train_dir_read
    dir_write = train_dir_write
    img_counter = 0
    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(dir_read, cls)
        images = os.listdir(path)
        images = [x for x in images if 'png' in x]
        sampled_images = random.sample(images, sample_ratio)
        generated_counter = 0
        for img in sampled_images:  # loops for sample_ratio images
            img_path = os.path.join(path, img)
            # for each sampled image create 10 modified images

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


TRAIN_COUNT = 700


def generate_samples2(num, sample_ratio=0.1):
    """
    Generates images for model training
    :param num: experiment number
    :param num_samples: number of samples to generate
    :param sample_ratio: number of samples per class
    :return:
    """
    print('Generating Train Images...')
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

    # create folders
    # train folders
    os.mkdir(os.path.join('data', f'train{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'train{num}', cls))
    # val folders
    os.mkdir(os.path.join('data', f'val{num}'))
    for cls in classes:
        os.mkdir(os.path.join('data', f'val{num}', cls))

    orig_train_dir_read = os.path.join("data", f'train')
    train_dir_read = os.path.join("data", f'train{num}')
    val_dir_write = os.path.join("data", f'val{num}')

    # copy images from train to val
    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(orig_train_dir_read, cls)
        images = os.listdir(path)
        sampled_images = set(random.sample(images, int(len(images) * sample_ratio)))
        for file_name in sampled_images:
            # move files from train to val folder
            shutil.copyfile(os.path.join(path, file_name), os.path.join(val_dir_write, cls, file_name))
        for file_name in set(images).difference(sampled_images):
            shutil.copyfile(os.path.join(path, file_name), os.path.join(train_dir_read, cls, file_name))
    # return
    # --------------------------------------------------------------------------

    train_dir_write = os.path.join("data", f'train{num}')

    dir_read = train_dir_read
    dir_write = train_dir_write
    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(dir_read, cls)
        images = os.listdir(path)
        images = [x for x in images if 'png' in x]
        generated_counter = 0
        for img in images:  # loops for sample_ratio images
            img_path = os.path.join(path, img)
            # for each sampled image create 10 modified images

            # effect - 2 images
            for spread in range(20, 40, 10):
                imgObj = Image.open(img_path)
                imgObj.effect_spread(spread)
                img_name = f'{img}_effect_{spread}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1

            if generated_counter > TRAIN_COUNT:
                break
            # box blur - 1 image

            imgObj = Image.open(img_path)
            imgObj.filter(filter=ImageFilter.BLUR)
            img_name = f'{img}_blur.png'
            imgObj.save(os.path.join(dir_write, cls, img_name))
            generated_counter += 1

            if generated_counter > TRAIN_COUNT:
                break

            # rotation - 2
            angles = []
            used_angles = set()
            for _ in range(2):
                angle = random.randint(1, 359)
                while angle in used_angles and 150 < angle < 210:
                    angle = random.randint(1, 359)
                used_angles.add(angle)
                angles.append(angle)

            for angle in angles:
                imgObj = Image.open(img_path)
                imgObj = imgObj.rotate(angle=angle)
                img_name = f'{img}_rotate_{angle}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1

            if generated_counter > TRAIN_COUNT:
                break

            if cls in {'iv', 'vi'}:
                for angle in [-10, 0, 10]:
                    imgObj = Image.open(img_path)
                    imgObj = imgObj.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
                    imgObj = imgObj.rotate(angle=angle)
                    img_name = f'{img}_mirror_rotate_{angle}.png'
                    imgObj.save(os.path.join(dir_write, 'vi' if cls == 'iv' else 'iv', img_name))
                    generated_counter += 1

            if generated_counter > TRAIN_COUNT:
                break


VAL_COUNT = 115


def generate_val_images2(num):
    """
    Generates vals from train images
    :param num: experiment number
    :param val_ratio: ratio of the val images
    :return:
    """
    print('Generating Val Images...')
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    # create folders
    val_dir = os.path.join("data", f'val{num}')

    dir_read = val_dir
    dir_write = val_dir

    for cls in tqdm(classes, total=len(classes), ascii=False, ncols=100, desc='Generating...'):
        path = os.path.join(dir_read, cls)
        images = os.listdir(path)
        images = [x for x in images if 'png' in x]
        generated_counter = 0
        for img in images:  # loops for sample_ratio images
            img_path = os.path.join(path, img)
            # for each sampled image create 10 modified images

            # effect - 2 images
            for spread in range(20, 40, 10):
                imgObj = Image.open(img_path)
                imgObj.effect_spread(spread)
                img_name = f'{img}_effect_{spread}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1

            if generated_counter > VAL_COUNT:
                break

            # box blur - 1 image

            imgObj = Image.open(img_path)
            imgObj.filter(filter=ImageFilter.BLUR)
            img_name = f'{img}_blur.png'
            imgObj.save(os.path.join(dir_write, cls, img_name))
            generated_counter += 1

            if generated_counter > VAL_COUNT:
                break

            # rotation - 2
            angles = []
            used_angles = set()
            for _ in range(2):
                angle = random.randint(1, 359)
                while angle in used_angles and 150 < angle < 210:
                    angle = random.randint(1, 359)
                used_angles.add(angle)
                angles.append(angle)

            for angle in angles:
                imgObj = Image.open(img_path)
                imgObj = imgObj.rotate(angle=angle)
                img_name = f'{img}_rotate_{angle}.png'
                imgObj.save(os.path.join(dir_write, cls, img_name))
                generated_counter += 1

            if generated_counter > VAL_COUNT:
                break

            if cls in {'iv', 'vi'}:
                for angle in [-10, 0, 10]:
                    imgObj = Image.open(img_path)
                    imgObj = imgObj.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
                    imgObj = imgObj.rotate(angle=angle)
                    img_name = f'{img}_mirror_rotate_{angle}.png'
                    imgObj.save(os.path.join(dir_write, 'vi' if cls == 'iv' else 'iv', img_name))
                    generated_counter += 1

            if generated_counter > VAL_COUNT:
                break


if __name__ == '__main__':
    num = 5
    # generate_samples(num=num, num_samples=10000)
    # generate_val_images(num=num, val_ratio=0.15)
    generate_samples2(num=num, sample_ratio=0.1)
    generate_val_images2(num=num)
    count_images(dir=f'train{num}')
    count_images(dir=f'val{num}')