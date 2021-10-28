import os
from PIL import Image

IMAGE_EXTENSION = '.tif'
CROP_SIZE = 720
FILENAME_WIDTH = 3

ORIGINAL_IMAGE_PATH = 'OriginalImage'
ORIGINAL_MASK_PATH = 'OriginalMask'

CROPPED_IMAGE_PATH = 'Image'
CROPPED_MASK_PATH = 'Mask'


def get_image_files(path):
    files = sorted([x for x in os.listdir(path) if x.endswith(IMAGE_EXTENSION)])
    return files


def main():
    original_images = get_image_files(ORIGINAL_IMAGE_PATH)
    original_masks = get_image_files(ORIGINAL_MASK_PATH)

    assert len(original_images) == len(original_masks)

    counter = 0

    for image_file, mask_file in zip(original_images, original_masks):
        image = Image.open(os.path.join(ORIGINAL_IMAGE_PATH, image_file))
        mask = Image.open(os.path.join(ORIGINAL_MASK_PATH, mask_file))

        assert image.size == mask.size

        for rotation in range(4):
            width, height = image.size

            for upper_x in (0, width // 2):
                for upper_y in (0, height // 2):
                    lower_x = upper_x + CROP_SIZE
                    lower_y = upper_y + CROP_SIZE

                    new_image = image.crop((upper_x, upper_y, lower_x, lower_y))
                    new_mask = mask.crop((upper_x, upper_y, lower_x, lower_y))

                    filename = f'{counter:03}{IMAGE_EXTENSION}'

                    new_image.save(os.path.join(CROPPED_IMAGE_PATH, filename))
                    new_mask.save(os.path.join(CROPPED_MASK_PATH, filename))

                    counter += 1

            image = image.transpose(method=Image.ROTATE_90)
            mask = mask.transpose(method=Image.ROTATE_90)


if __name__ == '__main__':
    main()
