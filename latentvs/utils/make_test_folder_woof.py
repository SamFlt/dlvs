from pathlib import Path

if __name__ == '__main__':
    root = Path('/local/sfelton/generator_scenes/imagewoof2')
    val = root / 'val'
    test = root / 'test'
    test.mkdir(exist_ok=False)
    images_per_class = 20
    for val_class_folder in val.iterdir():
        if not val_class_folder.is_dir():
            continue
        test_class_folder = test / val_class_folder.name
        test_class_folder.mkdir(exist_ok=False)
        val_images = list(val_class_folder.iterdir())
        test_images = val_images[-images_per_class:]
        for test_image in test_images:
            test_image.rename(test_class_folder / test_image.name)
