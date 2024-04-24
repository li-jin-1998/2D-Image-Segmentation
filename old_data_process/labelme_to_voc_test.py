import argparse
import glob
import os
import shutil

import labelme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input directory', default="/mnt/algo_storage_server/UNet/Data/Checked/")
    parser.add_argument('--output_dir', help='output directory', default="/mnt/algo_storage_server/UNet/Dataset")
    parser.add_argument('--labels', help='labels file', default="/home/lij/PycharmProjects/UNet/label.txt")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        print('Output directory already exists: ', args.output_dir)
        shutil.rmtree(args.output_dir)
        # sys.exit(1)

    os.makedirs(args.output_dir)

    os.makedirs(os.path.join(args.output_dir, 'image'))
    os.makedirs(os.path.join(args.output_dir, 'mask'))

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        # starts with -1
        class_id = i - 1
        class_name = line.strip()
        # check specific classes, skip ignore and reserve background
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_gum_'
        class_names.append(class_name)
        class_name_to_id[class_name] = class_id
    class_names = tuple(class_names)
    print(class_name_to_id, class_names)
    exit(0)
    out_class_names_file = os.path.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names in:', out_class_names_file)

    json_file_names = glob.glob(os.path.join(args.input_dir, '*.json'))
    image_names = []
    for file_name in json_file_names:
        print('Generating test sample from:', file_name)
        label_file = labelme.LabelFile(filename=file_name)
        base = os.path.splitext(os.path.basename(file_name))[0]
        image_names.append(base + '.png')
        out_img_file = os.path.join(args.output_dir, 'image', base + '.png')
        out_mask_file = os.path.join(args.output_dir, 'mask', base + '.png')
        with open(out_img_file, 'wb') as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        # print(img.shape," ",class_name_to_id)
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id)
        labelme.utils.lblsave(out_mask_file, lbl)

    image_number = len(image_names)
    image_list = range(image_number)
    out_test_list_file = os.path.join(args.output_dir, 'image_list.txt')
    with open(out_test_list_file, 'w') as f:
        for i in image_list:
            test_image_name = image_names[i] + '\n'
            f.write(test_image_name)


if __name__ == '__main__':
    main()
