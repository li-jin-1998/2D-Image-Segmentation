import argparse
import glob
import os
import shutil

import labelme
from sklearn.model_selection import KFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input directory', default="/mnt/algo_storage_server/UNet/Data/Checked/")
    parser.add_argument('--output_dir', help='output directory', default="/home/lij/PycharmProjects/Seg/data/")
    parser.add_argument('--labels', help='labels file', default="/home/lij/PycharmProjects/UNet/label.txt")
    parser.add_argument('--noviz', help='no visualization', action='store_true', default=True)
    args = parser.parse_args()

    test_output_dir = os.path.join(args.output_dir, 'test')
    train_output_dir = os.path.join(args.output_dir, 'train')

    if os.path.exists(args.output_dir):
        print('Output directory already exists: ', args.output_dir)
        shutil.rmtree(test_output_dir)
        shutil.rmtree(train_output_dir)
        # sys.exit(1)

    # os.makedirs(args.output_dir)
    os.makedirs(test_output_dir)
    os.makedirs(train_output_dir)
    # augmentation_output_dir = os.path.join(args.output_dir, 'augmentation')
    os.makedirs(os.path.join(test_output_dir, 'image'))
    os.makedirs(os.path.join(test_output_dir, 'mask'))
    os.makedirs(os.path.join(train_output_dir, 'image'))
    os.makedirs(os.path.join(train_output_dir, 'mask'))
    # os.makedirs(os.path.join(augmentation_output_dir, 'image', 'data'))
    # os.makedirs(os.path.join(augmentation_output_dir, 'mask', 'data'))
    # if not args.noviz:
    #     os.makedirs(os.path.join(test_output_dir, 'visualization'))
    #     os.makedirs(os.path.join(train_output_dir, 'visualization'))
    print('Creating test and train samples under:', args.output_dir)

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
    out_class_names_file = os.path.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names in:', out_class_names_file)

    json_file_names = glob.glob(os.path.join(args.input_dir, '*.json'))
    json_file_number = len(json_file_names)
    json_file_residue = json_file_number % 11
    print('json_file_residue is', json_file_residue)
    fold = KFold(n_splits=11, random_state=42, shuffle=True)
    test_image_names = []
    train_image_names = []
    for train_index, test_index in fold.split(
        json_file_names[:json_file_number - json_file_residue]):
        for i in test_index:
            test_file_name = json_file_names[i - 1]
            print('Generating test sample from:', test_file_name)
            label_file = labelme.LabelFile(filename=test_file_name)
            base = os.path.splitext(os.path.basename(test_file_name))[0]
            test_image_names.append(base + '.png')
            out_img_file = os.path.join(test_output_dir, 'image', base + '.png')
            out_mask_file = os.path.join(test_output_dir, 'mask', base + '.png')
            # if not args.noviz:
            #     out_viz_file = os.path.join(test_output_dir, 'visualization',
            #                                 base + '.png')
            with open(out_img_file, 'wb') as f:
                f.write(label_file.imageData)
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            print(img.shape, " ", class_name_to_id)
            lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id)
            labelme.utils.lblsave(out_mask_file, lbl)
            # if not args.noviz:
            #     viz = imgviz.label2rgb(lbl, imgviz.rgb2gray(img), font_size=15,
            #                            label_names=class_names, loc='rb')
            #     imgviz.io.imsave(out_viz_file, viz)

        for i in train_index:
            train_file_name = json_file_names[i - 1]
            print('Generating train sample from:', train_file_name)
            label_file = labelme.LabelFile(filename=train_file_name)
            base = os.path.splitext(os.path.basename(train_file_name))[0]
            train_image_names.append(base + '.png')
            out_img_file = os.path.join(train_output_dir, 'image',
                                        base + '.png')
            out_mask_file = os.path.join(train_output_dir, 'mask',
                                         base + '.png')
            # if not args.noviz:
            #     out_viz_file = os.path.join(train_output_dir, 'visualization',
            #                                 base + '.png')
            with open(out_img_file, 'wb') as f:
                f.write(label_file.imageData)
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id)
            labelme.utils.lblsave(out_mask_file, lbl)
            # if not args.noviz:
            #     viz = imgviz.label2rgb(lbl, imgviz.rgb2gray(img), font_size=15,
            #                            label_names=class_names, loc='rb')
            #     imgviz.io.imsave(out_viz_file, viz)
        break

    residue_index = range(json_file_number - json_file_residue,
                          json_file_number)
    for i in residue_index:
        test_file_name = json_file_names[i]
        print('Generating residue sample from:', test_file_name)
        label_file = labelme.LabelFile(filename=test_file_name)
        base = os.path.splitext(os.path.basename(test_file_name))[0]
        test_image_names.append(base + '.png')
        out_img_file = os.path.join(test_output_dir, 'image', base + '.png')
        out_mask_file = os.path.join(test_output_dir, 'mask', base + '.png')
        # if not args.noviz:
        #     out_viz_file = os.path.join(test_output_dir, 'visualization',
        #                                 base + '.png')
        with open(out_img_file, 'wb') as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id)
        labelme.utils.lblsave(out_mask_file, lbl)
        # if not args.noviz:
        #     viz = imgviz.label2rgb(lbl, imgviz.rgb2gray(img), font_size=15,
        #                            label_names=class_names, loc='rb')
        #     imgviz.io.imsave(out_viz_file, viz)

    test_image_number = len(test_image_names)
    test_image_list = range(test_image_number)
    out_test_list_file = os.path.join(test_output_dir, 'image_list.txt')
    with open(out_test_list_file, 'w') as f:
        for i in test_image_list:
            test_image_name = test_image_names[i] + '\n'
            f.write(test_image_name)

    train_image_number = len(train_image_names)
    train_image_list = range(train_image_number)
    out_train_list_file = os.path.join(train_output_dir, 'image_list.txt')
    with open(out_train_list_file, 'w') as f:
        for i in train_image_list:
            train_image_name = train_image_names[i] + '\n'
            f.write(train_image_name)


if __name__ == '__main__':
    main()
