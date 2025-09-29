import os
import json
import csv
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def save_features_to_csv(filename, features, csv_filename):
    # 如果 CSV 文件不存在，则创建一个新文件，并写入标题行
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Feature1', 'Feature2', ..., 'FeatureN'])  # 假设有 N 个特征

    # 打开 CSV 文件，追加特征值
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入每个样本的特征值
        for i in range(len(features)):
            row = [filename] + features[i].tolist()
            writer.writerow(row)


def predict_image(filename, img_path, model, class_indict, device, data_transform, csv_filename):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # Apply transformations
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output, features = model(img.to(device))
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        # 保存特征到 CSV 文件
        save_features_to_csv(filename, features, csv_filename)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    # plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights_mask+color/model-97.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Path to the folder containing images
    folder_path = "F:/hjy/project/MLP/data_color/all/"
    assert os.path.exists(folder_path), "folder: '{}' dose not exist.".format(folder_path)

    # CSV 文件路径
    csv_filename = 'color_features_all.csv'

    # Iterate through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            predict_image(filename, img_path, model, class_indict, device, data_transform, csv_filename)


if __name__ == '__main__':
    main()
