import os
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
import numpy as np
import random
from MBRS.network.Network import *
#from MBRS.utils.load_train_setting import *


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True



def main():
    seed_torch(42)  # it doesnot work if the mode of F.interpolate is "bilinear"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_name = "MBRS"  # xception/resnet18/resnet50
    result_folder = "results/" + time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime()) + "/"
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    writer = SummaryWriter('runs/' + project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    noise_layers = ["Combined([JpegMask(50),Jpeg(50),Identity()])"]
    lr = 1e-3
    message_length = 256
    save_images_number = 8
    # Load model
    network = Network(256, 256, 256, noise_layers, device, 16, lr, False, False)
    EC_path = "MBRS/results/MBRS_256_m256/models/EC_42.pth"
    network.load_model_ed(EC_path)

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    train_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/train/Real/*.png"),
                               os.path.join(dataset_path, "AdvMark/train/Fake/*/*.png"), 256)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    train_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/val/Real/*.png"),
                               os.path.join(dataset_path, "AdvMark/val/Fake/*/*.png"), 256)
    val_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart training : \n\n")

    epoch_number = 10
    for epoch in range(1, epoch_number + 1):

        running_result = {
            "error_rate": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder": 0.0,
            "g_loss_on_detector": 0.0,
            "d_cover_loss": 0.0,
            "d_encoded_loss": 0.0
        }

        start_time = time.time()

        '''
        train
        '''
        for step, (image, label) in enumerate(train_dataloader, 1):
            print(device)
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
            label = label.to(device)

            result = network.train(image, message, label)
            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(train_dataloader)))

            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Train/" + key, float(result[key]), (epoch - 1) * len(train_dataloader) + step)
                running_result[key] += float(result[key])

        '''
        train results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in running_result:
            content += key + "=" + str(running_result[key] / step) + ","
            writer.add_scalar("Train_epoch/" + key, float(running_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/train_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        validation
        '''

        val_result = {
            "error_rate": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder": 0.0,
            "g_loss_on_detector": 0.0,
            "d_cover_loss": 0.0,
            "d_encoded_loss": 0.0
        }

        start_time = time.time()

        saved_iterations = np.random.choice(np.arange(1, len(val_dataloader)+1), size=save_images_number, replace=False)
        saved_all = None

        for step, (image, label) in enumerate(val_dataloader, 1):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
            label = label.to(device)

            result, (images, encoded_images, noised_images) = network.validation(image, message, label)
            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(val_dataloader)))
            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Val/" + key, float(result[key]), (epoch - 1) * len(val_dataloader) + step)
                val_result[key] += float(result[key])

            if step in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(image, encoded_images, noised_images)
                else:
                    saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        save_images(saved_all, epoch, result_folder + "images/", resize_to=None)

        '''
        validation results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in val_result:
            content += key + "=" + str(val_result[key] / step) + ","
            writer.add_scalar("Val_epoch/" + key, float(val_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/val_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        save model
        '''
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
        path_discriminator = path_model + "D_" + str(epoch) + ".pth"
        network.save_model(path_encoder_decoder, path_discriminator)

        writer.close()


if __name__ == '__main__':
    main()
