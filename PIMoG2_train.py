import os
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
import numpy as np
import random
import torch
from PIMoG.model import Discriminator
from PIMoG.model import Encoder_Decoder
from torch.autograd import Variable
import lpips
import kornia
from Xception.xception import Network as Xception


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

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate

def main():
    seed_torch(42)  # it doesnot work if the mode of F.interpolate is "bilinear"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_name = "PIMoG2"  # xception/resnet18/resnet50
    result_folder = "results/" + time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime()) + "/"
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    writer = SummaryWriter('runs/' + project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    message_length = 128
    save_images_number = 8
    distortion = 'ScreenShooting'
    num_channels = 64
    net = Encoder_Decoder(distortion)
    net_Discriminator = Discriminator(num_channels)
    net_Discriminator.to(device)
    optimizer_Discriminator = torch.optim.Adam(net_Discriminator.parameters())
    net_optimizer = torch.optim.Adam(net.parameters())
    net.to(device)

    xception = Xception("xception", device, lr=1e-5, betas=(0.9, 0.99))
    model_path = './Xception/xception_full_c23.pth'
    xception.load_model(model_path)
    xception.model.eval()
    for p in xception.model.parameters():
        p.requires_grad = False

    # loss function
    criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_MSE = torch.nn.MSELoss().to(device)
    criterion_LPIPS = lpips.LPIPS().to(device)
    criterion_CE = torch.nn.CrossEntropyLoss().to(device)

    load(net, '/home/likaide/sda4/wxs/pycharm/tmp/pycharm_project_wxs/Detecors/results/PIMoG1_2024_03_19_18_12_03/models/PIMoG_69.pt')

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    train_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/train/Real/*.png"),
                               os.path.join(dataset_path, "AdvMark/train/Fake/*/*.png"), 256)
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True)

    train_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/val/Real/*.png"),
                               os.path.join(dataset_path, "AdvMark/val/Fake/*/*.png"), 256)
    val_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart training : \n\n")

    epoch_number = 10
    for epoch in range(1, epoch_number + 1):

        running_result = {
            "error_rate": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss": 0.0,
            "g_loss_on_encoder": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder": 0.0,
            "g_loss_on_detector": 0.0
        }

        start_time = time.time()

        '''
        train
        '''
        net.train()
        for step, (image, label) in enumerate(train_dataloader, 1):
            print(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length)))
            image, message = Variable(image), Variable(message)
            images = image.to(device)
            messages = message.to(device)
            labels = label.to(device)
            images.requires_grad = True

            #################
            #    forward:   #
            #################
            encoded_images, noised_images, decoded_messages = net(images, messages)
            loss_de = criterion_MSE(decoded_messages, messages)
            inputgrad = torch.autograd.grad(loss_de, images, create_graph=True)[0]
            mask = torch.zeros(inputgrad.shape).to(device)
            for ii in range(inputgrad.shape[0]):
                a = inputgrad[ii, :, :, :]
                a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                mask[ii, :, :, :] = a.detach()
            d_label_host = torch.full((images.shape[0], 1), 1, dtype=torch.float, device=device)
            d_label_encoded = torch.full((images.shape[0], 1), 0, dtype=torch.float, device=device)
            g_label_encoded = torch.full((images.shape[0], 1), 1, dtype=torch.float, device=device)

            # train the discriminator
            optimizer_Discriminator.zero_grad()
            d_image = net_Discriminator(images.detach())
            d_decoded = net_Discriminator(encoded_images.detach())
            d_loss = criterion_BCE(d_image, d_label_host) + criterion_BCE(d_decoded, d_label_encoded)
            d_loss.backward()

            optimizer_Discriminator.step()

            # train the Encoder_Decoder
            g_encoded = net_Discriminator(encoded_images)
            g_loss_on_discriminator = criterion_BCE(g_encoded, g_label_encoded)

            g_loss_on_decoder = criterion_MSE(decoded_messages, messages)
            g_loss_on_encoder = criterion_MSE(encoded_images * mask.float(), images * mask.float()) * 0.5 + criterion_MSE(
                encoded_images, images) * 2
            g_loss_on_encoder_LPIPS = torch.mean(criterion_LPIPS(encoded_images, images))
            g_loss_on_detector = criterion_CE(xception.model(encoded_images), labels)
            g_loss = 1 * g_loss_on_encoder + 3 * g_loss_on_decoder + 0.001 * g_loss_on_discriminator + 0.1 * g_loss_on_detector
            net_optimizer.zero_grad()
            g_loss.backward()

            net_optimizer.step()

            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

            error_rate = decoded_message_error_rate_batch(messages, decoded_messages)

            result = {
                "error_rate": error_rate,
                "psnr": psnr,
                "ssim": ssim,
                "g_loss": g_loss,
                "g_loss_on_encoder": g_loss_on_encoder,
                "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
                "g_loss_on_decoder": g_loss_on_decoder,
                "g_loss_on_detector": g_loss_on_detector
            }

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
            "g_loss_on_encoder": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder": 0.0,
            "g_loss_on_detector": 0.0
        }

        start_time = time.time()

        saved_iterations = np.random.choice(np.arange(1, len(val_dataloader)+1), size=save_images_number, replace=False)
        saved_all = None

        with torch.no_grad():
            net.eval()
            for step, (image, label) in enumerate(val_dataloader, 1):
                message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length)))
                image, message = Variable(image), Variable(message)
                images = image.to(device)
                messages = message.to(device)
                labels = label.to(device)

                #################
                #    forward:   #
                #################
                encoded_images, noised_images, decoded_messages = net(images, messages)

                g_loss_on_decoder = criterion_MSE(decoded_messages, messages)
                g_loss_on_encoder = criterion_MSE(encoded_images, images) * 2
                g_loss_on_encoder_LPIPS = torch.mean(criterion_LPIPS(encoded_images, images))
                g_loss_on_detector = criterion_CE(xception.model(encoded_images), labels)
                g_loss = 1 * g_loss_on_encoder_LPIPS + 3 * g_loss_on_decoder + 0.1 * g_loss_on_detector

                # psnr
                psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

                # ssim
                ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

                error_rate = decoded_message_error_rate_batch(messages, decoded_messages)

                result = {
                    "error_rate": error_rate,
                    "psnr": psnr,
                    "ssim": ssim,
                    "g_loss": g_loss,
                    "g_loss_on_encoder": g_loss_on_encoder,
                    "g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
                    "g_loss_on_decoder": g_loss_on_decoder,
                    "g_loss_on_detector": g_loss_on_detector
                }

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
        torch.save({'opt': net_optimizer.state_dict(),
                    'net': net.state_dict()},
                    path_model + 'PIMoG_' + str(epoch) + '.pt')

        writer.close()


if __name__ == '__main__':
    main()
