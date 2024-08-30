import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from FIN.models.encoder_decoder import FED
from Xception.xception import Network as Xception
import numpy as np
import torch
import os
import time
from shutil import copyfile
from network.noise_layers import *
from PIL import Image
import random, string
import os
from torchvision import transforms
import lpips
import kornia
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

'''
test
'''
criterion_LPIPS = lpips.LPIPS().to("cuda")

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


def get_path(path="temp/"):
    return path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png"

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)

def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
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

    seed_torch(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_log = "runs/test_log" + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()) + ".txt"
    writer = SummaryWriter('runs/' + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    message_length = 256
    save_images_number = 8
    strength_factor = 1
    fed = FED(False, message_length)
    fed.cuda()
    load(fed, '/home/likaide/sda4/wxs/pycharm/tmp/pycharm_project_wxs/Detecors/results/FIN_2024_03_03_11_43_24/models/fed_10.pt')
    xception = Xception("xception", device, lr=1e-5, betas=(0.9, 0.99))
    model_path = './Xception/xception_full_c23.pth'
    xception.load_model(model_path)
    xception.model.eval()

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    test_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/test/Real/*.png"),
                                os.path.join(dataset_path, "AdvMark/test/Fake/*/*.png1"), 256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart Testing : \n\n")

    test_result = {
        "error_rate": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "correct": 0.0
    }

    saved_iterations = np.random.choice(np.arange(1, len(test_dataloader)+1), size=save_images_number, replace=False)
    saved_all = None

    for step, (image, label) in enumerate(test_dataloader, 1):

        '''
		test
		'''
        fed.eval()

        with torch.no_grad():
            # use device to compute
            images = image.to(device)
            messages = torch.Tensor(np.random.choice([-0.5, 0.5], (image.shape[0], message_length))).to(device)
            labels = label.to(device)
            input_data = [images, messages]

            #################
            #    forward:   #
            #################

            encoded_images, left_noise = fed(input_data)
            encoded_images = images + (encoded_images - images) * strength_factor

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            ##################################################################
            for index in range(encoded_images.shape[0]):
                single_image = ((encoded_images[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0,255).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(single_image)
                file = get_path()
                while os.path.exists(file):
                    file = get_path()
                im.save(file)
                read = np.array(Image.open(file), dtype=np.uint8)
                #os.remove(file)

                encoded_images[index] = transform(read).unsqueeze(0).to(image.device)
            ##################################################################
            output = xception.model(encoded_images)
            # Cast to desired
            post_function = torch.nn.Softmax(dim=1)
            _, prediction = torch.max(post_function(output), 1)  # argmax
            prediction = torch.LongTensor([int(t.cpu().numpy()) for t in prediction])
            correct = float(torch.sum(prediction.cuda() == labels)) / len(images)
            ##################################################################
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean").item()

            # lpips
            lpips = torch.mean(criterion_LPIPS(encoded_images.detach(), images)).item()

            #noised_images_C, noised_images_R, noised_images_F = network.encoder_decoder.module.noise([encoded_images, images, masks])
            noised_images = eval('JpegTest()')([encoded_images.clone(), images, images])

            ##################################################################
            for index in range(noised_images.shape[0]):
                single_image = ((noised_images[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0,255).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(single_image)
                file = get_path()
                while os.path.exists(file):
                    file = get_path()
                im.save(file)
                read = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                noised_images[index] = transform(read).unsqueeze(0).to(image.device)
            ##################################################################

            #################
            #   backward:   #
            ################

            guass_noise = torch.zeros(left_noise.shape).cuda()
            output_data = [noised_images, guass_noise]
            re_img, decoded_messages = fed(output_data, rev=True)

        '''
		decoded message error rate
		'''
        error_rate = decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "correct": correct
        }

        for key in result:
            test_result[key] += float(result[key])


        if step in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, encoded_images, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        '''
		test results
		'''
        content = "Image " + str(step) + " : \n"
        for key in test_result:
            content += key + "=" + str(result[key]) + ","
            writer.add_scalar("Test/" + key, float(result[key]), step)
        content += "\n"

        with open(test_log, "a") as file:
            file.write(content)

        print(content)

    '''
	test results
	'''
    content = "Average : \n"
    for key in test_result:
        content += key + "=" + str(test_result[key] / step) + ","
        writer.add_scalar("Test_epoch/" + key, float(test_result[key] / step), 1)
    content += "\n"

    with open(test_log, "a") as file:
        file.write(content)

    print(content)
    #save_images(saved_all, "test", result_folder + "images/", resize_to=None)

    writer.close()


if __name__ == '__main__':
    main()
