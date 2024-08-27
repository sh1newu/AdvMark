from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from EfficientNet.predictor import Predictor
import time
import numpy as np
import random
import os
#from Xception.xception import *
import torch

'''
test
'''

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


def main():

    seed_torch(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_log = "runs/test_log" + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()) + ".txt"
    writer = SummaryWriter('runs/' + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    # Load model
    #model_path = "/home/likaide/sda4/wxs/pycharm/tmp/pycharm_project_wxs/Detecors/EfficientNet/5GAN1024png15000_xception.ckpt"
    #predictor = Predictor(model_path, device = device, arch = 'xception') #efficientnet-b3' or 'xception'

    model_path = "/home/likaide/sda4/wxs/pycharm/tmp/pycharm_project_wxs/Detecors/EfficientNet/2_efficient_EndEpoch.ckpt"
    predictor = Predictor(model_path, device = device, arch = 'efficientnet-b3') #efficientnet-b3' or 'xception'

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    test_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Real/*.png1"),
                               os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Fake/*.png"), 256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart Testing : \n\n")

    test_result = {
        "correct": 0.0
    }

    mean_arr = [0.485, 0.456, 0.406]
    stddev_arr = [0.229, 0.224, 0.225]

    for step, (image, label) in enumerate(test_dataloader, 1):
        image = image.to(device)
        label = label.to(device)

        # undo normalize
        image = image + 1  # now 0..2
        image = image * 0.5  # now 0..1

        # normalize image color channels
        for c in range(3):
            image[:, c, :, :] = (image[:, c, :, :].clone() - mean_arr[c]) / stddev_arr[c]

        '''
		test
		'''
        # Model prediction
        prediction = predictor.predict(image)

        correct = float(torch.sum(prediction.round().cuda() == label)) / len(image)

        result = {
            "correct": correct
        }

        for key in result:
            test_result[key] += float(result[key])

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

    writer.close()


if __name__ == '__main__':
    main()
