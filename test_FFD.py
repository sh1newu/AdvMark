from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from FFD.dataset import Dataset
from FFD.templates import get_templates
from FFD.xception import Model
import time
import torch
import numpy as np
import random
import os
import torch.nn.functional as F

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
    '''model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model_path = './Xception/xception_face_c40.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)'''
    BACKBONE = 'xcp'
    MAPTYPE = 'tmp'
    MODEL_NAME = '{0}_{1}'.format(BACKBONE, MAPTYPE)
    MODEL_DIR = 'FFD/models/' + MODEL_NAME + '/'
    TEMPLATES = get_templates()
    MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
    MODEL.load(75, MODEL_DIR)
    MODEL.model.cuda()

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    test_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Real/*.png"),
                               os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Fake/*.png1"), 256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart Testing : \n\n")

    test_result = {
        "correct": 0.0
    }

    for step, (image, label) in enumerate(test_dataloader, 1):
        image = image.to(device)
        label = label.to(device)
        image = F.interpolate(image, size=(299, 299), mode='bilinear')

        '''
		test
		'''
        # Model prediction
        MODEL.model.eval()
        x, _, _ = MODEL.model(image)
        pred = torch.max(x, dim=1)[1]

        correct = float(torch.sum(pred == label)) / len(image)

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
