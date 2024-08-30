from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time
import numpy as np
import random
import os
import torch
from multiple_attention.models.MAT import MAT

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
    net_config = dict(net='efficientnet-b4', feature_layer='b2', attention_layer='b5',
                      num_classes=2, M=4,
                      mid_dims=256, dropout_rate=0.25, drop_final_rate=0.5,
                      pretrained='', alpha=0.05, margin=0.5,
                      inner_margin=[0.1, -2])
    net = MAT(**net_config)
    path = torch.load('multiple_attention/pretrained/ff_c23.pth')
    path = path['state_dict']
    net.load_state_dict(path, strict=False)
    net.cuda()

    dataset_path = '/home/likaide/sda5/wxs/Dataset'
    test_dataset = ImgDataset2(os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Real/*.png1"),
                               os.path.join(dataset_path, "AdvMark/MBRS-AdvMark-Ensemble3/Fake/*.png"), 256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart Testing : \n\n")

    test_result = {
        "correct": 0.0
    }

    for step, (image, label) in enumerate(test_dataloader, 1):
        image = image.to(device)
        label = label.to(device)

        '''
		test
		'''
        # Model prediction
        net.eval()
        logits = net(image)
        pred = torch.nn.functional.softmax(logits, dim=1)[:, 1]

        correct = float(torch.sum(pred.round() == label)) / len(image)

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
