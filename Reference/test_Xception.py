from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time
import numpy as np
import random
import os
from Xception.xception import *

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
    project_name = "xception" # xception/resnet18/resnet50
    network = Network(project_name, device, lr=1e-5, betas=(0.9, 0.99))
    model_path = './Xception/xception_full_c23.pth'
    #model_path = "/home/likaide/sda4/wxs/pycharm/tmp/pycharm_project_wxs/Detecors/results/xception_2023_10_04_21_25_03/models/xception_face_epoch18.pth"
    network.load_model(model_path)

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
        network.model.eval()
        output = network.model(image)
        post_function = torch.nn.Softmax(dim=1)
        output = post_function(output)

        # Cast to desired
        _, prediction = torch.max(output, 1)  # argmax
        prediction = torch.LongTensor([int(t.cpu().numpy()) for t in prediction])

        correct = float(torch.sum(prediction.cuda() == label)) / len(image)

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
