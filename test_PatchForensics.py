from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from patch_forensics.models.networks import networks
import time
import torch
import numpy as np
import random
import os

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
    net_D = networks.define_patch_D('xception_block5', 'xavier', [0])
    net_D = net_D.module
    load_path = 'patch_forensics/checkpoints/gp1-gan-winversion_seed0_xception_block5_constant_p10/bestval_net_D.pth'
    checkpoint = torch.load(load_path, map_location=str(device))
    state_dict = checkpoint['state_dict']
    net_D.load_state_dict(state_dict)
    net_D.eval()

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
        pred_logit = net_D(image)
        #softmax = torch.nn.Softmax(dim=1)

        n = pred_logit.shape[0]
        # vote_predictions probability is a tally of the patch votes
        votes = torch.argmax(pred_logit, dim=1).view(n, -1)
        vote_predictions = torch.mean(votes.float(), axis=1)
        vote_predictions = torch.stack([1 - vote_predictions,
                                        vote_predictions], axis=1)
        '''before_softmax_predictions = softmax(
            torch.mean(pred_logit, dim=(-1, -2)))
        after_softmax_predictions = torch.mean(
            softmax(pred_logit), dim=(-1, -2))
            
        patch_predictions = softmax(pred_logit)
        prediction_raw = patch_predictions.to('cpu').detach().numpy() # N2HW
        patch_preds = prediction_raw.transpose(0, 2, 3, 1) # NHW2
        n, h, w, c = patch_preds.shape
        patch_labels = np.tile(label, (1, h, w))
        patch_preds = patch_preds.reshape(-1, 2)
        patch_labels = patch_labels.reshape(-1)'''

        correct = float(torch.sum(torch.argmax(vote_predictions, axis=1) == 1 - label)) / len(image) # label here: real(1) fake(0)

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
