import h5py
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

import config
import data
from utils import utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_image_loader(image_dir):
    transform = transforms.Compose([
        transforms.Resize(int(config.IMAGE_SIZE / config.CENTRAL_FRACTION)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    dataset = data.AbstractImages(image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.PREPROCESS_BATCH_SIZE, 
        num_workers=config.DATA_WORKERS, 
        shuffle=False,
        pin_memory=True,
    )
    return dataloader


def preprocess_images(net, dataloader, split="train", device="cpu"):
    if split == "train":
        save_filepath = config.PREPROCESSED_TRAIN_FILEPATH
    elif split == "val":
        save_filepath = config.PREPROCESSED_VAL_FILEPATH
    else: 
        save_filepath = config.PREPROCESSED_TEST_FILEPATH

    features_shape = (
        len(dataloader.dataset), 
        config.OUTPUT_FEATURES, 
        config.OUTPUT_SIZE, 
        config.OUTPUT_SIZE, 
    )

    with h5py.File(save_filepath, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        image_ids = fd.create_dataset('ids', shape=(len(dataloader.dataset,)), dtype='int32')

        with torch.no_grad():
            i = 0
            j = 0
            for ids, imgs in tqdm(dataloader):
                #imgs = Variable(imgs.cuda(non_blocking=True))
                #imgs = Variable(imgs)
                imgs = imgs.to(device)
                out = net(imgs)

                j = i + imgs.size(0)
                features[i:j, :, :, :] = out.data.cpu().numpy().astype('float16')
                image_ids[i:j] = ids.numpy().astype('int32')
                i = j


def main():
    device = utils.set_device(mps=False, cuda=False)
    if device == "cuda":
        cudnn.benchmark = True
    net = Net().to(device)
    net.eval()

    trainloader = create_image_loader(config.IMAGES_TRAIN_DIR)
    valloader = create_image_loader(config.IMAGES_VAL_DIR)
    #testloader = create_image_loader(config.IMAGES_TEST_DIR)

    preprocess_images(net, trainloader, split="train", device=device)
    preprocess_images(net, valloader, split="val", device=device)
    #preprocess_images(net, testloader, split="test", device=device)


if __name__ == '__main__':
    main()
