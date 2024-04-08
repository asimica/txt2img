import torch
import torch.optim as optim
# from nn import draw_conv_filters
from pathlib import Path
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from t5 import t5_encode_text, get_encoded_dim
# from utils import draw_conv_filters, plot_training_progress
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MyModel(nn.Module):
    def __init__(self, conv1_in_channels=1, conv1_out_channels=16, conv1_kernel_size=(5, 5), pool1_kernel_size=(2,2),
                  conv2_out_chanels=32, pool2_kernel_size=(2,2), fc1_in_features= 32 * 49, fc1_out_features=512, class_count=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_in_channels, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, padding='same', bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=pool1_kernel_size)   
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_chanels, kernel_size=conv1_kernel_size, padding='same', bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool2_kernel_size)     
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=fc1_out_features, bias=True)
        self.fc2 = nn.Linear(in_features=fc1_out_features, out_features=class_count, bias=True)

    def forward(self, x):   
        h = self.conv1(x)
        h = self.maxpool1(h)
        h = torch.relu(h)
        h = self.conv2(h)
        h = self.maxpool2(h)
        h = torch.relu(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc2(h)
        return logits
    

def train(model, ds_train, ds_valid, config):
    save_dir = config['save_dir']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']

    train_dl = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
    valid_dl = DataLoader(ds_valid, batch_size=batch_size)

    train_steps = len(train_dl.dataset) // batch_size
    valid_steps = len(valid_dl.dataset) // batch_size

    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossFn = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=opt, step_size=2*train_steps, gamma=1e-1)

    stats = {
	"train_loss": [],
	"train_acc": [],
	"valid_loss": [],
	"valid_acc": [],
    "lr":[]
    }
    for e in range(0, max_epochs):
        model.train()
        total_train_loss, total_valid_loss = 0,0
        train_correct, valid_correct = 0,0
        for i, (x, y) in enumerate(train_dl):
            logits = model(x)
            loss = lossFn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            total_train_loss += loss
            train_correct += (logits.argmax(1) == y.argmax(1)).sum()
            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (e, i*batch_size, len(train_dl.dataset), loss.detach().numpy()))
            if i % 100 == 0:
                # draw_conv_filters(e, i*batch_size, model.conv1.weight.detach().numpy(), save_dir)
                pass
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (train_correct / ((i+1)*batch_size) * 100))
        with torch.no_grad():
            model.eval()
            for (x, y) in valid_dl:
                logits = model(x)
                total_valid_loss += lossFn(logits, y)
                valid_correct += (logits.argmax(1) == y.argmax(1)).sum()
        avg_train_loss = total_train_loss / train_steps
        avg_valid_loss = total_valid_loss / valid_steps
        train_acc = train_correct / len(train_dl.dataset) * 100
        valid_acc = valid_correct / len(valid_dl.dataset) * 100
        stats["train_loss"].append(avg_train_loss.detach().numpy())
        stats["train_acc"].append(train_acc)
        stats["valid_loss"].append(avg_valid_loss.detach().numpy())
        stats["valid_acc"].append(valid_acc)    
        stats["lr"] += [lr_scheduler.get_lr()]
        print("Train accuracy: %.2f" % train_acc)
        print("Validation accuracy: %.2f" % valid_acc)
    # plot_training_progress(SAVE_DIR, stats)

def evaluate(model, ds_test, config):
    print("\nRunning test evaluation")
    batch_size = config['batch_size']
    test_dl = DataLoader(ds_test, batch_size=batch_size)
    lossFn = nn.CrossEntropyLoss()
    test_steps = len(test_dl.dataset) // batch_size
    cnt_correct, loss = 0, 0
    for (x, y) in test_dl:
        logits = model(x)
        cnt_correct += (logits.argmax(1) == y.argmax(1)).sum()
        loss += lossFn(logits, y)
    acc = cnt_correct / len(test_dl.dataset) * 100
    avg_loss = loss / test_steps
    print("accuracy = %.2f" % acc)
    print("avg loss = %.2f\n" % avg_loss)

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep='|')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# ___________________________________________ dgan tutorial ___________________________________________
    
# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

if __name__ == "__main__":
    ANNOTATIONS_PATH = Path(__file__).parent / 'datasets' / 'flickr30k_images' / 'results.csv'
    IMAGES_DIR = Path(__file__).parent / 'datasets' / 'flickr30k_images' / 'flickr30k_images'
    SAVE_DIR = Path(__file__).parent / 'out'
	# config = {}
	# config['max_epochs'] = 8
	# config['batch_size'] = 50
	# config['save_dir'] = SAVE_DIR
	# config['weight_decay'] = 1e-3
	# config['lr'] = 1e-1

	# imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
	# data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True)

	# transform = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	transforms.Normalize((0.1307,), (0.3081,))
	# 	])
	# target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
	# ds_train_valid = MNIST(DATA_DIR, train=True, download=True, transform=transform, target_transform= target_transform)
	# ds_test = MNIST(DATA_DIR, train=False, download=True, transform=transform, target_transform= target_transform)

	# ds_train = Subset(ds_train_valid, range(55000))
	# ds_valid = Subset(ds_train_valid, range(55000, len(ds_train_valid)))

	# ptconv = PTConv1()
	# train(ptconv, ds_train, ds_valid, config)
	# evaluate(ptconv, ds_test, config)

    # dataset = CustomImageDataset(ANNOTATIONS_PATH, IMAGES_DIR)
    # img, label = dataset[1]
    # img = np.transpose(img.detach().numpy(), (1, 2, 0))
    # plt.imshow(img)
    # plt.show()
    # print(label)
    # print(img.shape)

    # encoding, attention_mask = t5_encode_text(text=["testa", "another one", "and", "ANOTHER"])
    # print(f'encoding: {encoding.shape}')
    # print(f'encoding: {encoding.__class__}')
    # print(f'attention_mask: {attention_mask}')

    # ____________________________________________________ dgan tutorial ____________________________________________________ 

    
    workers = 2             # Number of workers for dataloader
    batch_size = 128        # Batch size during training
    image_size = 64         # Spatial size of training images. All images will be resized to this size using a transform
    nc = 3                  # Number of channels in the training images. For color images this is 3
    nz = 100                # Size of z latent vector (i.e. size of generator input)
    ngf = 64                # Size of feature maps in generator
    ndf = 64                # Size of feature maps in discriminator
    num_epochs = 10         # Number of training epochs
    lr = 0.0002             # Learning rate for optimizers
    beta1 = 0.5             # Beta1 hyperparameter for Adam optimizers
    ngpu = 2                # Number of GPUs available. Use 0 for CPU mode.
    dataroot = IMAGES_DIR
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=workers)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # real_batch = next(iter(dataloader))
    # print(f'real_batch {real_batch.__class__}')
    # print(f'real batch shape {real_batch[0][0].detach().numpy().shape}')
    # plt.imshow(np.transpose(real_batch[0][11], (1,2,0)))
    # plt.show()

    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    # init generator
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    print(netG)

    # init discriminator
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    print(netD)

    # loss function
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # used for visualization of training progress
    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...") 
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(grid) 
                for j in range(10):
                    img = fake.detach().numpy()[j]
                    for k in range(3):
                        min = np.min(img[k])
                        max = np.max(img[k])
                        img[k] = (img[k] - min) / (max - min)
                    img = np.transpose(img,(1,2,0))
                    plt.imsave(f"workspace/out/fake_img_{iters}_{j}.jpg", img)

            iters += 1

        
    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())