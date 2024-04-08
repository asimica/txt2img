import os
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel
import matplotlib.pyplot as plt
from random import randrange

class TxtToImgDataset(Dataset):
	def __init__(self, img_dir, labels_file, captions_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(labels_file, sep=' ')
		self.captions_dir = captions_dir
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		real_img_name = self.img_labels.iloc[idx, 1].replace(r'/', '\\')
		rand_idx = randrange(len(self.img_labels))
		while rand_idx == idx:
			rand_idx = randrange(len(self.img_labels))
		wrong_img_name = self.img_labels.iloc[rand_idx, 1].replace(r'/', '\\')
		
		real_img_path = os.path.join(self.img_dir, real_img_name)
		real_image = read_image(real_img_path) 

		wrong_img_path = os.path.join(self.img_dir, wrong_img_name)
		wrong_image = read_image(wrong_img_path) 

		captions_file_name = real_img_name.replace(r'.jpg', '.txt')
		captions_path = os.path.join(self.captions_dir, captions_file_name)
		with open(captions_path) as file:
			captions = [line.rstrip() for line in file]

		label = self.img_labels.iloc[idx, 0]
		if self.transform:
			real_image = self.transform(real_image)
		if self.transform:
			wrong_image = self.transform(wrong_image)
		if self.target_transform:
			label = self.target_transform(label)
		return real_image, wrong_image, captions

    
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
        self.seq1 = nn.Sequential(
            nn.Linear(txt_size, nt),
            nn.LeakyReLU(0.2, True)
        )
        self.seq2 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nt, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, z, txt_emb):
        # print(f"z.shape: {z.shape}")
        # print(f"txt_emb.shape: {txt_emb.shape}")
        h1 = self.seq1(txt_emb)
        # print(f"h1.shape: {h1.shape}")
        h2 = torch.cat((z.view(z.size(0), -1), h1.view(h1.size(0), -1)), dim=1)
        h2 = h2.view(h2.size(0), h2.size(1), 1, 1)
        # print(f"h2.shape: {h2.shape}")
        out = self.seq2(h2)
        # print(f"out.shape: {out.shape}")
        return out       
    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.seq1 = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(txt_size, nt),
            nn.LeakyReLU(0.2, True)
        )
        self.seq3 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8 + nt, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),            
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )
        
    def forward(self, img, txt_emb):
        # print(f"img.shape:{img.shape}")
        # print(f"txt_emb.shape:{txt_emb.shape}")
        h1 = self.seq1(img)
        # print(f"h1.shape:{h1.shape}")
        h2 = self.seq2(txt_emb)
        h2 = h2.unsqueeze(2)
        h2 = h2.unsqueeze(3)
        h2 = h2.repeat(1, 1, 8, 8)
        # print(f"h2.shape:{h2.shape}")
        h3 = torch.cat((h1, h2), dim=1)
        # print(f"h3.shape:{h3.shape}")
        out = self.seq3(h3)
        # print(f"out.shape:{out.shape}")
        return out


def t5_encode(text, max_text_length, tokenizer, model):
    tokenized = tokenizer.batch_encode_plus(text,
                                            padding='longest',
                                            max_length=max_text_length,
                                            truncation=True,
                                            return_tensors="pt")  # Returns torch.tensor instead of python integers	

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()
    # Don't need gradient - T5 frozen during Imagen training
    with torch.no_grad():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        encoding = t5_output.last_hidden_state.detach()
    # Wherever the encoding is masked, make equal to zero
    encoding = encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)
    encoding_mean = torch.mean(encoding, dim=1)    
    return encoding_mean


if __name__ == "__main__":
    CAPTIONS_DIR = Path(__file__).parent / 'datasets' / 'birds' / 'text' / 'text'
    LABELS_FILE = Path(__file__).parent / 'datasets' / 'CUB_200_2011' / 'CUB_200_2011' / 'images.txt'    
    ANNOTATIONS_PATH = Path(__file__).parent / 'datasets' / 'flickr30k_images' / 'results.csv'
    # IMAGES_DIR = Path(__file__).parent / 'datasets' / 'flickr30k_images' / 'flickr30k_images'
    IMAGES_DIR = Path(__file__).parent / 'datasets' / 'CUB_200_2011' / 'CUB_200_2011' / 'images'
    SAVE_DIR = Path(__file__).parent / 'out'

    # ____________________________________________________ dgan tutorial ____________________________________________________ 

    workers = 2             # Number of workers for dataloader
    batch_size = 64        # Batch size during training
    image_size = 128         # Spatial size of training images. All images will be resized to this size using a transform
    nc = 3                  # Number of channels in the training images. For color images this is 3
    nz = 100                # Size of z latent vector (i.e. size of generator input)
    ngf = 64                # Size of feature maps in generator
    ndf = 64                # Size of feature maps in discriminator
    num_epochs = 200         # Number of training epochs
    lr = 0.0002             # Learning rate for optimizers
    beta1 = 0.5             # Beta1 hyperparameter for Adam optimizers
    ngpu = 2                # Number of GPUs available. Use 0 for CPU mode.
    handle = 't5-small'
    txt_size = 512         # Number of dimensions for raw text 
    nt = 256                # Number of dimensions for text features 
    max_text_length = 256
    dataroot = IMAGES_DIR
    
    # dataset = dset.ImageFolder(root=dataroot,
    #                         transform=transforms.Compose([
    #                             transforms.Resize(image_size),
    #                             transforms.CenterCrop(image_size),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                         ]))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=workers)

    dataset = TxtToImgDataset(img_dir=IMAGES_DIR, labels_file=LABELS_FILE, captions_dir=CAPTIONS_DIR,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    # transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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

    # init text encoder
    tokenizer = T5Tokenizer.from_pretrained(handle)
    model = T5EncoderModel.from_pretrained(handle)
    if torch.cuda.is_available():
        model = model.to(device)    

    # loss function
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)  # used for visualization of training progress

    # fixed_encoding_mean = torch.randn(batch_size, txt_size, device=device)
    fixed_captions = ["-" for _ in range(batch_size)]
    fixed_captions[0] = "this yellow colored bird appears blue, but has undertones of red and green."
    fixed_captions[1] = "this is a black and brown eyering with white feet"
    fixed_captions[2] = "this bird has a rounded crown, a sharp bill, and a bright blue eye."
    fixed_captions[3] = "this bird is black with some iridescent green and blue to it"
    fixed_captions[4] = "this small bird has bright yellow eyes, a small head, a sharp black beak and black feet and all black feathers."
    fixed_captions[5] = "the bird is completely black except for his small yellow eyes."
    fixed_captions[6] = "this bird has a yellow eye ring and orange bill."
    fixed_captions[7] = "this particular bird has a belly that is all black and has red eye rings"
    fixed_captions[8] = "the blue bird has a blue eyering and a black bill."
    fixed_captions[9] = "this black bird has a pointy bill and brown colored eyes."
    fixed_captions = tuple(i for i in fixed_captions)
    fixed_encoding_mean = t5_encode(fixed_captions, max_text_length, tokenizer, model)

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

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            wrong_cpu = data[1].to(device)
            captions = data[2]
            for caption in captions:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                
                encoding_mean = t5_encode(caption, max_text_length, tokenizer, model)

                # txt_emb = torch.randn(b_size, txt_size, device=device)

                # (a) train with real

                # Forward pass real batch through D
                output = netD(real_cpu, encoding_mean).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # (b) train with wrong (newly added)

                label.fill_(fake_label)
                output = netD(wrong_cpu, encoding_mean).view(-1)
                errD_wrong = criterion(output, label)
                errD_wrong.backward()
                D_x_wrong = output.mean().item()

                # (c) train with fake

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise, encoding_mean)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach(), encoding_mean).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake + errD_wrong
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake, encoding_mean).view(-1)
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
                        fake = netG(fixed_noise, fixed_encoding_mean).detach().cpu()
                    grid = vutils.make_grid(fake, padding=2, normalize=True)
                    img_list.append(grid) 
                    for j in range(10):
                        img = fake.detach().numpy()[j]
                        for k in range(3):
                            min = np.min(img[k])
                            max = np.max(img[k])
                            img[k] = (img[k] - min) / (max - min)
                        img = np.transpose(img,(1,2,0))
                        plt.imsave(f"out/fake_img_{iters}_{j}.jpg", img)

                iters += 1
