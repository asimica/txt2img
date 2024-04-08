import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision import transforms
from random import randrange
# from t5 import t5_encode_text, get_encoded_dim

CAPTIONS_DIR = Path(__file__).parent / 'datasets' / 'birds' / 'text' / 'text'
IMAGES_DIR = Path(__file__).parent / 'datasets' / 'CUB_200_2011' / 'CUB_200_2011' / 'images'
LABELS_FILE = Path(__file__).parent / 'datasets' / 'CUB_200_2011' / 'CUB_200_2011' / 'images.txt'

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

		print(f"real_img_name = {real_img_name}")
		print(f"wrong_img_name = {wrong_img_name}")
		
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

if __name__ == "__main__":

	# dataset = TxtToImgDataset(IMAGES_DIR, LABELS_FILE, CAPTIONS_DIR)

	# image, captions = dataset[2]
	# # print(image)
	# print(captions)
	
	# image = np.transpose(image,(1,2,0))
	# # img = mpimg.imread(img_path2)
	# imgplot = plt.imshow(image)
	# plt.show()

	# image_size = 128
	# batch_size = 5
	# workers = 2
	# ngpu = 0
	# dataset = TxtToImgDataset(img_dir=IMAGES_DIR, labels_file=LABELS_FILE, captions_dir=CAPTIONS_DIR,
	# 							transform=transforms.Compose([
	# 								transforms.Resize(image_size),
	# 								transforms.CenterCrop(image_size),
	# 								# transforms.ToTensor(),
	# 	 							transforms.ConvertImageDtype(torch.float32),
	# 								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	# 							]))
	# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
	# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


	# from einops import rearrange
	# from transformers import T5Tokenizer, T5EncoderModel

	# for i, data in enumerate(dataloader, 0):
	# 	# image, captions = data
	# 	# print(captions)
	# 	image = data[0].to(device)
	# 	wrong_image = data[1].to(device)
	# 	captions = data[2]


		# print(f"image.shape = {image.shape}")
		# print(f"captions len = {len(captions)}")
		# print(f"captions = {captions[1]}") # (list, tuple), (10, batch_size)


		# handle = 't5-small'
		# tokenizer = T5Tokenizer.from_pretrained(handle)
		# model = T5EncoderModel.from_pretrained(handle)

		# # Move to cuda is available
		# if torch.cuda.is_available():
		# 	device = torch.device('cuda')
		# 	model = model.to(device)
		# else:
		# 	device = torch.device('cpu')	

		# # text = ["test text", "this is what?", "and?", "what about this one? ha??"]
		# text = ["test test", "test text once again"]

		# max_length = 256
		# tokenized = tokenizer.batch_encode_plus(
		# 	captions[1],
		# 	padding='longest',
		# 	max_length=max_length,
		# 	truncation=True,
		# 	return_tensors="pt",  # Returns torch.tensor instead of python integers
		# )	

		# print(f"tokenized = {tokenized}")

		# input_ids = tokenized.input_ids.to(device)
		# attention_mask = tokenized.attention_mask.to(device)

		# model.eval()

		# # Don't need gradient - T5 frozen during Imagen training
		# with torch.no_grad():
		# 	t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
		# 	final_encoding = t5_output.last_hidden_state.detach()

		# # Wherever the encoding is masked, make equal to zero
		# final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

		# final_encoding_mean = torch.mean(final_encoding, dim=1)

		# print(f"final_encoding_mean.shape={final_encoding_mean.shape}")

		# break	

	# encoding, attention_mask = t5_encode_text(text=["test this man", "another one", "and", "ANOTHER"])
	# print(f'encoding: {encoding.shape}')
	# print(f'encoding: {encoding.__class__}')
	# print(f'attention_mask: {attention_mask}')	


	# https://github.com/AssemblyAI-Examples/MinImagen/blob/main/minimagen/t5.py
	# print("jea")

	IMAGES_DIR = Path(__file__).parent / 'datasets' / 'CUB_200_2011' / 'CUB_200_2011' / 'images' / '004.Groove_billed_Ani'

	real_img_name = 'Groove_Billed_Ani_0002_1670.jpg'
	real_img_path = os.path.join(IMAGES_DIR, real_img_name)
	real_image = read_image(real_img_path) 

	if torch.Size([3, 360, 500]) == real_image.shape:
		real_image = real_image.repeat(3, 1, 1)
	print(f"real_image.shape = {real_image.shape}")