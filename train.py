import config
from dataset.dataset import Dataset
import utils.initialize as initialize
from models.generator import Generator
from models.discriminator import Discriminator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


if __name__ == '__main__':
    hp = config.Hyperparameter
    const = config.Constant

    initialize.random_seed(hp.SEED)
    initialize.delete_images()
    initialize.create_image_folders()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize.weights_init)
    discriminator.apply(initialize.weights_init)

    dataset = Dataset()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader

    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()

    optimizer_g = optim.Adam(generator.parameters(), hp.LR, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), hp.LR, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    total_loss_G = 0
    total_loss_D = 0
    avg_loss_G = 0
    avg_loss_D = 0

    for epoch in range(hp.EPOCH):
        print('---------------------------')
        print('Epoch:', epoch, end='\n\n')
        for i, (data, label) in enumerate(train_loader):
            batch_size = label.shape[0] if label.shape[0] < hp.BATCH_SIZE else hp.BATCH_SIZE

            data, label = data.to(device), label.to(device)
            noise = torch.randn((batch_size, hp.NUM_NOISE))
            class_onehot = torch.zeros((batch_size, hp.NUM_CLASS))
            class_onehot[np.arange(batch_size), label] = 1

            input = torch.cat((noise, class_onehot), 1).to(device)
            fake = generator(input)
            real_dis_label = torch.ones((batch_size, 1)).to(device)
            fake_dis_label = torch.zeros((batch_size, 1)).to(device)
            aux_label = label.clone()

            # train discriminator with real
            discriminator.zero_grad()
            dis_output, aux_output = discriminator(data)
            dis_errD_real = dis_criterion(dis_output, real_dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = (dis_errD_real + aux_errD_real) / 2
            errD_real.backward()

            # train discriminator with fake
            dis_output, aux_output = discriminator(fake.detach())
            dis_errD_fake = dis_criterion(dis_output, fake_dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = (dis_errD_fake + aux_errD_fake) / 2
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizer_d.step()

            # train generator
            generator.zero_grad()
            dis_output, aux_output = discriminator(fake)
            dis_errG = dis_criterion(dis_output, real_dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = (dis_errG + aux_errG) / 2
            errG.backward()
            optimizer_g.step()

            # compute loss
            cur_iter = epoch * len(train_loader) + i
            total_loss_G += errG.item()
            total_loss_D += errD.item()
            avg_loss_G = total_loss_G / (cur_iter + 1)
            avg_loss_D = total_loss_D / (cur_iter + 1)

            if i % 100 == 0:
                print('Loss G:', avg_loss_G)
                print('Loss D:', avg_loss_D)

        print('* * * * * * * * * * * * * *')
        print('Loss G:', avg_loss_G)
        print('Loss D:', avg_loss_D)
        print('---------------------------')

        # save image
        noise = torch.randn((1, hp.NUM_NOISE))
        for i in range(hp.NUM_CLASS):
            class_onehot = torch.zeros((1, hp.NUM_CLASS))
            class_onehot[np.arange(1), i] = 1
            input = torch.cat((noise, class_onehot), 1).to(device)
            img = generator(input).squeeze()
            img = (img + 1) / 2

            img = img.cpu().detach().numpy() * 255
            im = Image.fromarray(img)
            im = im.convert('L')
            im.save(const.IMAGE_FOLDER + str(i) + '/' + str(epoch) + '.png')
