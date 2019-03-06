import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import models._DRAGAN as dcgm
import torch
import argparse

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--loadD', default=None, help='path to discriminator')
parser.add_argument('--loadG', default=None, help='path to discriminator')
opt = parser.parse_args()

out_dir = '../dataset/MNIST'
dataset = datasets.MNIST(root=out_dir, train=False, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]
                         ))

# root_dir = '../dataset/faces'
# dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]
# ))

nc = 1
ndf = 128
alpha = 1
ngpu = 1
p = 1
batch_size = 64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

dict_d = torch.load(opt.loadD, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
model = dcgm.DiscriminatorNetLessCheckerboardToCanonical(nc=nc, ndf=ndf, alpha=alpha, ngpu=ngpu)
if torch.__version__ == '0.4.0':
    del dict_d['net.1.bn2.num_batches_tracked']
    del dict_d['net.2.bn3.num_batches_tracked']
    del dict_d['net.3.bn4.num_batches_tracked']
    del dict_d['net.4.bn5.num_batches_tracked']
model.load_state_dict(dict_d)
model.to(gpu)

dict_g = torch.load(opt.loadG, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
generator = dcgm.GeneratorNetLessCheckerboardUpsample(nc=nc, ngf=128, ngpu=ngpu)
generator.load_state_dict(dict_g, strict=False)
generator.to(gpu)
torch.manual_seed(1234)

print('Before batch norm stabilization')
model.eval()
generator.eval()
for i in range(0, 20):
    noise = torch.randn(1, 100, 1, 1, device=gpu)
    images = generator(noise)
    images = F.pad(images, (p, p, p, p), mode='replicate')
    _, test_prob = model(images)
    print('Fake prob before: {}'.format(test_prob.item()))

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    batch_data = batch_data.to(gpu)
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
    _, test_prob = model(batch_data)
    print('Real prob before: {}'.format(test_prob.item()))
    if n_batch == 20:
        break

model.train()
print('Before batch norm stabilization train')
for i in range(0, 20):
    noise = torch.randn(1, 100, 1, 1, device=gpu)
    images = generator(noise)
    images = F.pad(images, (p, p, p, p), mode='replicate')
    test_prob = model(images)
    print('Fake prob before: {}'.format(test_prob.item()))

for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    batch_data = batch_data.to(gpu)
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
    test_prob = model(batch_data)
    print('Real prob before: {}'.format(test_prob.item()))
    if n_batch == 20:
        break

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

generator.train()
for n in range(0, 938):
    print("Stabilizing batch norm {}/{}".format(n, len(dataloader)))
    noise = torch.randn(64, 100, 1, 1, device=gpu)
    _ = generator(noise)


for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    print("Stabilizing batch norm {}/{}".format(n_batch, len(dataloader)))
    batch_data = batch_data.to(gpu)
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
    _ = model(batch_data)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

model.eval()
for n_batch, (batch_data, _) in enumerate(dataloader, 0):
    batch_data = batch_data.to(gpu)
    batch_data = F.pad(batch_data, (p, p, p, p), mode='replicate')
    _, test_prob = model(batch_data)
    print('Real prob: {}'.format(test_prob.item()))
    if n_batch == 19:
        break

generator.eval()
for i in range(0, 20):
    noise = torch.randn(1, 100, 1, 1, device=gpu)
    image = generator(noise)
    image = F.pad(image, (p, p, p, p), mode='replicate')
    _, test_prob_f = model(image)
    print('Fake prob: {}'.format(test_prob_f.item()))
