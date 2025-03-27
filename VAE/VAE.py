import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 128
learning_rate = 1e-3
latent_dim = 20
num_epochs = 20
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, latent_dimx  ):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mean = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        reconstruction = torch.sigmoid(self.fc4(x))
        return reconstruction
    
def reparameterize(mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar
    
def vae_loss(reconstruction, x, mean, logvar):
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    print("reconstruction_loss:", reconstruction_loss)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    print("kl_divergence:", kl_divergence)
    return reconstruction_loss + kl_divergence

vae = VAE(latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    vae.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)  # Flatten the images
        optimizer.zero_grad()

        reconstruction, mean, logvar = vae(data)
        loss = vae_loss(reconstruction, data, mean, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader.dataset):.4f}")

def visualize_reconstructions(model, test_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.view(data.size(0), -1).to(device)
        reconstructions, _, _ = model(data)

        fig, axes = plt.subplots(2, num_images, figsize=(10, 2))
        for i in range(num_images):
            axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructions[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
        plt.show()

test_loader = DataLoader(dataset=datasets.MNIST(root='./data', train=False, transform=transform), batch_size=10, shuffle=True)
visualize_reconstructions(vae, test_loader)