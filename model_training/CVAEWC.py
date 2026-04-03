import torch
import torch.nn as nn


GRID_SIZE = 4
CONDITION_EMBED_DIM = 150


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_activation: bool = True):
        super().__init__()
        layers = [nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features)]
        if use_activation:
            layers.append(nn.Sigmoid())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConditionProjector(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            LinearBlock(input_features, 16 * GRID_SIZE * GRID_SIZE * GRID_SIZE),
            LinearBlock(16 * GRID_SIZE * GRID_SIZE * GRID_SIZE, 8 * GRID_SIZE * GRID_SIZE * GRID_SIZE),
            LinearBlock(8 * GRID_SIZE * GRID_SIZE * GRID_SIZE, CONDITION_EMBED_DIM),
        )

    def forward(self, x):
        return self.layers(x)


class Smoother(nn.Module):
    def __init__(self, condition_channels: int):
        super().__init__()
        flattened_features = 64 * GRID_SIZE * GRID_SIZE * GRID_SIZE
        hidden_features = 32 * GRID_SIZE * GRID_SIZE * GRID_SIZE
        self.encoder = ConvEncoderBlock(condition_channels, 64)
        self.head = nn.Sequential(
            LinearBlock(flattened_features, hidden_features),
            LinearBlock(hidden_features, CONDITION_EMBED_DIM),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


class Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, grid_size: int = GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        feature_dim = 64 * grid_size * grid_size * grid_size
        conv_feature_dim = 128 * grid_size * grid_size * grid_size
        latent_hidden_dim = 16 * grid_size * grid_size * grid_size

        self.conv_stack = nn.Sequential(
            ConvEncoderBlock(input_channels, 32),
            ConvEncoderBlock(32, 64),
            ConvEncoderBlock(64, 128),
        )
        self.feature = LinearBlock(conv_feature_dim, feature_dim)

        self.concentration_head = ConditionProjector(feature_dim)
        self.sro_head = ConditionProjector(feature_dim)

        merged_dim = feature_dim + 2 * CONDITION_EMBED_DIM
        self.mu_head = nn.Sequential(
            LinearBlock(merged_dim, latent_hidden_dim),
            nn.Linear(latent_hidden_dim, latent_dim),
        )
        self.logvar_head = nn.Sequential(
            LinearBlock(merged_dim, latent_hidden_dim),
            nn.Linear(latent_hidden_dim, latent_dim),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = self.feature(x)

        concentration = self.concentration_head(x)
        sro = self.sro_head(x)

        merged = torch.cat((x, concentration, sro), dim=1)
        mu = self.mu_head(merged)
        logvar = self.logvar_head(merged)
        return mu, logvar, concentration, sro


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition1_channels: int,
        condition2_channels: int,
        input_channels: int,
        grid_size: int = GRID_SIZE,
    ):
        super().__init__()
        del condition1_channels
        del condition2_channels

        self.input_channels = input_channels
        self.grid_size = grid_size

        hidden_dim = 128 * grid_size * grid_size * grid_size
        output_dim = input_channels * grid_size * grid_size * grid_size
        decoder_input_dim = latent_dim + 2 * CONDITION_EMBED_DIM

        self.layers = nn.Sequential(
            LinearBlock(decoder_input_dim, hidden_dim),
            LinearBlock(hidden_dim, hidden_dim),
            LinearBlock(hidden_dim, hidden_dim),
            LinearBlock(hidden_dim, hidden_dim),
            LinearBlock(hidden_dim, output_dim),
        )

    def forward(self, z, condition1, condition2):
        x = torch.cat((z, condition1, condition2), dim=1)
        x = self.layers(x)
        return x.view(-1, self.input_channels, self.grid_size, self.grid_size, self.grid_size)


class CVAE(nn.Module):
    def __init__(self, input_channels: int, condition1_channels: int, condition2_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, condition1_channels, condition2_channels, input_channels)
        self.smoother_c = Smoother(condition1_channels)
        self.smoother_w = Smoother(condition2_channels)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, w):
        mu, logvar, nconcen_p, nsro_p = self.encoder(x)
        nconcen_t = self.smoother_c(c)
        nsro_t = self.smoother_w(w)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, nconcen_p, nsro_p)
        return reconstruction, mu, logvar, nconcen_p, nconcen_t, nsro_p, nsro_t


if __name__ == "__main__":
    input_channels = 6
    condition1_channels = 3
    condition2_channels = 9
    latent_dim = 100
    batch_size = 2

    model = CVAE(input_channels, condition1_channels, condition2_channels, latent_dim)

    x = torch.randn(batch_size, input_channels, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    c = torch.randn(batch_size, condition1_channels, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    w = torch.randn(batch_size, condition2_channels, GRID_SIZE, GRID_SIZE, GRID_SIZE)

    reconstruction, mu, logvar, nconcen_p, nconcen_t, nsro_p, nsro_t = model(x, c, w)

    print("Reconstruction shape:", reconstruction.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("Concentration pred shape:", nconcen_p.shape)
    print("Concentration target shape:", nconcen_t.shape)
    print("SRO pred shape:", nsro_p.shape)
    print("SRO target shape:", nsro_t.shape)
