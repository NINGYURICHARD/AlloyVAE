from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import itertools
import importlib
import math
import os

N = 1000
iterations = 1000
C = np.zeros((N, iterations, 2, 4, 4, 4))
S = np.zeros((N, 6, 4, 4, 4))
V_TAU = np.zeros((N, 1))
ERRO_C = np.zeros((N,1))
ERRO_W = np.zeros((N,1))
learning_rate = 1e-4
Gamma = 50
alpha = 0.1
Reso=4
# load your check point here
# nu_try='1209CVAEWC_3'

Ratio = 0.9
latent_dim = 300
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
MODEL = importlib.import_module(f'checkpoint.{nu_try}.CVAEWC')
model = MODEL.CVAE(6,2,9,latent_dim)
model.eval()
checkpoint = torch.load(f'/home/nyanac/stress_predt2/VAE/checkpoint/{nu_try}/{nu_try}.tar', map_location=DEVICE) # substitute with your path
model.load_state_dict(checkpoint["state_dict"])
model.to(DEVICE)
model.eval()
new_folder_path = f'/home/nyanac/stress_predt2/VAE/checkpoint/{nu_try}/optimize_c/1000_0211_fixednoise'  # substitute with your path
os.mkdir(new_folder_path)
def cyclic_beta_scheduler(epoch):
    if (epoch // 20) % 2 == 0:
        return Gamma
    else:
        return Gamma
for i in range(N):
    print("cycle=", i)
    torch.manual_seed(0) 
    model.eval()

    c = torch.normal(0.5, (0.0252) ** 0.5, size=(1, 2, 4, 4, 4))
    c = torch.clamp(c, 0, 1)
    c = c.to(DEVICE)
    c.requires_grad_(True)
    # w = torch.Tensor(conditions_test[i_predt, 3:, :, :, :]).unsqueeze(0)
    w = torch.ones((1, 9, 4, 4, 4))*0.5
    w = w.to(DEVICE)
    torch.manual_seed(i) 
    noise = torch.randn(1, latent_dim)
    noise = noise.to(DEVICE)
    noise.requires_grad_(True)

    c_values = []
    s_values = []
    selfcheck_c = []
    selfcheck_w = []
    noise_values = []
    Erro_c = []
    Erro_w = []
    VAR = []
    Total_loss = 1e7
    Variance_tau = 0
    best_loss = float('inf')
    no_improve_count = 0
    max_noimprove = 0


    for k in range(iterations):
        # print(f"{k + 1} iteration")    
        if c.grad is not None:
            c.grad.zero_()
        if noise.grad is not None:
            noise.grad.zero_()

        gamma = cyclic_beta_scheduler(k)
        Nconcen = model.smoother_c(c)  # (1, 40)
        NSRO = model.smoother_w(w)  # (1, 40)

        stress_generate = model.decoder(noise, Nconcen, NSRO)
        mu, logvar, Nconcen_predt_generate, NSRO_predt_generate = model.encoder(stress_generate)

        erro_c = []
        diff = Nconcen_predt_generate[0] - Nconcen[0]
        for j in range(len(Nconcen[0])):
            if Nconcen[0][j] >= 0.01:
                mr = abs(diff[j] / Nconcen[0][j])
                erro_c.append(mr)

        erro_w = []
        diff = NSRO_predt_generate[0] - NSRO[0]
        for j in range(len(Nconcen[0])):
            if NSRO[0][j] >= 0.01:
                mr = abs(diff[j] / NSRO[0][j])
                erro_w.append(mr)

        self_check = sum(erro_c)/len(erro_c) + sum(erro_w)/len(erro_w)

        tau = (1/math.sqrt(6)) * (stress_generate[0,0] - stress_generate[0,1] + stress_generate[0,4] - stress_generate[0,5])
        variance_tau = torch.var(tau)
        # mean_s = torch.mean(stress_generate)


        c_means = c.mean(dim=[2, 3, 4])  
        
        mean_loss_c = torch.mean((c_means - c_means.mean())**2)  
        # mean_loss_s = (mean_s - 0.5)**2
        
        total_loss = alpha*(1/variance_tau) + gamma*self_check + mean_loss_c #+ mean_loss_s

        total_loss.backward()

        with torch.no_grad():
            c -= 100*learning_rate * c.grad
            c.data = torch.clamp(c.data, 0, 1)
            noise -= 100*learning_rate * noise.grad

        C[i,k] = c.cpu().detach().numpy().reshape(2,4,4,4)

    S[i] = stress_generate.cpu().detach().numpy().reshape(6,4,4,4)
    V_TAU[i] = variance_tau.cpu().detach().numpy()
    ERRO_C[i] = torch.mean(torch.tensor(erro_c)).cpu().detach().numpy()
    ERRO_W[i] = torch.mean(torch.tensor(erro_w)).cpu().detach().numpy()

# substitute with your path
np.save(file=new_folder_path+f"/C.npy", arr=C)
np.save(file=new_folder_path+f"/S.npy", arr=S)
np.save(file=new_folder_path+f"/V_TAU.npy", arr=V_TAU)
np.save(file=new_folder_path+f"/ERRO_C.npy", arr=ERRO_C)
np.save(file=new_folder_path+f"/ERRO_W.npy", arr=ERRO_W)
print(f"done")