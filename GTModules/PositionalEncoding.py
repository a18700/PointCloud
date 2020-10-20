def PositionalEncoding_gpu(x_emb,x):
    #depth BxN
    x_new = x-x.mean(dim=2).view(x.size(0),x.size(1),1)
    #max_dist = torch.max(torch.sqrt(torch.sum(x_new**2,dim=1,keepdim=True)),dim=2,keepdim=True)[0]

    depth = torch.zeros(x.size(0),x.size(2)).type(torch.LongTensor).cuda()
    pe = torch.zeros_like(x_emb).cuda()
    distances = torch.sqrt(torch.sum(x_new**2,dim=1,keepdim=True)) # Bx1xN
    distances = (10*distances).type(torch.LongTensor)

    depth = distances.view(x.size(0),x.size(2))
    idx = torch.tensor(range(x_emb.size(1))).repeat(x_emb.size(0)*x_emb.size(2),1).view(x_emb.size(0),x_emb.size(1),x_emb.size(2)).cuda()
    depth = depth.repeat(1,x_emb.size(1)).view(x_emb.size(0),x_emb.size(1),x_emb.size(2)).type(torch.FloatTensor).cuda()
    temp= depth/(10000**(idx.type(torch.FloatTensor)/x_emb.size(1)).cuda())
    pe = torch.from_numpy(np.sin(temp.cpu().numpy())).cuda()
    return pe # BxNxD

def PositionalEncoding_cpu(x_emb,x):
    x_new = x-x.mean(dim=2).view(x.size(0),x.size(1),1)
    
    distances = torch.sqrt(torch.sum(x_new**2,dim=1,keepdim=True))
    depth = distances.view(x.size(0),x.size(2)) 
    idx = torch.tensor(range(x_emb.size(1))).repeat(x_emb.size(0)*x_emb.size(2),1).view(x_emb.size(0),x_emb.size(1),x_emb.size(2))
    depth = depth.repeat(1,x_emb.size(1)).view(x_emb.size(0),x_emb.size(1),x_emb.size(2)).type(torch.FloatTensor)
    temp= depth/(10000**(idx.type(torch.FloatTensor)/x_emb.size(1)))
    pe = torch.from_numpy(np.sin(temp.numpy())).cuda()
    return pe

class PositionalEncoding_mlp(nn.Module):
    def __init__(self,num_points):
        super(PositionalEncoding_mlp, self).__init__()
        self.num_points = num_points
        self.nn = nn.Sequential(nn.Linear(3, 64),
                                nn.BatchNorm1d(num_points),
                                nn.Tanh(),
                                nn.Linear(64, 128),
                                nn.BatchNorm1d(num_points),
                                nn.Tanh(),
                                nn.Linear(128, 256),
                                nn.BatchNorm1d(num_points),
                                nn.Tanh(),
                                nn.Linear(256, 512),
                                nn.BatchNorm1d(num_points),
                                nn.Tanh())
    

    def forward(self, x_emb, x):
        # Absolute PE
        x_new = x-x.mean(dim=2).view(x.size(0),x.size(1),1)
        
        r = torch.sqrt(torch.sum(x_new**2,dim=1,keepdim=True))
        r = r.view(x.size(0),x.size(2),1)
        
        azimuth = torch.atan(x_new.transpose(1,0)[1]/x_new.transpose(1,0)[0])#.view(x.size(0),x.size(2),1)   BxN
        nan = (azimuth!=azimuth)
        pos = (x_new.transpose(1,0)[1]>0)
        neg = (x_new.transpose(1,0)[1]<=0)
        azimuth[nan*pos] = torch.tensor(math.pi/2).cuda()
        azimuth[nan*neg] = torch.tensor(3*math.pi/2).cuda()
        azimuth = azimuth.view(x.size(0),x.size(2),1)
        inclination = torch.acos(x_new.transpose(1,0)[2]/r.view(x.size(0),x.size(2)))#.view(x.size(0),x.size(2),1) #x_new: 8x1024, r: 8x1024x1
        inclination[inclination!=inclination]=torch.tensor(0).cuda()
        inclination = inclination.view(x.size(0),x.size(2),1)
        total = torch.cat((r,azimuth,inclination),dim=-1)
        pe = self.nn(total)
        return pe.transpose(2,1)
        
        
