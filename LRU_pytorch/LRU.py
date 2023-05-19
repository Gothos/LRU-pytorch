import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRU(nn.Module):
    def __init__(self,in_features,out_features,state_features, rmin=0, rmax=1,max_phase=6.283):
        super().__init__()
        self.out_features=out_features
        self.D=nn.Parameter(torch.randn([out_features,in_features])/math.sqrt(in_features))
        u1=torch.rand(state_features)
        u2=torch.rand(state_features)
        self.nu_log= nn.Parameter(torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        self.theta_log= nn.Parameter(torch.log(max_phase*u2))
        Lambda_mod=torch.exp(-torch.exp(self.nu_log))
        self.gamma_log=nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod)-torch.square(Lambda_mod))))
        B_re=torch.randn([state_features,in_features])/math.sqrt(2*in_features)
        B_im=torch.randn([state_features,in_features])/math.sqrt(2*in_features)
        self.B=nn.Parameter(torch.complex(B_re,B_im))
        C_re=torch.randn([out_features,state_features])/math.sqrt(state_features)
        C_im=torch.randn([out_features,state_features])/math.sqrt(state_features)
        self.C=nn.Parameter(torch.complex(C_re,C_im))
        self.state=torch.complex(torch.zeros(state_features),torch.zeros(state_features))

    def forward(self, input,state=None):
        self.state=self.state.to(self.B.device) if state==None else state
        Lambda_mod=torch.exp(-torch.exp(self.nu_log))
        Lambda_re=Lambda_mod*torch.cos(torch.exp(self.theta_log))
        Lambda_im=Lambda_mod*torch.sin(torch.exp(self.theta_log))
        Lambda=torch.complex(Lambda_re,Lambda_im)
        Lambda=Lambda.to(self.state.device)
        gammas=torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        gammas=gammas.to(self.state.device)
        output=torch.empty([i for i in input.shape] +[self.out_features],device=self.B.device)
        #Handle input of (Batches,Seq_length, Input size)
        if input.dim()==3:
            for i,batch in enumerate(input):
                out_seq=torch.empty(input.shape[1],self.out_features)
                for j,step in enumerate(batch):
                    self.state=(Lambda@self.state + gammas* self.B@step.to(dtype= self.B.dtype))
                    out_step= (self.C@self.state).real + self.D@step
                    out_seq[j]=out_step
                self.state=torch.complex(torch.zeros_like(self.state.real),torch.zeros_like(self.state.real))
                output[i]=out_seq
        #Handle input of (Seq_length, Input size)
        if input.dim()==2:
            for i,step in enumerate(input):
                self.state=(Lambda@self.state + gammas* self.B@step.to(dtype= self.B.dtype))
                out_step= (self.C@self.state).real + self.D@step
                output[i]=out_step
            self.state=torch.complex(torch.zeros_like(self.state.real),torch.zeros_like(self.state.real))
        return output
