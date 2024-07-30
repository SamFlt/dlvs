import torch

class Mcosc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        cosx = torch.cos(x)
        below_threshold = torch.abs(x) < 2.5e-4
        ctx.save_for_backward(cosx, x, below_threshold)
        return torch.where(below_threshold, torch.ones_like(x) / 2.0, ((1.0 - cosx) / x ** 2))
        

    @staticmethod
    def backward(ctx, grad_output):
        cosx, x, b = ctx.saved_tensors
        # gcosx = torch.where(b, torch.zeros_like(cosx), -(1 / x ** 2))
        gx = torch.where(b, torch.zeros_like(x), (x * torch.sin(x) - 2 * (-cosx + 1)) / (x ** 3))
        
        return grad_output * gx
class Msinc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sinx = torch.sin(x)
        sincx = sinx / x
        below_threshold = torch.abs(x) < 2.5e-4
        ctx.save_for_backward(sinx, x, below_threshold)
        return torch.where(below_threshold, torch.ones_like(x) / 6.0, ((1.0 - sincx) / x ** 2))
        

    @staticmethod
    def backward(ctx, grad_output):
        sinx, x, b = ctx.saved_tensors
        cosx = torch.cos(x)
        gx = torch.where(b, torch.zeros_like(x), (-x * cosx - 2 * x + 3 * sinx) /(x ** 4))
        return grad_output * gx

class Sinc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        below_threshold = torch.abs(x) < 1e-8
        sinx = torch.sin(x)
        ctx.save_for_backward(sinx, x, below_threshold)
        return torch.where(below_threshold, torch.ones_like(x), sinx / x)
        

    @staticmethod
    def backward(ctx, grad_output):
        sinx, x, b = ctx.saved_tensors
        cosx = torch.cos(x)
        gx = torch.where(b, torch.zeros_like(x), (cosx * x - sinx) / (x ** 2))
        return grad_output * gx



def fmcosc(cosx, x):
    return torch.where(torch.abs(x) < 2.5e-4, torch.ones_like(x) / 2.0, ((1.0 - cosx) / x ** 2))
def fmsinc(sincx, x):
    return torch.where(torch.abs(x) < 2.5e-4, torch.ones_like(x) / 6.0, ((1.0 - sincx) / x ** 2))
def fsinc(s, x):
    return torch.where(torch.abs(x) < 1e-8, torch.ones_like(x), s / x)

def torch_se3_exponential_map(vs):
    B = vs.size(0)
    T = torch.zeros((B, 4, 4), device=vs.device)
    T[:, 3, 3] = 1
    us = vs[:, 3:]
    theta = torch.norm(us, dim=-1, p=2, keepdim=False)
    c = torch.cos(theta)
    sinc = Sinc.apply(theta)
    msinc = Msinc.apply(theta)
    mcosc = Mcosc.apply(theta)
    tx, ty, tz = vs[:, 0], vs[:, 1], vs[:, 2]
    ux, uy, uz = us[:, 0], us[:, 1], us[:, 2]
    ux2, uy2, uz2 = ux ** 2, uy ** 2, uz ** 2
    uxy, uxz, uyz = ux * uy, ux * uz, uy * uz
    T[:, 0, 0] = c + mcosc * ux2
    T[:, 0, 1] = -sinc * uz + mcosc * uxy
    T[:, 0, 2] = sinc * uy + mcosc * uxz
    T[:, 1, 0] = sinc * uz + mcosc * uxy
    T[:, 1, 1] = c + mcosc* uy2
    T[:, 1, 2] = -sinc*ux + mcosc* uyz
    T[:, 2, 0] = -sinc*uy + mcosc * uxz
    T[:, 2, 1] = sinc*ux + mcosc * uyz
    T[:, 2, 2] = c + mcosc* uz2
    
    
    T[:, 0, 3] = tx * (sinc + ux2 * msinc) + \
                 ty * (uxy * msinc - uz * mcosc) + \
                 tz * (uxz * msinc + uy * mcosc)
    T[:, 1, 3] = tx * (uxy * msinc + uz * mcosc) + \
                 ty * (sinc + uy2 * msinc) + \
                 tz * (uyz * msinc - ux * mcosc)
    T[:, 2, 3] = tx * (uxz * msinc - uy * mcosc) + \
                 ty * (uyz * msinc + ux * mcosc) + \
                 tz * (sinc + uz2 * msinc)

    # I = torch.eye(3).view(1, 3, 3).repeat(B, 1, 1) * theta
    # t2 = (1 - c) * us
    # t3 = (theta - torch.sin(theta)) * (us ** 2)
    # t = torch.einsum('brc, bc -> bc', (I + t2 + t3), vs[:, :3])
    # T[:, :3, 3] = t


    return T

def torch_se3_inverse_map(T):
    def build_tu(R):
        B = R.size(0)
        s = (R[:, 1, 0] - R[:, 0, 1]) ** 2 + \
            (R[:, 2, 0] - R[:, 0, 2]) ** 2 + \
            (R[:, 2, 1] - R[:, 1, 2]) ** 2
        s = torch.sqrt(s)/2.0;
        c = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]-1.0) / 2.0
        theta = torch.atan2(s, c)
        tu = torch.zeros((B, 3), device=R.device)

        rv1 = torch.zeros_like(tu)


        sinc = fsinc(s, theta)
        sinc2 = 2 * sinc
        rv1[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / sinc2
        rv1[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / sinc2
        rv1[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / sinc2
        
        rv2 = torch.zeros_like(tu)
        eps = 1e-8
        zero_col = torch.zeros_like(rv2[:, 0])
        def else_val(i):
            return theta * torch.sqrt((R[:, i, i] - c) / (1 - c))
        rv2[:, 0] = torch.where(R[:, 0, 0] - c < eps, zero_col, else_val(0))
        rv2[:, 1] = torch.where(R[:, 1, 1] - c < eps, zero_col, else_val(1))
        rv2[:, 2] = torch.where(R[:, 2, 2] - c < eps, zero_col, else_val(2))

        rv2[:, 0] = torch.where((R[:, 2, 1] - R[:, 1, 2]) < 0, -rv2[:, 0], rv2[:, 0])
        rv2[:, 1] = torch.where((R[:, 0, 2] - R[:, 2, 0]) < 0, -rv2[:, 1], rv2[:, 1])
        rv2[:, 2] = torch.where((R[:, 1, 0] - R[:, 0, 1]) < 0, -rv2[:, 2], rv2[:, 2])
        
        
        


        return torch.where((c + 1).unsqueeze(-1) > 1e-4, rv1, rv2)
    B = T.size(0)
    tu = build_tu(T)



    v = torch.zeros((B, 6), device=T.device)
    v[:, 3:] = tu
    
    theta = torch.norm(tu, dim=-1, keepdim=False)
    c = torch.cos(theta)
    sinc = fsinc(torch.sin(theta), theta)
    msinc = fmsinc(sinc, theta)
    mcosc = fmcosc(c, theta)

    ux, uy, uz = tu[:, 0], tu[:, 1], tu[:, 2]
    ux2, uy2, uz2 = ux ** 2, uy ** 2, uz ** 2
    uxy, uxz, uyz = ux * uy, ux * uz, uy * uz
    
    a = torch.zeros((B, 3, 3), device=T.device)
    a[:, 0, 0] = sinc + ux2 * msinc
    a[:, 0, 1] = uxy * msinc - uz * mcosc
    a[:, 0, 2] = uxz * msinc + uy * mcosc

    a[:, 1, 0] = uxy * msinc + uz * mcosc
    a[:, 1, 1] = sinc + uy2 * msinc
    a[:, 1, 2] = uyz * msinc - ux * mcosc
    
    a[:, 2, 0] = uxz * msinc - uy * mcosc
    a[:, 2, 1] = uyz * msinc + ux * mcosc
    a[:, 2, 2] = sinc + uz2 * msinc
    
    det = a[:, 0, 0] * a[:, 1, 1] * a[:, 2, 2] + a[:, 1, 0] * a[:, 2, 1] * a[:, 0, 2] + \
          a[:, 0, 1] * a[:, 1, 2] * a[:, 2, 0] - a[:, 2, 0] * a[:, 1, 1] * a[:, 0, 2] - \
          a[:, 1, 0] * a[:, 0, 1] * a[:, 2, 2] - a[:, 0, 0] * a[:, 2, 1] * a[:, 1, 2]
    
    for i in range(B):
        ai = a[i]
        Ti = T[i]
        if torch.abs(det[i]) > 1e-5:
            
            v[i, 0] =  (Ti[0][3]*ai[1][1]*ai[2][2] +\
                        Ti[1][3]*ai[2][1]*ai[0][2] +\
                        Ti[2][3]*ai[0][1]*ai[1][2] -\
                        Ti[2][3]*ai[1][1]*ai[0][2] -\
                        Ti[1][3]*ai[0][1]*ai[2][2] -\
                        Ti[0][3]*ai[2][1]*ai[1][2])/det[i]
            v[i, 1] =  (ai[0][0]*Ti[1][3]*ai[2][2] +\
                        ai[1][0]*Ti[2][3]*ai[0][2] +\
                        Ti[0][3]*ai[1][2]*ai[2][0] -\
                        ai[2][0]*Ti[1][3]*ai[0][2] -\
                        ai[1][0]*Ti[0][3]*ai[2][2] -\
                        ai[0][0]*Ti[2][3]*ai[1][2])/det[i]
            v[i, 2] =  (ai[0][0]*ai[1][1]*Ti[2][3] +\
                        ai[1][0]*ai[2][1]*Ti[0][3] +\
                        ai[0][1]*Ti[1][3]*ai[2][0] -\
                        ai[2][0]*ai[1][1]*Ti[0][3] -\
                        ai[1][0]*ai[0][1]*Ti[2][3] -\
                        ai[0][0]*ai[2][1]*Ti[1][3])/det[i]
        else:
            v[i, 0] = T[i, 0, 3]
            v[i, 1] = T[i, 1, 3]
            v[i, 2] = T[i, 2, 3]

    return v
    

def torch_homogeneous_inverse(aTbs):
    B = aTbs.size(0)
    bTas = torch.eye(4, device=aTbs.device).unsqueeze(0).repeat(B, 1, 1)

    
    Rs = aTbs[:, :3, :3]
    ts = aTbs[:, :3, 3]
    rts = torch.transpose(Rs, 1, 2)
    # Batch matrix vector multiplication
    tinvs = -(torch.einsum('ijk,ik->ij', rts, ts))
    bTas[:, :3, :3] = rts
    bTas[:, :3, 3] = tinvs
    return bTas

def torch_inverse_velocities(vs):
    Ts = torch_se3_exponential_map(vs)
    Tinvs = torch_homogeneous_inverse(Ts)
    vinv = torch_se3_inverse_map(Tinvs)
    return vinv


if __name__ == '__main__':
    bs = 10
    v = torch.rand(bs, 6)
    v.requires_grad = True
    from time import time
    start = time()
    execs = 100
    for i in range(execs):
        T = torch_se3_exponential_map(v)
    elapsed = (time() - start) / execs
    print(f'average time for exponential map for batch size {bs}: {elapsed}')
    start = time()
    for i in range(execs):
        vv = torch_se3_inverse_map(T)
        l = vv.sum()
        
    elapsed = (time() - start) / execs
    print(f'average time for inverse map for batch size {bs}: {elapsed}')
    
    T = torch_se3_exponential_map(v)
    vv = torch_se3_inverse_map(T)
    print(torch.mean(torch.abs(vv - v)))
    l = vv.sum()
    l.backward()
    print(v.grad)
    
    print(v, torch_inverse_velocities(v))
    print(v + torch_inverse_velocities(v))
    

    
    