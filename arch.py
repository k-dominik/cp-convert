from cellpose.resnet_torch import CPnet


class CPnet2(CPnet):
    def forward(self, data):
        out, _ = super().forward(data)
        return out
