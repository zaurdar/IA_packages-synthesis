import numpy as np
class preprocessing_pipeline:
    def __init__(self, tree=None):
        def fnct(x):
            return x
        if tree is None:
            tree = {"father":{"lil_bro":"dawn", "fnct":fnct,"param":{}},"dawn":{"lil_bro":"dawn", "fnct":fnct,"param":{}}}
        self.tree = tree
    def join(self,fnct,last_node,node,param):
        self.tree[last_node]["lil_bro"] = node
        self.tree[node] = {"lil_bro":"dawn","fnct":fnct,"param":param}
    def fit(self):
        temp ="father"
        pipe = lambda x: x
        while self.tree[temp]["lil_bro"] != "dawn":
            temp = self.tree[temp]["lil_bro"]
            f = self.tree[temp]["fnct"]
            param = self.tree[temp]["param"]
            pipe = lambda x, pipe=pipe, f=f, params=param: f(pipe(x), **param)
        return pipe
    def rendering(self,shape):
        lines = []
        temp = "father"
        step = 0
        dummy = np.zeros(shape)
        out_dummy = self.tree[temp]["fnct"](dummy)
        lines.append(f"└── [{step}──node : {temp}────────────────────────────────")
        while self.tree[temp]["lil_bro"] != "dawn":
            temp = self.tree[temp]["lil_bro"]
            step+=1
            dummy = out_dummy
            params = self.tree[temp]["param"]
            out_dummy = self.tree[temp]["fnct"](dummy, **params)
            lines.append(f"├── [{step}──node : {temp}──input_shape : [{dummy.shape}]──output_shape : [{out_dummy.shape}]")
        for line in lines:
            print(line)