"""Anteile der Noten-Teile auf den ersten 1.000 Testbildern: frisch initialisiert und nach dem Training."""
import os,sys,json
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
NB=sys.argv[1]; sys.path.insert(0,NB+"/v6_experiments")
import numpy as np, tensorflow as tf, keras
from keras import models
import exp
_,xt=exp.load_data(); x=xt[:1000]
def run(fn,model):
    acc={}
    n=0
    for i in range(0,len(x),100):
        b=x[i:i+100]; p=model(b,training=False)
        for k,v in fn(b,p).items(): acc[k]=acc.get(k,0)+float(tf.reduce_mean(v))
        n+=1
    return {k:v/n for k,v in acc.items()}
def show(name,d):
    t=sum(d.values()); print(name,{k:round(v,5) for k,v in d.items()},'summe',round(t,5),'anteile',{k:round(100*v/t,1) for k,v in d.items()},flush=True); return d
out={}
keras.utils.set_random_seed(42)
# v1, v3: nur ein Teil, nur Summe
v1=models.load_model(NB+"/models/autoencoder_v1.keras",compile=False)
bce=lambda a,b:{"bce":keras.losses.binary_crossentropy(a,b,axis=(1,2,3))}
out["v1_start"]=show("v1 start",run(bce,keras.models.clone_model(v1)));out["v1_end"]=show("v1 ende",run(bce,v1))
v3=models.load_model(NB+"/models/autoencoder_v3.keras",compile=False)
mse=lambda a,b:{"mse":tf.reduce_mean(tf.square(a-b),axis=[1,2,3])}
out["v3_start"]=show("v3 start",run(mse,keras.models.clone_model(v3)));out["v3_end"]=show("v3 ende",run(mse,v3))
# v4 echt: Gewicht 1e-5, ohne Größenänderung
v4=models.load_model(NB+"/models/autoencoder_v4.keras",compile=False)
ext=exp.vgg_extractor(32,["block1_conv2","block2_conv2","block3_conv3"])
def v4t(a,b):
    t=ext(exp.to_vgg(a),training=False);p=ext(exp.to_vgg(b),training=False)
    ls=[tf.reduce_mean(tf.square(u-w),axis=[1,2,3]) for u,w in zip(t,p)]
    return {"pixel":tf.reduce_mean(tf.square(a-b),axis=[1,2,3]),"vgg_w":1e-5*sum(ls)/3,"l1raw":ls[0],"l2raw":ls[1],"l3raw":ls[2]}
f=lambda a,b:{k:v for k,v in v4t(a,b).items()}
for tag,m in (("start",keras.models.clone_model(v4)),("ende",v4)):
    d=run(f,m); print("v4",tag,{k:round(v,5) for k,v in d.items()},"vgg/pixel",round(d["vgg_w"]/d["pixel"],1),"schichten %",[round(100*d[k]/(d["l1raw"]+d["l2raw"]+d["l3raw"]),1) for k in("l1raw","l2raw","l3raw")],flush=True); out["v4_"+tag]=d
# v5
r1=exp.make_r1_loss(0.2)
v5=models.load_model(NB+"/models/autoencoder_v5.keras",compile=False)
out["v5_start"]=show("v5 start",run(r1,keras.models.clone_model(v5)));out["v5_end"]=show("v5 ende",run(r1,v5))
def w(tag,widths=(64,128,256)):
    m=exp.build_v5(8,widths); m.load_weights(f"{NB}/v6_experiments/runs/{tag}.weights.h5"); return m
out["B1_start"]=show("B1 start",run(r1,exp.build_v5(8,(64,128,256))));out["B1_end"]=show("B1 ende",run(r1,w("B1")))
out["B0_end"]=show("B0 ende",run(r1,w("B0",(32,64,128))))
out["B3_end"]=show("B3 ende (nur L1+SSIM)",run(r1,w("B3")))
r2=exp.make_r2_loss(0.2,0.1,0.01,48,["block1_conv2","block2_conv2","block3_conv3"])
out["B2_start"]=show("B2 start",run(r2,exp.build_v5(8,(64,128,256))));out["B2_end"]=show("B2 ende",run(r2,w("B2")))
json.dump(out,open(sys.argv[2],"w"),indent=1)
