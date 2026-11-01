import json,base64,glob,os
d=json.load(open('data.json'))
img={os.path.basename(f)[:-4]:'data:image/png;base64,'+base64.b64encode(open(f,'rb').read()).decode() for f in glob.glob('strips/*.png')}
b=json.load(open('../runs/baselines.json'))
pick=lambda m:{k:m[k] for k in('mse','ssim','nyquist_power') if k in m}
ex={'pca512':pick(b['pca512']),'original':pick(b['original'])}
for r in('R0','R1','R2','A1','A2'):
    ex[r]=pick(json.load(open(f'../runs/{r}_metrics.json')))
t=open('template.html').read().replace('{{DATA}}',json.dumps(d,separators=(',',':'))).replace('{{IMG}}',json.dumps(img)).replace('{{EXTRA}}',json.dumps(ex)).replace('{{FEAT}}',open('feat.json').read())
open('vom_farbklumpen_zum_frosch.html','w').write(t)
