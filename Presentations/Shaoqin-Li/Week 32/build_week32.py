"""Build Week 32 from saved P08/P09 evidence; no training or simulation."""
from pathlib import Path
import json, statistics as st, hashlib, html, base64, textwrap, subprocess, sys, re, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/sql/claude_brain/projects/embodied_ai_r1/video_to_r1_pipeline/results')
W=ROOT/'week_queue_2026-10-29'; N=ROOT/'p09_s6_2026-10-29'
AS=HERE/'assets'; AS.mkdir(exist_ok=True)
NAVY='#182C40'; TEAL='#087F80'; RUST='#B24B43'; GOLD='#B68428'; BG='#F7F8FA'; GRAY='#516170'; INK='#1D2E40'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':BG,'axes.facecolor':BG,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
E={}; paths=[]
def load(p):
 paths.append(p); return json.loads(p.read_text())
for row in ['A1','A2','A3','A4','A5']:
 d=load(next((N/row/'raw').glob('*_s6.json'))); r=d['episodes']; parity=load(next((N/row).glob('export_parity_*.json')))
 E[row]={'n':len(r),'pass':sum(x['status']=='PASS' for x in r),'complete':sum(x['survived'] for x in r),'gates':{k:sum(x['gates'][k] for x in r) for k in r[0]['gates']},'rms':st.mean(x['joint_rms_mean'] for x in r),'survival':st.median(x['seconds'] for x in r),'flight':[min(x['flight_s'] for x in r),max(x['flight_s'] for x in r)],'parity':max(parity['parity_rollout_max_abs'],parity['parity_random_max_abs'])}
 E[row]['nonflight']=sum(all(v for k,v in x['gates'].items() if k!='flight') for x in r)
 E[row]['seconds']=int(re.search(r'elapsed_s=(\d+)',(N/row/'train_status.txt').read_text()).group(1))
for label,folder,tag in [('baseline',W/'S1/rework_v2','m20496v2'),('S3',W/'S3/raw','s3m23495'),('S4',W/'S4/raw','s4m26300'),('bankB',W/'S5/bankB','m20496_bankB')]:
 E[label]={}
 for cell in ['B0','T','R','F-low','F-high','D25','D50']:
  d=load(folder/f'{tag}_{cell}.json');r=d['episodes'];land=[ev['landing_peak_bw'] for x in r for ev in x['events'] if ev.get('landing_peak_bw') is not None]
  E[label][cell]={'pass':sum(x['status']=='PASS' for x in r),'falls':sum(bool(x['fell']) for x in r),'landing_mean':st.mean(land) if land else None,'landing_max':max(land) if land else None}
D={k:load(N/k/'train_diag_every250.json') for k in ['A1','A2','A3','A4','A5']}
P04=ROOT/'run_jump_P04_2026-10-28'
P04_RUN=load(P04/'raw/runftv2_model_14498_vx2.json')
P04_BEFORE=load(P04/'raw/jump_model_17497_jump_vx2.json')
P04_AFTER=load(P04/'raw/jumpapex_model_20496_jump_vx2.json')
assert P04_RUN['summary']['passed']==20 and P04_BEFORE['summary']['passed']==0 and P04_AFTER['summary']['passed']==20
paths.extend([P04/'README.md',P04/'JUMP_STAGE.md',P04/'BRANCH_B_retarget_repair.md',HERE.parent/'week_31/speaker_script_week31.md'])
for old,new in [('poster_walk_run_delay.png','week31_historical_walk.png'),('poster_jumping_jacks.png','week31_historical_jacks.png')]:
 src=HERE.parent/'week_31/week31_shots'/old;shutil.copy2(src,AS/new);paths.append(src)

(HERE/'verified_metrics.json').write_text(json.dumps(E,indent=2))
(HERE/'evidence_manifest.json').write_text(json.dumps([{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],indent=2))
# Genuine saved-data audit output, captured by the builder (not a training console).
audit=['WEEK 32 / SAVED-DATA AUDIT','Source: P09 raw episode JSONs','row  clip complete  full L1  median(s)  joint RMS(rad)']
for k in ['A1','A2','A3','A4','A5']:
 x=E[k];audit.append(f'{k:3}  {x["complete"]:2}/20          {x["pass"]:2}/20    {x["survival"]:5.2f}      {x["rms"]:.3f}')
audit+=['','A4: 16/20 meet all gates EXCEPT flight.','All full L1 results remain FAIL.','No simulation or training ran in this audit.']
text='\n'.join(audit);print(text);(HERE/'audit_output.txt').write_text(text+'\n')
im=Image.new('RGB',(1440,600),'#142638');draw=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',28)
for i,line in enumerate(audit): draw.text((35,26+i*43),line,font=font,fill='#DCEAEF' if i else '#65D4C0')
im.save(AS/'audit.png')
def savechart(name):
 plt.tight_layout(pad=1.5);plt.savefig(AS/name,dpi=170,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(10,3.8));cells=['B0','T','R','F-low','F-high'];v=[E['baseline'][c]['pass'] for c in cells]
ax.bar(cells,v,color=[TEAL,RUST,TEAL,TEAL,TEAL],width=.55);ax.set(ylim=(0,23),ylabel='Episodes passing / 20',xticklabels=['Nominal','Random timing','Two commands','Low friction','High friction'])
for i,y in enumerate(v):ax.text(i,y+.5,f'{y}/20',ha='center',fontweight='bold')
ax.text(2,22,'Repeated commands: 40/40 events in 20 episodes',ha='center',fontsize=10,color=GRAY);savechart('nominal.png')
fig,axes=plt.subplots(1,2,figsize=(10.5,3.7));cs=['B0','D25','D50'];labs=['0 ms','25 ms','50 ms']
for ax,key,title in [(axes[0],'pass','Full run-jump passes'),(axes[1],'falls','Episodes that fell')]:
 vals=[E['baseline'][c][key] for c in cs];ax.bar(labs,vals,color=[TEAL,RUST,RUST],width=.5);ax.set(ylim=(0,22),title=title)
 for i,y in enumerate(vals):ax.text(i,y+.5,f'{y}/20',ha='center')
savechart('delay.png')
fig,ax=plt.subplots(figsize=(9.7,3.8));labels=['Original / bank A','Landing candidate','Original / bank B'];keys=['baseline','S3','bankB'];xs=range(3)
ax.bar([x-.17 for x in xs],[E[k]['B0']['landing_mean'] for k in keys],width=.32,label='Mean',color=TEAL);ax.bar([x+.17 for x in xs],[E[k]['B0']['landing_max'] for k in keys],width=.32,label='Maximum',color=RUST)
ax.axhline(7,color=GOLD,ls='--',label='Candidate target: max 7 BW');ax.set(xticks=list(xs),xticklabels=labels,ylabel='Landing peak / body weight',ylim=(0,12));ax.legend(ncol=3,fontsize=10,loc='upper center')
for x,k in zip(xs,keys):
 for dx,field in [(-.17,'landing_mean'),(.17,'landing_max')]:ax.text(x+dx,E[k]['B0'][field]+.15,f'{E[k]["B0"][field]:.2f}',ha='center',fontsize=10)
savechart('landing.png')
fig,ax=plt.subplots(figsize=(10,3.9));ks=['A1','A2','A3','A4','A5'];ax.bar(ks,[E[k]['complete'] for k in ks],color=TEAL,label='Clip completed',width=.55);ax.scatter(ks,[E[k]['pass'] for k in ks],color=RUST,s=75,zorder=3,label='Full L1 passes');ax.set(ylim=(-1.5,23),ylabel='Episodes / 20');ax.legend(ncol=2,fontsize=11,loc='upper center')
for i,k in enumerate(ks):ax.text(i,E[k]['complete']+.8,f'{E[k]["complete"]}/20',ha='center');
ax.text(4,7,'c3 variant',ha='center',fontsize=11,color=GRAY);savechart('sweep.png')
fig,ax=plt.subplots(figsize=(10,3.9));g=['no_fall','flight','recovery','joint_rms'];x=range(4)
for dx,row,col in [(-.18,'A2',TEAL),(.18,'A4',NAVY)]:
 vals=[E[row]['gates'][k] for k in g];ax.bar([i+dx for i in x],vals,width=.33,label=row,color=col)
 for i,y in enumerate(vals):ax.text(i+dx,y+.3,str(y),ha='center',fontsize=11)
ax.set(xticks=list(x),xticklabels=['No fall/contact','Flight duration','Recovery speed','Joint-angle error'],ylim=(0,23),ylabel='Gate passes / 20');ax.legend(ncol=2,loc='upper center');savechart('gates.png')
fig,ax=plt.subplots(figsize=(10,3.7));
for i,k in enumerate(['A2','A4']):
 a,b=E[k]['flight'];ax.plot([a,b],[i,i],lw=12,solid_capstyle='round',color=TEAL if i==0 else NAVY);ax.text((a+b)/2,i+.16,f'{a:.3f}–{b:.3f} s',ha='center')
ax.axvline(.22,color=GRAY,ls=':',label='R1 reference geometry: 0.22 s');ax.axvline(.24,color=RUST,ls='--',label='Frozen physical-flight gate: 0.24 s');ax.set(yticks=[0,1],yticklabels=['A2','A4'],xlabel='Flight duration (seconds)',xlim=(.18,.255),ylim=(-.5,1.65));ax.legend(fontsize=10,loc='upper center',bbox_to_anchor=(.5,1.29),frameon=False);savechart('flight.png')
# Actual native-training summaries saved every 250 updates, not evaluation rollouts.
COLS={'A1':GOLD,'A2':TEAL,'A3':RUST,'A4':NAVY,'A5':'#8557A6'}
def tx(k):
 rows=D[k];start=rows[0]['iteration'];return [r['iteration']-start for r in rows]
fig,ax=plt.subplots(figsize=(10.5,4.2))
for k in ['A1','A2','A3','A4','A5']:
 label=k+(' (resumed A2)' if k=='A4' else ' (c3 variant)' if k=='A5' else '')
 ax.plot(tx(k),[r['Mean episode length']*.02 for r in D[k]],marker='o',ms=3,lw=2,color=COLS[k],label=label)
ax.set(xlabel='Updates since first saved diagnostic in this row',ylabel='Mean training episode duration (s)',ylim=(0,9));ax.legend(loc='upper left',ncol=2,fontsize=10)
savechart('training_episodes.png')
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for k in ['A2','A3','A4','A5']:
 axes[0].plot(tx(k),[r['Metrics/motion/p09_joint_rms'] for r in D[k]],marker='o',ms=3,color=COLS[k],label=k+(' resumed' if k=='A4' else ' variant' if k=='A5' else ''))
 axes[1].plot(tx(k),[r['Metrics/motion/error_anchor_lin_vel'] for r in D[k]],marker='o',ms=3,color=COLS[k])
axes[0].set(title='Episode-mean joint error',ylabel='Training joint RMS (rad)');axes[1].set(title='Anchor velocity error',ylabel='Last-step training error (m/s)')
for ax in axes:ax.set_xlabel('Updates since first saved diagnostic')
axes[0].legend(fontsize=10);savechart('training_errors.png')
fig,ax=plt.subplots(figsize=(10.5,4.2));a=ax.twinx();xs=tx('A1')
l1=ax.plot(xs,[r['Mean episode length']*.02 for r in D['A1']],color=TEAL,marker='o',lw=2,label='Mean training episode duration')
l2=a.plot(xs,[r['Curriculum/ee_z_threshold'] for r in D['A1']],color=RUST,ls='--',marker='s',lw=2,label='Training end-effector tolerance')
ax.set(xlabel='A1 training update',ylabel='Mean training episode duration (s)',ylim=(0,6));a.set(ylabel='Training tolerance (m)',ylim=(.2,.6));ax.legend(l1+l2,[x.get_label() for x in l1+l2],fontsize=10,loc='upper left');savechart('training_curriculum.png')

fig,ax=plt.subplots(figsize=(11,3.8));ax.set_xlim(0,11);ax.set_ylim(0,3);ax.axis('off')
from matplotlib.patches import FancyBboxPatch
phases=[('RUN','Alternating steps'),('TAKE OFF','Push off once'),('FLIGHT','Both feet airborne'),('LAND','Recover support'),('RUN AGAIN','Recover speed')]
for i,(name,desc) in enumerate(phases):
 x=.1+i*2.2;box=FancyBboxPatch((x,1.05),1.95,1.2,boxstyle='round,pad=0.05',fc=TEAL if i==2 else '#E2EAEF',ec='none');ax.add_patch(box)
 ax.text(x+.975,1.82,name,ha='center',va='center',fontsize=15,color='white' if i==2 else NAVY,fontweight='bold');ax.text(x+.975,1.32,desc,ha='center',fontsize=10,color='white' if i==2 else INK)
 if i<4:ax.annotate('',xy=(x+2.17,1.65),xytext=(x+1.98,1.65),arrowprops={'arrowstyle':'->','color':GRAY,'lw':2})
ax.text(5.5,.45,'One continuous physics rollout: no reset, teleport or separate animation between phases',ha='center',fontsize=12,color=GRAY)
savechart('week32_goal.png')

# Slide content is also the source of the box-by-box explanations and speaker script.
slides=[]
def S(title,sub,boxes,notes,source='',chart=None,take='',layout='cards'):
 slides.append(dict(title=title,sub=sub,boxes=[dict(title=a,text=b,meaning=c) for a,b,c in boxes],notes=notes,source=source,chart=chart,take=take,layout=layout))
S('Week 32 | From a jump command to motion tracking','Shaoqin Li · Applied and Embodied AI Lab · CCNY\nOctober 31, 2026 · Results through October 30 · Simulation only',[
 ('20/20','The commanded jump passed nominal tests.','This is the existing 84-observation policy in its nominal simulated test cell, not a physical R1.'),
 ('20/20','The best imitation run completed the reference clip.','Completion means no early termination; it does not imply every quality gate passed.'),
 ('0/20','The best imitation run passed the full L1 test.','Every imitation candidate still failed the unchanged full contract. This is the central limitation.')],
 'This week I tested two ways to make the R1 run and jump. The command policy passed its nominal tests. I also got an imitation policy through the whole reference clip, but it still failed the full success test. Everything I show is simulation.', 'P08 S1/S5; P09 A4 raw')
S('I asked three practical questions','A repeatable test mattered more than a single good-looking rollout.',[
 ('1 · Does the jump repeat?','I varied command timing, friction, payload and action delay.','A skill needs repeated evaluation beyond its one nominal setting.'),
 ('2 · Can I improve the weak spots?','I tried a landing penalty and delay training.','These are separate fine-tunes with explicit acceptance rules, not a single combined reward search.'),
 ('3 · Can I track a run-jump reference?','I tested five bounded imitation attempts; I skipped the sixth.','A1–A5 were trained; A6 was not admitted. Failed attempts remain part of the evidence.')],
 'I organized the week around these three questions. First I measured the command policy. Then I tried to reduce impact and improve delay tolerance. Finally I tested the reference-tracking approach.', 'P08/P09 plans and final reports')
S('I kept the two policy families separate','Their inputs, robot models and success tests are different.',[
 ('Command policy · 84 observations','Run command → jump countdown → learned response','The jump is requested by an explicit countdown; it is not replaying a human motion reference.'),
 ('Imitation policy · 135 observations','Human motion → R1 reference → tracking policy','The actor observes a reference-dependent tracking task, including time/reference semantics.'),
 ('Comparison rule','I reported each family under its own frozen gates.','Do not transfer landing forces, motor limits or success thresholds between the two embodiments.')],
 'These are two approaches to the same milestone. The command policy learns what to do when I request a jump. The tracking policy follows a retargeted motion. Their observation spaces and actuator models differ, so I do not compare them as matched experiments.', 'P08 manifest; P09 export parity/config manifests')
S('I froze the evaluation before choosing a model','Training ran in mjlab; strict evaluation used saved policies in CPU MuJoCo.',[
 ('Training setup','4,096 simulated environments\nOne GPU job at a time\nSaved checkpoints and configuration pins','The 4,096 count is the number of parallel simulated environments, not robots or independent training seeds.'),
 ('Evaluation setup','20 seeded episodes per row\nFull L1 target: at least 16/20\nOriginal c2 bank and thresholds','The fixed 7000–7019 bank supports model selection; it is not an untouched confirmation bank.'),
 ('Two different flight gates','Command policy: at least 0.12 s\nOriginal c2 imitation: at least 0.24 s','The imitation threshold is tied to the original human source. A previous report typo confused these gates; scoring used the correct values.')],
 'I loaded pinned checkpoints rather than a latest-file guess. I kept the original detector and thresholds during evaluation. For imitation, I used the same twenty initial-condition seeds to compare candidates; those are evaluation episodes, not twenty independent training runs.', 'P08 run_jump_metrics; P09 s6_eval.py')
S('I reproduced the nominal commanded jump','Original model_20496 · seed bank A · corrected evaluator',[],
 'The original command policy passed all twenty nominal episodes. Two commands per episode gave forty successful events. Random command timing was weaker: sixteen of twenty passed, below the eighteen-of-twenty acceptance target. Friction edge cells passed.', 'P08 S1/rework_v2 raw', 'nominal.png','Random command timing remained a failed robustness cell: 16/20.')
S('I found a clear action-delay boundary','The delay was inserted into the action path; this was not an observation-delay test.',[],
 'At twenty-five milliseconds, the composite run-jump test failed even though the policy did not fall. At fifty milliseconds, all twenty episodes fell. Survival and the complete skill test are different outcomes.', 'P08 S1/rework_v2 B0/D25/D50', 'delay.png','No fall did not mean the complete run-jump skill passed.')
S('I reduced landing peaks, but missed the target','The landing candidate kept nominal jump passes at 20/20.',[],
 'The landing candidate reduced the mean peak from 8.77 to 6.27 body weights and the maximum from 10.00 to 8.01. But the target was a maximum no greater than seven. It missed that target and was not promoted. Torque saturation also increased.', 'P08 S3/raw and S5/bankB B0', 'landing.png','The landing target failed; the original policy remained the fallback.')
S('I gained delay-tolerant running and lost the jump','S4 delay fine-tune · final model_26300 · no promotion',[
 ('Running at 25 / 50 ms','20/20 passed the run and no-fall checks.','Under the candidate, both delay cells stayed upright and met the running subgates.'),
 ('Full jump test','0/20 in every evaluated cell.','The candidate lost the jump even in the nominal setting; median B0 flight was 0.075 s with less than 1 mm rise.'),
 ('My decision','I rejected the candidate and kept model_20496.','Improving a subskill did not satisfy the predeclared all-gates promotion rule.')],
 'Delay training changed the behavior, but not in the way I needed. The policy could run under the tested delays without falling, while the jump disappeared. I stopped this branch rather than choosing an earlier checkpoint after seeing the failure.', 'P08 S4/raw; S5 decision',take='The desired run-jump delay envelope is still unsolved.')
S('Then I built the imitation test around a repaired reference','Pipeline: source motion → retarget → repair/audit → train → strict rollout',[
 ('Reference preparation','I checked joint ranges, contact transitions and stored body poses.','Kinematic admission catches representation and contact problems; it does not prove dynamic feasibility.'),
 ('Native diagnosis','The original pilot failed early in both the native environment and strict harness.','A harness-only defect could not explain the whole failure; full equivalence was not established.'),
 ('My next hypothesis','I tested whether start-phase exposure was blocking progress.','All 118 sampled starts before 2.5 seconds terminated on the end-effector condition in D0. This motivated A2, not a proof of causality.')],
 'The imitation path needed a reference the robot could at least represent. The original pilot failed during the approach, before the intended jump. I checked the stored motion and native environment, then tested changes to learning rather than declaring the robot incapable.', 'P09 D0; S6 admission')
S('I checked the saved result files directly','This is actual output from a CPU saved-data audit, not a training-console screenshot.',[],
 'This output was captured while rebuilding the presentation from the raw episode JSON files. It shows which attempts completed the clip and which passed the complete gate. It is an audit of saved results, not another training run.', 'build_week32.py → audit_output.txt; evidence_manifest.json','audit.png','Clip completion and full success stayed separate throughout the audit.')
S('I monitored the learning curves, not only the final score','Actual saved native-training diagnostics · sampled every 250 updates',[],
 'These are the actual training curves, not the final evaluation scores. A2 and resumed A4 developed longer episodes. A1 became shorter again as its tolerance tightened. A3 and the c3 variant remained short. The x-axis starts at the first saved diagnostic for each row; A4 already inherited A2 training, so these are not matched fresh starts. Training uses varied reference starts, pushes and randomization, so episode duration is not a full skill pass.', 'P09 A1–A5/train_diag_every250.json','training_episodes.png','Longer training episodes were useful evidence, but did not replace strict evaluation.')
S('I also tracked error during training','Native training summaries differ from the strict frame-zero evaluation.',[],
 'The left graph is episode-mean training joint RMS, and the right graph is anchor velocity error at the last step. I do not use these as L1 passes or compare them directly with the strict evaluation values. A4 is a resumed policy and A5 changes reference. A1 did not log the new joint RMS metric, so it is omitted rather than filled in. Each curve comes from its saved native-training diagnostics.', 'P09 A2–A5/train_diag_every250.json','training_errors.png','These are training diagnostics, not proof of full-clip success or a causal ablation.')
S('A1 shortened again when I restored strict tolerance','The training curriculum finished at the original 0.25 m tolerance.',[],
 'A1 started with a wider end-effector termination tolerance, then tightened it to the original setting. The duration curve fell sharply around the return to strict tolerance. This describes one training run; it does not prove a unique cause. The final strict frame-zero evaluation still failed all twenty episodes. The data were already logged; I did not rerun training to make this graph.', 'P09 A1/train_diag_every250.json','training_curriculum.png','The relaxed training phase did not yield a passing strict policy.')
S('Five attempts finished; none passed the full L1 gate','A1–A4: original c2 reference · A5: separately anchored c3 variant',[],
 'A1 relaxed the training termination threshold and then tightened it. A2 changed start-phase sampling. A3 increased velocity reward weight. A4 combined the A2 configuration with the velocity change and extra training. A5 changed the reference under the original sampler. Only A2 and A4 completed the clip; every full L1 count remained zero.', 'P09 A1–A5 raw JSONs','sweep.png','A6 was skipped; I did not reuse its slot for an unlisted experiment.')
S('I changed where training episodes started','A2 retained the full motion and its reference pose/velocity at every reset.',[
 ('25% · frame zero','I repeatedly exposed the policy to the actual beginning.','This supplies starts at the frame used by the strict evaluation.'),
 ('25% · early approach','I sampled starts between 0 and 1.5 seconds.','This increases exposure to the approach segment without cropping away the rest.'),
 ('25% · transition','I sampled starts between 2.0 and 3.4 seconds.','This covers takeoff and landing neighborhoods; the entire reference remains available.'),
 ('25% · adaptive','I kept the original full-clip adaptive sampler.','The existing sampler and failure bookkeeping still contribute one component.')],
 'I split the starts into four equal components. This was a training-only change. The evaluation still began at frame zero with the original termination threshold. A2 was the first attempt to complete the clip, but this is still a single-seed observation.', 'P09 A2 manifest; p09_cfgs_v2_copy.py')
S('I saw progress in recovery and joint tracking','A4 resumed from A2, added the velocity-weight change and trained 3,000 more updates.',[],
 'A4 recovered at the required speed in all twenty episodes. Eighteen met the joint-error gate, compared with zero for A2. The no-fall count stayed eighteen. Flight remained zero. Sixteen episodes met all non-flight gates together; I do not count them as full passes.', 'P09 A2/A4 raw JSONs','gates.png','A4 improved partial metrics, but all 20 episodes still failed full L1.')
S('Flight duration remained the binding common failure','The human-source threshold stayed fixed; I did not lower it to fit the results.',[],
 'A2 flew for 0.200 to 0.225 seconds. A4 flew for 0.195 to 0.215 seconds. The R1 reference geometry gives about 0.22 seconds of airborne time, while the physical gate is 0.24. That suggests a tension between close tracking and the flight requirement. Geometric reference flight and physical force-based flight are different measurements, so this does not prove impossibility.', 'P09 A2/A4 raw; S6 reference admission','flight.png','Reference feasibility needs investigation; the success gate still failed.')
S('I kept the extra-training confound visible','A2 and A4 are a lineage, not a matched ablation.',[
 ('A2 · fresh seed 48','3,000 updates\nMean joint RMS: 0.178 rad\nRecovery: 10/20','The start mixture is the scientific change relative to the original pilot.'),
 ('A4 · resumed seed 49','3,000 additional updates\nMean joint RMS: 0.148 rad\nRecovery: 20/20','The candidate also inherited A2 training and changed the velocity weight. Its adaptive sampler history restarted.'),
 ('What I can claim','The combined candidate improved these measured outcomes.','I cannot attribute the gain only to velocity weighting; extra training and the new seed are confounds.')],
 'The result is encouraging as engineering progress, but it is not an isolated test of the velocity reward. A4 had more total updates and a resumed training lineage. I need matched-budget, multi-seed tests before making a causal claim.', 'P09 A2/A4 manifests and raw',take='Lower joint error did not establish a complete running-jump skill.')
S('I tried one other reference and kept one slot unused','A5 and A6 were not pooled with the original c2 comparison.',[
 ('A5 · c3 variant','0/20 full passes\nMedian survival: 1.22 s\nAll 20 ended before the jump','This reference has its own 0.32-second flight anchor. It used the original sampler, not A2’s mixture.'),
 ('A6 · skipped','No admitted slower continuous reference existed within the bounded preparation scope.','Skipping is a protocol outcome, not a failed training run. Uniform slow motion would alter flight feasibility.'),
 ('What remained untested','c3 with the A2/A4 configuration','This is a future option. It was not run and is not presented as a result.')],
 'The alternate c3 reference still failed early under the original configuration. I did not train c3 with the successful start mixture in this sweep. I also did not manufacture a slower reference just to fill the sixth slot.', 'P09 A5 raw/admission; A6 README')
S('I fixed measurement and launch checks along the way','The fixes changed reliability of the workflow, not the success thresholds.',[
 ('Countdown evaluation','I corrected off-grid jump-command timing.','The evaluator had started some countdowns one tick late. Corrected T results still failed at 16/20 for the original model.'),
 ('Training budget','I caught a five-hour default in a two-hour resume launcher.','The A4 launch path was changed to enforce the two-hour cap before the shared watchdog was initialized.'),
 ('Reward caveat','The apex estimate is biased by a push during flight.','The issue was diagnosed and disclosed; the production reward and pushes stayed unchanged in this experiment.')],
 'I found bugs and measurement limitations during the work. I corrected the evaluator timing and resume cap before relying on the affected workflow. The apex reward’s behavior under pushes remained a disclosed limitation, not something silently repaired during the comparison.', 'P08 S1fix/S4pre; P09 resume launcher v2')
S('I stopped where the evidence stopped','No physical robot experiment or new rollout video is claimed.',[
 ('Simulation only','Nominal tests are not real-robot validation.','P09 did not include the delay, friction, mass or push robustness sweep used for the separate command policy.'),
 ('Single training seeds','Twenty evaluation episodes are not twenty trained models.','A4 also has a resumed lineage and extra compute. Generalization and uncertainty need new tests.'),
 ('Private reference data','LAFAN1 motions, derived models and videos stay private.','This deck uses numeric summaries and diagrams. No LAFAN-derived rollout is embedded; the workstation had no rendered video.')],
 'The most important boundary is that all of this is simulation. I have not validated these policies on hardware. I also do not have a new rollout video from this sweep. The plotted evidence comes from the saved numeric evaluations.', 'P08/P09 final reports; data handling rules',take='P09 used 5.874 GPU-hours; A6 was skipped, not replaced.')
S('I would discuss three next tests with Prof. Sun','Proposed options only; none of these experiments has run.',[
 ('Reference feasibility','Audit the shortened c2 flight and a physically consistent repair.','A changed reference would be a separately labeled variant with its own admission and provenance.'),
 ('Transfer the useful recipe','Test the start mixture on the admitted c3 variant.','This needs a new bounded experiment; A5 alone did not test this combination.'),
 ('Separate mechanisms','Compare matched budgets and more training seeds.','This would distinguish sampling, extra training and reward-weight effects before a stronger research claim.')],
 'My first question is whether we should prioritize repairing the reference’s flight or transferring the start mixture to c3. I would then compare matched training budgets with more seeds. I am presenting these as choices for discussion, not completed work or permission to run them.', 'P09 final report: untested options')
S('My Week 32 result: progress, but no full imitation pass','The partial results tell me where to work next.',[
 ('Commanded jump','Nominal 20/20; random timing and action delay remain weak.','The original command model remains the policy of record; neither fine-tune met promotion rules.'),
 ('Reference tracking','A4 completes 20/20, but full L1 remains 0/20.','Recovery and tracking improved while flight duration stayed below the fixed gate.'),
 ('Discussion question','Should I repair flight feasibility first, or test the mixture on c3?','The next experiment should target the measured defect rather than add unbounded training.')],
 'I can now separate three things clearly: a command policy that jumps nominally, a tracker that completes the clip, and the full skill contract that remains unmet. That separation is the useful result of the week.', 'P08 S5; P09 final/raw')
S('Evidence and reproducibility','Private working presentation · raw metrics frozen for this build',[
 ('Command-policy evidence','week_queue_2026-10-29/\nS1/rework_v2 · S3/raw · S4/raw · S5/bankB','These folders contain the corrected episode-level results behind the command and landing charts.'),
 ('Imitation evidence','p09_s6_2026-10-29/\nA1–A5/raw · manifests · parity · train_status','These are the original c2 rows and separately labeled c3 variant. No confirmation bank was used because nothing passed.'),
 ('Companion files','Speaker script · box-by-box HTML\nverified_metrics.json · evidence_manifest.json','The companion explains each text box. Hashes point to the exact local evidence used; the builder reruns only saved-data aggregation.')],
 'The supporting files preserve the source of every result slide. I can reproduce the charts without new training or simulation. The presentation notes and box-by-box companion contain the definitions that would be too dense for the main slides.', 'P08/P09 saved artifacts; snapshot recorded in README')
# Context-first revision requested by the owner: keep the evidence, rebuild the narrative.
old={x['title']:x for x in slides};slides=[]
def keep(title):slides.append(old[title])
S('Week 32 | Connecting running and jumping','Shaoqin Li · Applied and Embodied AI Lab · CCNY\nOctober 31, 2026 · Background → goal → experiments → measured results',[
 ('Starting point','Week 31: separate movement skills in simulation.','Walking, jumping jacks and limited motion switching existed, but the complete continuous running-jump milestone had not been established.'),
 ('This week’s question','Can the R1 run, jump, land and keep running?','The desired behavior is one continuous physics-controlled sequence. It is not an edited sequence of separate clips.'),
 ('What I can show today','Verified measurements and training curves; no new Week 32 rollout video.','Historical Week 31 images are labeled explicitly. Week 32 figures are numerical evidence, not robot-motion images.')],
 'I will start with the project goal and what I had last week, because the meaning of this week’s results depends on that context. I am building a set of movement skills for the R1. Last week I had separate pieces. This week I worked on a continuous run, jump, landing and return to running. All the work is in simulation. The workstation handoff contains measurements and training logs, but no new rendered rollout video.', 'Week 31 script; Week 32 P04/P08/P09 reports')
S('The project goal is a reusable bag of R1 movement skills','A skill should be controllable and connect to the next skill.',[
 ('Robot and environment','Unitree R1 humanoid, modeled in MuJoCo physics.','MuJoCo computes contact, gravity and movement. The actor must balance the body by commanding joints; this is not a prerecorded animation.'),
 ('Long-term use','Move through a simulated scene: stand, walk, run, jump and perform actions.','The skill-bag idea is to combine reusable behaviors for an interactive game or scene rather than train one long fixed performance.'),
 ('Why connections matter','A policy must recover into a usable state for the next action.','A good-looking jump is insufficient if the robot falls after landing or cannot resume locomotion.')],
 'My long-term idea is a bag of skills that I can use in one simulated scene. The R1 should follow movement commands and perform actions while staying balanced. That makes transitions important: I need to know not only whether it can jump, but whether it can run into the jump and keep moving afterward. Nothing in this presentation is a physical-robot demonstration.', 'Week 31 slides 1–2 and 22–23')
S('Week 31 established three separate pieces','These were useful starting points, not the completed Week 32 milestone.',[
 ('Speed-controlled locomotion','One policy could stand, walk and move faster when I changed its speed command.','The fast gait was still a fast walk: brief time with both feet airborne did not meet the later true-running contract.'),
 ('Video-based jumping jacks','A separate tracking policy followed a retargeted jumping-jacks motion.','This demonstrated a simulated jump-like skill with flight phases, primarily around one place; it was not a forward running jump.'),
 ('A first switching test','I tested handovers between tennis and dance policies.','Some direct switches worked; difficult poses and standing-pose handovers exposed limitations. This did not establish a general skill-composition system.')],
 'Week 31 had three parts. First, I trained a speed-command locomotion policy. Second, I obtained a jumping-jacks tracker through the human-video pipeline. Third, I tried switching between two other motion policies. Those results showed useful individual skills, but they did not prove that one robot state could flow through running, a forward jump, landing and running again.', 'Week 31 slides 5–8, 10–12, 18–22')
S('These images show the Week 31 starting point','HISTORICAL WEEK 31 IMAGES · NOT evidence of the new Week 32 policies',[],
 'These are two historical frames from the previous presentation. The left is the delay-trained locomotion policy, and the right is the jumping-jacks comparison between a human video and the simulated R1. I am showing them only to make the starting point concrete. They are not images of the new command-jump or imitation policies evaluated this week.', 'Week 31 poster_walk_run_delay.png; poster_jumping_jacks.png',take='Week 31 showed separate skills; Week 32 asked whether running and jumping could form one sequence.')
slides[-1]['pictures']=[{'file':'week31_historical_walk.png','label':'WEEK 31 · Speed-command locomotion'},{'file':'week31_historical_jacks.png','label':'WEEK 31 · Human video and jumping-jacks tracker'}]
S('Week 31 left three specific gaps','These gaps explain the experiment order in Week 32.',[
 ('Fast walk ≠ true run','Speed alone did not establish a running gait.','The later running test requires repeated brief flight phases and alternating support, not just a high commanded speed.'),
 ('Jumping jacks ≠ running jump','A jump in place did not establish approach, forward travel and recovery.','A continuous running jump needs the skills to work while the body is already moving and after the landing.'),
 ('Working once ≠ reliable skill','Delay and difficult transitions could break a policy.','Repeated seeded evaluations, disturbed conditions and defined failure criteria are needed before a stronger claim.')],
 'The important distinction is between the names of the skills and the measured behavior. A policy going fast can still be walking. Jumping jacks can leave the ground without solving a running jump. And one successful rollout does not tell me how the skill handles different timing or delays. Those were the gaps I turned into explicit tests.', 'Week 31 slides 12, 14–16, 20, 22')
S('Week 32 targeted one continuous running-jump sequence','I tested the whole sequence, not just whether both feet left the floor.',[],
 'This diagram is the goal, not a rollout image. The policy should establish a run, take off into a larger jump, travel forward in flight, land, and return to running. The simulator is not reset between phases. I also need to distinguish the large commanded jump from the short flight phases that occur naturally while running.', 'P04 fallback specification; P08 contract','week32_goal.png','The goal includes the approach and recovery, not only the airborne moment.')
S('I split the week into a prerequisite and two routes','This keeps the command-policy result separate from the imitation result.',[
 ('First · establish true running','I changed the fast-walk task and checked a genuine flight-based running test.','This created a running controller to build on, before adding a jump request.'),
 ('Route A · commanded jump','I taught the controller to respond to an explicit jump request.','This route does not copy a human running-jump reference. Its rewards and command signal specify the behavior.'),
 ('Route B · motion imitation','I repaired a human running-jump reference and trained a separate tracker.','This route tests whether the R1 can follow the reference under physics. It has different inputs and a separate success test.')],
 'There are two routes in the results, not one policy that changed its purpose halfway through. I first needed a true-running controller. One route extended that controller with a jump command. In parallel, the other route prepared a running-jump motion reference for a separate imitation tracker. Success in the command route does not make the imitation route a success.', 'P04 README/JUMP_STAGE; P02; P08/P09 plans')
S('Four terms make the rest of the talk easier to follow','A learned controller is different from the motion it is trying to follow.',[
 ('Policy','The trained controller: robot observations in, joint commands out.','The policy runs repeatedly while physics advances; it must react to the body state, not merely play recorded joint angles.'),
 ('Reference','A desired motion trajectory, expressed for the R1 body.','A reference tells the tracker where the body and joints should be over time. It is a target, not proof that a robot can physically execute it.'),
 ('Training versus evaluation','Training updates the policy; evaluation freezes it and measures behavior.','A reward is a numerical training incentive. An evaluation gate is a fixed success requirement; a high reward or long episode does not override a failed gate.')],
 'A policy is the actual learned controller. A reference is a target motion. Training changes the controller to pursue its rewards or reference, while evaluation freezes the controller and applies a fixed test. This is why I separate the learning curves from the final success counts. The terms are related but not interchangeable.', 'P04/P09 task and evaluation contracts')
keep('I kept the two policy families separate')
slides[-1]['boxes'][0]['title']='Route A · command policy';slides[-1]['boxes'][1]['title']='Route B · imitation policy'
keep('I froze the evaluation before choosing a model')
S('Before jumping, I trained and verified a true running gait','Week 32 prerequisite · speed-command fine-tune · 20 simulated trials',[
 ('What I changed','I adjusted the running gait schedule and rewarded real flight at faster commands.','The parent was the Week 31 speed-command policy, not a new human-motion tracker. A second fine-tune increased foot-clearance margin.'),
 ('What the running test required','Forward speed, repeated flight phases, alternating steps and no fall.','Both feet must be off the ground under force-and-height checks. Short windows throughout the run must satisfy the gait requirements.'),
 ('Measured result','20/20 passed at a 2.0 m/s command; about 16.7% airborne time.','The selected running parent averaged about 1.95 m/s in this saved evaluation. It was then used to initialize the commanded-jump task.')],
 'This is a step missing from my earlier summary. I did not jump directly from Week 31’s fast walk to the final jump result. I first fine-tuned that locomotion policy to produce a genuine running gait and checked it under a flight-based contract. The selected second version passed all twenty steady-running trials at a two-meter-per-second command, with about sixteen-point-seven percent airborne time. This was a running result, not yet a running-jump result.', 'P04 raw/runftv2_model_14498_vx2.json; JUMP_STAGE §1')
S('I then gave that running controller a jump request','A countdown signal told the controller when a larger jump was requested.',[
 ('Before the request','Keep running at the requested forward speed.','The policy already had a running behavior; the jump training was initialized from that controller.'),
 ('At the request','A short countdown input activated jump-specific rewards.','The new input changed the observation contract from 83 to 84 values. The policy still controlled the same robot joints.'),
 ('After the landing','Return to the running task without resetting the simulator.','The test measures recovery speed and true running after landing. It does not count a jump followed by a fall as success.')],
 'For the command route I appended one jump-countdown input to the running controller. During a short requested window, the rewards encouraged the larger flight and height; outside the window, the normal running task remained. This makes the requested jump distinct from the small flight phases in an ordinary run. I tested the full before-and-after sequence in one physics rollout.', 'P04 JUMP_STAGE §2; jump task self-tests')
S('My first commanded jump was too flat, so I fixed the target','I aligned the training reward with the height measured by the test.',[
 ('First attempt','0/20 full passes; mean rise after takeoff was about 3.0 cm.','Other gates passed, but the jump-height requirement was at least 5 cm. The reward had measured pelvis rise from the command, counting some push-off height.'),
 ('Correction','Reward whole-body rise above the takeoff point, then continue training.','I changed the rewarded height definition and its weight, retaining the evaluation gate. This was an engineering correction with extra training, not an isolated ablation.'),
 ('Corrected result','20/20 nominal passes; mean rise about 10.1 cm.','The height increased, but the mean landing peak rose to about 8.8 body weights. That cost motivated the landing experiment shown next.')],
 'The first jump policy could make a long, flat leap but missed the height requirement. I found that its reward measured pelvis rise from the time of the request, while the test measured whole-body rise after takeoff. Those are different quantities. I changed the reward definition and weight, trained further and reported the fixed final checkpoint. It passed twenty of twenty nominal trials, but the stronger jump also made landing forces larger.', 'P04 raw/jump_model_17497_jump_vx2.json; jumpapex_model_20496_jump_vx2.json')
for title in ['I reproduced the nominal commanded jump','I found a clear action-delay boundary','I reduced landing peaks, but missed the target','I gained delay-tolerant running and lost the jump']:keep(title)
S('For imitation, I changed the source of the motion','Week 31 used human videos; this Week 32 branch used recorded human motion.',[
 ('Week 31 pipeline','Human video → estimated human pose → R1 target → trained tracker.','A video reconstruction stage such as GVHMR estimates body movement before GMR maps it to the R1.'),
 ('Week 32 source','LAFAN1 recorded running-jump motions → R1 retargeting and repair.','This branch did not obtain its references from a new camera-video reconstruction. Starting from recorded human motion removes that particular estimation step, not all reference problems.'),
 ('Reference names','c2: original running-jump target. c3: separately tested alternative.','c2 source flight is 0.30 s, with a 0.24 s physical gate. c3 source flight is 0.40 s, with its own 0.32 s gate. They are not interchangeable tests.')],
 'The imitation route continues the idea of teaching motions from human data, but this week the source was different. Week 31 reconstructed movement from ordinary video. This running-jump branch used LAFAN1 recorded human motion, then retargeted and repaired it for the R1. I call the two chosen references c2 and c3 in the logs. These are just candidate names, not separate controller architectures. The alternative c3 keeps its own source-anchored test.', 'Week 31 slides 18–21; P02 README; P09 A5 admission')
keep('Then I built the imitation test around a repaired reference')
S('A complete imitation pass required four things together','At least 16 of 20 episodes must pass every requirement, not merely finish the clip.',[
 ('1 · Complete without a fall','Follow the full reference and avoid unintended body contact.','The no-fall gate combines completion with no non-foot contact. A policy can finish the clip but still fail this gate.'),
 ('2 · Flight and recovery','Meet the required airborne duration, then regain forward speed.','Original c2 requires at least 0.24 seconds of physical flight and recovery to about 2.31 m/s in the defined post-landing window.'),
 ('3 · Track the joint motion','Keep mean and peak-percentile joint errors within the fixed limits.','The mean RMS limit is 0.15 radians, about 9 degrees; the 95th-percentile limit is 0.30 radians, about 17 degrees. These measure joint mismatch, not a fall.')],
 'The logs call this the L1 gate. I use full test in the slides. It is a conjunction: finishing without a fall or unintended body contact, enough physical flight, adequate recovery speed, and sufficiently accurate joint tracking. For the original c2 reference, physical flight must last at least 0.24 seconds. Mean joint RMS must stay at or below 0.15 radians, roughly nine degrees, with the higher-percentile bound also met. A row needs sixteen full passes out of twenty. The command route uses a different test, so its 0.12-second jump threshold must not be substituted here.', 'P09 unchanged s6_eval.py; P03/P09 gate text')
S('I tested three explanations for the early tracking failure','A1, A2 and A3 are attempt names in the logs, not three different robots.',[
 ('A1 · too-strict early termination?','Temporarily widen the hands/feet tracking tolerance, then restore it.','The hypothesis was that early stopping prevented useful learning. Final evaluation always kept the original strict tolerance.'),
 ('A2 · start-phase exposure?','Guarantee starts at the beginning, approach and jump transition.','The hypothesis was that the original adaptive sampler concentrated on easier motion segments. Evaluation still started from frame zero.'),
 ('A3 · too little speed incentive?','Double the existing velocity-tracking reward weight.','This kept the original sampler and tolerance. It tested whether the available velocity signal needed more emphasis; no reference or evaluation gate changed.')],
 'The original pilot ended before reaching the intended jump. I did not treat that as proof that the jump was physically impossible. I tested three learning explanations separately. A1 widened and then restored a tracking tolerance. A2 changed which reference phases training episodes began from. A3 increased an existing speed-tracking reward. Each was a fresh single-seed training attempt, with its one intended scientific change plus read-only diagnostics.', 'P09 A1–A3 manifests; D0 diagnosis')
keep('I changed where training episodes started')
keep('Five attempts finished; none passed the full L1 gate')
slides[-1]['sub']='A1: tolerance · A2: start phases · A3: speed reward · A4: combined · A5: alternate reference'
for title in ['I monitored the learning curves, not only the final score','I also tracked error during training','A1 shortened again when I restored strict tolerance','I saw progress in recovery and joint tracking','Flight duration remained the binding common failure','I kept the extra-training confound visible','I tried one other reference and kept one slot unused']:keep(title)
S('The failed attempts also narrowed the next question','These are observations from this sweep, not universal conclusions.',[
 ('A1 · temporary relaxation','Episodes shortened again at strict tolerance; final full test 0/20.','A wider training tolerance did not produce a passing strict policy in this one run. The missing native A1 RMS is not filled in.'),
 ('A3 · more velocity reward alone','No episode completed the clip; final full test 0/20.','Increasing velocity reward under the original sampler did not fix the early failure in this run. That does not prove velocity reward is irrelevant in every setup.'),
 ('A5 · different reference alone','The c3 variant also failed before the jump; full test 0/20.','The alternate reference used the original sampler. The sweep did not test c3 together with A2’s phase mixture, so that remains an untested combination.')],
 'The negative results help locate what still needs testing. Simply relaxing termination and returning to strict settings was not enough. Simply increasing the speed reward was not enough. A different reference under the original configuration also failed early. The phase-mixture attempts made the largest observed progress on c2, but more seeds and matched budgets are needed before I turn that observation into a general causal claim.', 'P09 A1/A3/A5 raw; final report')
for title in ['I stopped where the evidence stopped','I would discuss three next tests with Prof. Sun','My Week 32 result: progress, but no full imitation pass','Evidence and reproducibility']:keep(title)
# Translate shorthand on visible slides; keep original source filenames intact.
for item in slides:
 for key in ['title','sub','take']:
  item[key]=item[key].replace('full L1','full test').replace('Full L1','Full test').replace('L1 target','Full-test target').replace('frame-zero','frame-zero')
 for box in item['boxes']:
  for key in ['title','text']:box[key]=box[key].replace('full L1','full test').replace('Full L1','Full test').replace('L1 target','Full-test target')
# Shorten a title whose meaning needs an explicit subject.
for item in slides:
 if item['title']=='A1 shortened again when I restored strict tolerance':item['title']='A1 episodes shortened as I restored strict tolerance'
 item['notes']+='\n\nThe first slides explain the Week 31 background; all Week 32 counts refer to simulation. Historical pictures are explicitly labeled and do not substitute for current rollout media.' if item.get('pictures') else ''


# Build PowerPoint.
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
def rgb(s):return RGBColor.from_string(s.lstrip('#'))
def tb(slide,text,x,y,w,h,size=20,color=INK,bold=False):
 sh=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h));tf=sh.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(.02);tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(text.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.space_after=Pt(9)
  for r in p.runs:r.font.name='Aptos';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=rgb(color)
 return sh
for i,s in enumerate(slides,1):
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(BG)
 band=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width, Inches(.11));band.fill.solid();band.fill.fore_color.rgb=rgb(TEAL);band.line.fill.background()
 tb(sl,s['title'],.55,.42,12.2,.85,29,NAVY,True);tb(sl,s['sub'],.57,1.29,12.1,.76,16,GRAY)
 if s.get('pictures'):
  for j,photo in enumerate(s['pictures']):
   p=AS/photo['file'];iw,ih=Image.open(p).size;scale=min(5.8/iw,3.65/ih);w,h=iw*scale,ih*scale;x=.58+j*6.25
   sl.shapes.add_picture(str(p),Inches(x+(5.8-w)/2),Inches(2.25),width=Inches(w),height=Inches(h))
   tb(sl,photo['label'],x,6.02,5.8,.32,13,TEAL,True)
 elif s['chart']:
  p=AS/s['chart'];iw,ih=Image.open(p).size;mw,mh=11.8,4.25;scale=min(mw/iw,mh/ih);w,h=iw*scale,ih*scale
  sl.shapes.add_picture(str(p),Inches((13.333-w)/2), Inches(2.05),width=Inches(w),height=Inches(h))
 else:
  boxes=s['boxes'];cols=2 if len(boxes)==4 else 3;ww=(12.2-(cols-1)*.22)/cols
  for j,b in enumerate(boxes):
   x=.57+(j%cols)*(ww+.22);y=2.35+(j//cols)*1.9;hh=1.7 if cols==2 else 3.35
   sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x), Inches(y), Inches(ww), Inches(hh));sh.fill.solid();sh.fill.fore_color.rgb=rgb('#E8EEF1');sh.line.fill.background()
   tb(sl,b['title'],x+.22,y+.23,ww-.44,.65,23,TEAL,True)
   tb(sl,b['text'],x+.22,y+.99,ww-.44,hh-1.1,19 if cols==3 else 17)
 if s['take']:tb(sl,s['take'],.6,6.38,12.1,.44,18,TEAL,True)
 tb(sl,'WEEK 32  ·  SIMULATION ONLY',.57,7.05,4.4,.22,10,GRAY);tb(sl,s['source'],4.7,7.05,7.4,.24,9,GRAY);tb(sl,f'{i:02d}',12.4,7.03,.4,.26,11,GRAY)
 sl.notes_slide.notes_text_frame.text=s['notes']+'\n\nEvidence: '+s['source']+'\n\n'+ '\n\n'.join(b['title']+': '+b['meaning'] for b in s['boxes'])
prs.save(HERE/'embodied ai week 32.pptx')
(HERE/'slide_content.json').write_text(json.dumps(slides,indent=2,ensure_ascii=False))
script=['# Week 32 speaker script','Shaoqin Li · October 31, 2026 · About 25–30 minutes, with optional detail in the companion. Simulation only.','']
for i,s in enumerate(slides,1):script +=[f'## Slide {i} — {s["title"]}','',s['notes'],'','Evidence: '+s['source'],'']
script+=['## Q&A preparation','','**Did the tracker pass?** No. A4 is 0/20 full L1. Sixteen meet all non-flight gates together; no full pass is inferred.','','**Did velocity weighting cause the gain?** Not established. A4 has extra training, resumed state and a new seed.','','**Why no new video?** This sweep produced numeric evaluations but no rendered rollout. No unrelated old video is substituted.','','**Does the 0.22-second reference make 0.24 seconds impossible?** No such proof. Geometric reference flight and physical force-based flight differ; this is a feasibility warning.','','**Which policy stays final?** model_20496 is the original command-policy fallback. A4 is only the best failed c2 tracking candidate, not a replacement.']
(HERE/'speaker_script_week32.md').write_text('\n'.join(script))
print('Saved',len(slides),'slides')
