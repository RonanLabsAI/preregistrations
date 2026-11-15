#!/usr/bin/env python3
"""Replay captured served EXL3 inputs and compare opt-in occupancy candidates."""
import argparse,ctypes as C,hashlib,json,math,pathlib,statistics,struct,time
p=argparse.ArgumentParser()
p.add_argument('--root',required=True);p.add_argument('--fatbin',required=True)
p.add_argument('--candidate-fatbin',required=True);p.add_argument('--rows',type=int,choices=[1,16],required=True)
p.add_argument('--mode',choices=['validation','ab'],required=True)
p.add_argument('--candidate',choices=['r96','t4r96','t2r64'],default='r96');p.add_argument('--out',required=True)
a=p.parse_args();assert hashlib.sha256(pathlib.Path(a.fatbin).read_bytes()).hexdigest()=='1dadf5ec0919f46c6a3d737044cbfe36889a9da9259ad17f3368060d0052afa5'
root=pathlib.Path(a.root)/f'c{a.rows}'
cu=C.CDLL('libcuda.so.1');ptr=C.c_void_p;u64=C.c_uint64;i32=C.c_int
# Explicit signatures avoid pointer truncation in the driver ABI.
sigs={
'cuInit':[C.c_uint],'cuDeviceGet':[C.POINTER(i32),i32],
'cuDevicePrimaryCtxRetain':[C.POINTER(ptr),i32],'cuCtxSetCurrent':[ptr],
'cuModuleLoad':[C.POINTER(ptr),C.c_char_p],'cuModuleGetFunction':[C.POINTER(ptr),ptr,C.c_char_p],
'cuMemAlloc_v2':[C.POINTER(u64),C.c_size_t],'cuMemcpyHtoD_v2':[u64,ptr,C.c_size_t],
'cuMemcpyDtoH_v2':[ptr,u64,C.c_size_t],'cuMemsetD8_v2':[u64,C.c_ubyte,C.c_size_t],
'cuStreamCreate':[C.POINTER(ptr),C.c_uint],'cuStreamSynchronize':[ptr],
'cuLaunchKernel':[ptr,*([C.c_uint]*7),ptr,ptr,ptr],
'cuEventCreate':[C.POINTER(ptr),C.c_uint],'cuEventRecord':[ptr,ptr],
'cuEventElapsedTime':[C.POINTER(C.c_float),ptr,ptr],
'cuStreamBeginCapture_v2':[ptr,i32],'cuStreamEndCapture':[ptr,C.POINTER(ptr)],
'cuGraphInstantiateWithFlags':[C.POINTER(ptr),ptr,u64],'cuGraphLaunch':[ptr,ptr],
'cuFuncGetAttribute':[C.POINTER(i32),i32,ptr],
'cuOccupancyMaxActiveBlocksPerMultiprocessor':[C.POINTER(i32),ptr,i32,C.c_size_t],
'cuMemGetInfo_v2':[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)],'cuMemsetD8Async':[u64,C.c_ubyte,C.c_size_t,ptr], 'cuMemAllocHost_v2':[C.POINTER(ptr),C.c_size_t], 'cuMemcpyHtoDAsync_v2':[u64,ptr,C.c_size_t,ptr]}
for n,s in sigs.items():getattr(cu,n).argtypes=s

def call(n,*args):
 r=getattr(cu,n)(*args)
 if r:raise RuntimeError(f'{n} CUDA error {r}')

def alloc(n,data=None):
 v=u64();call('cuMemAlloc_v2',C.byref(v),n)
 if data is None:call('cuMemsetD8_v2',v,0,n)
 else:
  b=C.create_string_buffer(data);assert len(data)==n;call('cuMemcpyHtoD_v2',v,b,n)
 return v.value

def put(fmt,values):return alloc(struct.calcsize(fmt),struct.pack(fmt,*values))

def pack(fields):
 b=bytearray()
 for typ,val in fields:
  n=struct.calcsize(typ);b.extend(b'\0'*((-len(b))%n));b.extend(struct.pack(typ,val))
 return C.create_string_buffer(bytes(b))

def fields(ps,ints,tail=()):return [('Q',x) for x in ps]+[('i',x) for x in ints]+list(tail)

def launch(name,grid,block,shared,buf):
 size=C.c_size_t(len(buf)-1)
 extra=(ptr*5)(ptr(1),C.cast(buf,ptr),ptr(2),C.cast(C.byref(size),ptr),ptr(0))
 call('cuLaunchKernel',ks[name],*grid,block,1,1,shared,stream,None,extra)


call('cuInit',0);dev=i32();call('cuDeviceGet',C.byref(dev),0);ctx=ptr();call('cuDevicePrimaryCtxRetain',C.byref(ctx),dev);call('cuCtxSetCurrent',ctx)
free=C.c_size_t();total=C.c_size_t();call('cuMemGetInfo_v2',C.byref(free),C.byref(total));assert free.value>=24*2**30
stream=ptr();call('cuStreamCreate',C.byref(stream),1)
mods=[];ks={};resources={}
for filename in [a.fatbin,a.candidate_fatbin]:
 mod=ptr();call('cuModuleLoad',C.byref(mod),filename.encode());mods.append(mod)
 names=['tf_exl3_group','tf_exl3_experts_rot_in','tf_exl3_grouped_1','tf_exl3_gateup_epilogue','tf_exl3_down_combine'] if len(mods)==1 else ['tf_exl3_grouped_1_w28_'+c for c in ['r96','t4r96','t2r64']]
 for name in names:
  f=ptr();call('cuModuleGetFunction',C.byref(f),mod,name.encode());ks[name]=f;attrs={}
  for attr,label in [(4,'registers_per_thread'),(1,'static_shared_bytes'),(3,'local_bytes')]:
   v=i32();call('cuFuncGetAttribute',C.byref(v),attr,f);attrs[label]=v.value
  if 'grouped' in name:
   b=i32();call('cuOccupancyMaxActiveBlocksPerMultiprocessor',C.byref(b),f,128,0);attrs['resident_blocks_per_sm']=b.value
  resources[name]=attrs
meta=[json.loads((root/f's{i}.json').read_text()) for i in range(6)];experts=meta[0]['fields'][6][1]
sizes={};weights={};tables={};mapping={};allocated=0
for s in meta:
 for b in filter(None,s['buffers']):
  old=s['fields'][b['arg']][1]
  if b['phase']=='weight':
   weights[(old,b['expert'])]=b;tables[old]=True
  else:sizes[old]=max(sizes.get(old,0),b['bytes'])
for old,size in sizes.items():mapping[old]=alloc(size);allocated+=size
for old in tables:
 values=[0]*experts
 for (table,e),b in weights.items():
  if table==old:
   data=(root/b['file']).read_bytes();assert len(data)==b['bytes'];values[e]=alloc(len(data),data);allocated+=len(data)
 mapping[old]=put('Q'*experts,values);allocated+=experts*8
# The down kernel receives the same table for both projections.
for s in meta:
 for b in filter(None,s['buffers']):
  if b['phase']=='pre':
   data=(root/b['file']).read_bytes();host=C.create_string_buffer(data);call('cuMemcpyHtoD_v2',mapping[s['fields'][b['arg']][1]],host,len(data))
assert allocated+256*2**20<4*2**30
scrub=alloc(256*2**20)
warm_inputs=[]
for stage,arg in [(0,0),(1,0)]:
 s=meta[stage];b=next(b for b in s['buffers'] if b and b['phase']=='pre' and b['arg']==arg)
 data=(root/b['file']).read_bytes();host=ptr();call('cuMemAllocHost_v2',C.byref(host),len(data));C.memmove(host,data,len(data))
 warm_inputs.append((mapping[s['fields'][arg][1]],host,len(data)))
def prepare(repeat):
 call('cuMemsetD8Async',scrub,repeat%256,256*2**20,stream)
 for dest,host,size in warm_inputs:call('cuMemcpyHtoDAsync_v2',dest,host,size,stream)
seq=[]
for s in meta:
 fs=[(typ,mapping[v] if typ=='Q' and v else v) for typ,v in s['fields']]
 seq.append((s['name'],tuple(s['grid']),s['block'],s['shared'],pack(fs)))
picks=struct.unpack('i'*(a.rows*8),(root/'s0-a0-pre.bin').read_bytes());unique=len(set(picks));numerics=[]
def read(address,size):
 host=C.create_string_buffer(size);call('cuMemcpyDtoH_v2',host,address,size);return host.raw

def verify_sequence(steps,label):
 for i,step in enumerate(steps):
  launch(*step);call('cuStreamSynchronize',stream)
  for b in filter(None,meta[i]['buffers']):
   if b['phase']!='post':continue
   want=(root/b['file']).read_bytes();got=read(mapping[meta[i]['fields'][b['arg']][1]],b['bytes'])
   if i==0 and b['arg']==1:want,got=want[:unique*4],got[:unique*4]
   if i==0 and b['arg']==3:want,got=want[:unique*a.rows*4],got[:unique*a.rows*4]
   assert got==want,f'{label} stage {i} arg {b["arg"]}: captured byte mismatch'
 output_arg=5 if meta[5]['fields'][5][1] else 3
 output_rows=a.rows if output_arg==5 else a.rows*8
 output=read(mapping[meta[5]['fields'][output_arg][1]],output_rows*4096*4)
 assert all(math.isfinite(v) for v in struct.unpack('f'*(output_rows*4096),output))
 return hashlib.sha256(output).hexdigest()
base_hash=verify_sequence(seq,'champion')
variant=list(seq)
for i in [2,4]:
 name,grid,block,shared,buf=variant[i];tile=dict(r96=8,t4r96=4,t2r64=2)[a.candidate]
 variant[i]=(name+'_w28_'+a.candidate,(grid[0],grid[1]*8//tile,grid[2]),block,shared,buf)
if a.mode=='ab':
 for repeat in range(5):
  assert verify_sequence(variant,a.candidate)==base_hash
 numerics=[dict(candidate=a.candidate,bitwise_all_partials_and_stages=True,repeat_runs=5,finite=True,output_sha256=base_hash)]
call('cuProfilerStart')
blocks=[]
if a.mode=='validation':
 graph=ptr();call('cuStreamBeginCapture_v2',stream,0)
 for step in seq:launch(*step)
 call('cuStreamEndCapture',stream,C.byref(graph));exe=ptr();call('cuGraphInstantiateWithFlags',C.byref(exe),graph,0)
 for repeat in range(6):
  for j in range(20):
   prepare(repeat);call('cuGraphLaunch',exe,stream);call('cuStreamSynchronize',stream)
else:
 executables={}
 for arm,steps in [('champion',seq),(a.candidate,variant)]:
  graph=ptr();call('cuStreamBeginCapture_v2',stream,0)
  for step in steps:launch(*step)
  call('cuStreamEndCapture',stream,C.byref(graph));exe=ptr();call('cuGraphInstantiateWithFlags',C.byref(exe),graph,0);executables[arm]=exe
 events=[ptr(),ptr()]
 for event in events:call('cuEventCreate',C.byref(event),0)
 for repeat in range(-1,10):
  arms={}
  order=['champion',a.candidate] if repeat%2==0 else [a.candidate,'champion']
  for arm in order:
   times=[];steps=seq if arm=='champion' else variant
   for j in range(20):
    prepare(repeat)
    call('cuEventRecord',events[0],stream)
    call('cuGraphLaunch',executables[arm],stream)
    call('cuEventRecord',events[1],stream);call('cuStreamSynchronize',stream)
    elapsed=C.c_float();call('cuEventElapsedTime',C.byref(elapsed),*events);times.append(elapsed.value*1000)
   arms[arm]=times
  blocks.append(dict(repeat=repeat,order=order,us=arms))
call('cuProfilerStop')
result=dict(rows=a.rows,unique=unique,route_provenance='exact served captured sequence',mode=a.mode,candidate=a.candidate,allocated_bytes=allocated+256*2**20,resources=resources,numerics=numerics,champion_captured_bitwise=True,output_sha256=base_hash,blocks=blocks,source_hashes={str(i):hashlib.sha256((root/f's{i}.json').read_bytes()).hexdigest() for i in range(6)})
pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','resources','source_hashes']}),flush=True)
