#!/usr/bin/env python3
"""Replay original EXL3 fatbin at frozen shapes; synthetic routes are labeled.
No model, router or daily lifecycle operations. Uses only the selected CUDA device.
"""
import argparse,ctypes as C,hashlib,json,math,pathlib,statistics,struct,time
FATBIN_SHA='1dadf5ec0919f46c6a3d737044cbfe36889a9da9259ad17f3368060d0052afa5'
p=argparse.ArgumentParser();p.add_argument('--fatbin',required=True);p.add_argument('--rows',type=int,choices=[1,16],required=True);p.add_argument('--unique',type=int,required=True);p.add_argument('--counter',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
assert hashlib.sha256(pathlib.Path(a.fatbin).read_bytes()).hexdigest()==FATBIN_SHA
assert 8<=a.unique<=a.rows*8 and a.unique%8==0
D=4096;N=2048;E=256;P=a.rows*8;U=a.unique
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
'cuMemGetInfo_v2':[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]}
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
free=C.c_size_t();total=C.c_size_t();call('cuMemGetInfo_v2',C.byref(free),C.byref(total));assert free.value>=24*2**30, 'requires 4 GiB cap plus 16 GiB reserve and 4 GiB replay reserve'
mod=ptr();call('cuModuleLoad',C.byref(mod),a.fatbin.encode());stream=ptr();call('cuStreamCreate',C.byref(stream),1)
names=['tf_exl3_group','tf_exl3_experts_rot_in','tf_exl3_grouped_1','tf_exl3_gateup_epilogue','tf_exl3_down_combine'];ks={};resources={}
for name in names:
 f=ptr();call('cuModuleGetFunction',C.byref(f),mod,name.encode());ks[name]=f
 attrs={}
 for attr,label in [(4,'registers_per_thread'),(1,'static_shared_bytes'),(3,'local_bytes')]:
  v=i32();call('cuFuncGetAttribute',C.byref(v),attr,f);attrs[label]=v.value
 b=i32();call('cuOccupancyMaxActiveBlocksPerMultiprocessor',C.byref(b),f,1024 if name==names[0] else 128 if name==names[2] else 256 if name==names[4] else 32,P*4 if name==names[0] else 0);attrs['resident_blocks_per_sm']=b.value;resources[name]=attrs
# A deterministic bounded MCG bitstream, seed 1234, no model weights.
word=1234;chunk=bytearray()
for j in range(262144):
 word=(1664525*word+1013904223)&0xffffffff;chunk.extend(struct.pack('I',word))
mat_bytes=D*N//2
weights=[]
for mat in range(3):
 base=alloc(U*mat_bytes)
 buf=C.create_string_buffer(bytes(chunk))
 for at in range(0,U*mat_bytes,len(chunk)):call('cuMemcpyHtoD_v2',base+at,buf,len(chunk))
 weights.append(put('Q'*E,[base+(e%U)*mat_bytes for e in range(E)]))
k2=put('i'*E,[8]*E);scalesD=alloc(E*D*2,b'\x00\x3c'*(E*D));scalesN=alloc(E*N*2,b'\x00\x3c'*(E*N))
x=alloc(a.rows*D*2,b'\x80\x3b'*(a.rows*D)) # finite BF16 1/256
picks=[(r*8+s)%U for r in range(a.rows) for s in range(8)];pick=put('i'*P,picks)
uids=alloc(P*4);count=alloc(4);members=alloc(P*a.rows*4)
xg=alloc(P*D*2);xu=alloc(P*D*2);xd=alloc(P*N*2);z=alloc(8*P*N*4);y=alloc(P*D*4);out=alloc(a.rows*D*4);wts=put('f'*P,[0.125]*P)
# Cache scrub is excluded from timing, to avoid repeated small-weight residency.
scrub=alloc(256*2**20)
seq=[
(names[0],(1,1,1),1024,P*4,pack(fields([pick,uids,count,members],[a.rows,8,E,a.rows]))),
(names[1],(P,D//128,2),32,0,pack(fields([x],[D,1])+fields([pick,scalesD,scalesD,xg,xu],[D,8,E]))),
(names[2],(P,N//128,8),128,0,pack(fields([xg,xu,weights[0],weights[1],k2,k2,uids,count,members,z],[D,N,P,4,a.rows,8,P,2,0,0,-1]))),
(names[3],(P,N//128,1),32,0,pack(fields([z,pick,scalesN,scalesN,scalesN,xd],[P,N,4,E],[('f',0.0),('i',0)]))),
(names[2],(P,D//128,1),128,0,pack(fields([xd,xd,weights[2],weights[2],k2,k2,uids,count,members,z],[N,D,P,1,a.rows,8,P,1,0,0,-1]))),
(names[4],(a.rows,D//128,1),256,0,pack(fields([z,pick,scalesD,y,wts,out],[P,D,1,E,8]))) ]
for step in seq:launch(*step)
call('cuStreamSynchronize',stream)
# Verify routing and that every output is finite; repeatability is checked after graph replay.
h=C.create_string_buffer(a.rows*D*4);call('cuMemcpyDtoH_v2',h,out,len(h));values=struct.unpack('f'*(a.rows*D),h.raw);assert all(math.isfinite(v) for v in values)
first_hash=hashlib.sha256(h.raw).hexdigest()
uh=C.create_string_buffer(4);call('cuMemcpyDtoH_v2',uh,count,4);assert struct.unpack('i',uh.raw)[0]==U
if a.counter:
 call('cuProfilerStart')
 for step in seq:launch(*step)
 call('cuStreamSynchronize',stream);call('cuProfilerStop');samples=[]
else:
 graph=ptr();call('cuStreamBeginCapture_v2',stream,0)
 for step in seq:launch(*step)
 call('cuStreamEndCapture',stream,C.byref(graph));exe=ptr();call('cuGraphInstantiateWithFlags',C.byref(exe),graph,0)
 # Kernel start/end timestamps come from Nsight Systems, as in served traces.
 samples=[]
 for repeat in range(6):
  for j in range(20):
   call('cuMemsetD8_v2',scrub,repeat,256*2**20);call('cuGraphLaunch',exe,stream);call('cuStreamSynchronize',stream)
call('cuMemcpyDtoH_v2',h,out,len(h));assert hashlib.sha256(h.raw).hexdigest()==first_hash
r=dict(rows=a.rows,unique=U,route_provenance='C1 distinct-membership equivalent; synthetic weights and values' if a.rows==1 else 'synthetic balanced collision sensitivity; NOT captured C16 routing',seed=1234,fatbin_sha256=FATBIN_SHA,resources=resources,steps=[dict(name=n,grid=g,block=b,shared=s) for n,g,b,s,_ in seq],samples_us=samples,finite=True,repeatable=True,output_sha256=first_hash,wall_seconds=time.monotonic())
pathlib.Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(rows=a.rows,unique=U,samples_us=samples,finite=True)),flush=True)
