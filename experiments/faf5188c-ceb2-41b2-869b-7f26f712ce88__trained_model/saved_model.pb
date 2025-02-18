��2
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.0-dev202210222v1.12.1-83490-gb1d4c35fba28��+
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_2/kernel
�
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_2/kernel
�
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
5Adam/v/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/v/transformer_encoder/layer_normalization_1/beta
�
IAdam/v/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp5Adam/v/transformer_encoder/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
5Adam/m/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/m/transformer_encoder/layer_normalization_1/beta
�
IAdam/m/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp5Adam/m/transformer_encoder/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
6Adam/v/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86Adam/v/transformer_encoder/layer_normalization_1/gamma
�
JAdam/v/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp6Adam/v/transformer_encoder/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
6Adam/m/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86Adam/m/transformer_encoder/layer_normalization_1/gamma
�
JAdam/m/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp6Adam/m/transformer_encoder/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
3Adam/v/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/v/transformer_encoder/layer_normalization/beta
�
GAdam/v/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp3Adam/v/transformer_encoder/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
3Adam/m/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/m/transformer_encoder/layer_normalization/beta
�
GAdam/m/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp3Adam/m/transformer_encoder/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
4Adam/v/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/v/transformer_encoder/layer_normalization/gamma
�
HAdam/v/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp4Adam/v/transformer_encoder/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
�
4Adam/m/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/m/transformer_encoder/layer_normalization/gamma
�
HAdam/m/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp4Adam/m/transformer_encoder/layer_normalization/gamma*
_output_shapes	
:�*
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	�*
dtype0
�
EAdam/v/transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*V
shared_nameGEAdam/v/transformer_encoder/multi_head_attention/attention_output/bias
�
YAdam/v/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpEAdam/v/transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
EAdam/m/transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*V
shared_nameGEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias
�
YAdam/m/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
GAdam/v/transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*X
shared_nameIGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel
�
[Adam/v/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel*$
_output_shapes
:��*
dtype0
�
GAdam/m/transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*X
shared_nameIGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel
�
[Adam/m/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel*$
_output_shapes
:��*
dtype0
�
:Adam/v/transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/value/bias
�
NAdam/v/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/value/bias*
_output_shapes
:	�*
dtype0
�
:Adam/m/transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/value/bias
�
NAdam/m/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/value/bias*
_output_shapes
:	�*
dtype0
�
<Adam/v/transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*M
shared_name><Adam/v/transformer_encoder/multi_head_attention/value/kernel
�
PAdam/v/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp<Adam/v/transformer_encoder/multi_head_attention/value/kernel*$
_output_shapes
:��*
dtype0
�
<Adam/m/transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*M
shared_name><Adam/m/transformer_encoder/multi_head_attention/value/kernel
�
PAdam/m/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp<Adam/m/transformer_encoder/multi_head_attention/value/kernel*$
_output_shapes
:��*
dtype0
�
8Adam/v/transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*I
shared_name:8Adam/v/transformer_encoder/multi_head_attention/key/bias
�
LAdam/v/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp8Adam/v/transformer_encoder/multi_head_attention/key/bias*
_output_shapes
:	�*
dtype0
�
8Adam/m/transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*I
shared_name:8Adam/m/transformer_encoder/multi_head_attention/key/bias
�
LAdam/m/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp8Adam/m/transformer_encoder/multi_head_attention/key/bias*
_output_shapes
:	�*
dtype0
�
:Adam/v/transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/key/kernel
�
NAdam/v/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/key/kernel*$
_output_shapes
:��*
dtype0
�
:Adam/m/transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/key/kernel
�
NAdam/m/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/key/kernel*$
_output_shapes
:��*
dtype0
�
:Adam/v/transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/query/bias
�
NAdam/v/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/query/bias*
_output_shapes
:	�*
dtype0
�
:Adam/m/transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/query/bias
�
NAdam/m/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/query/bias*
_output_shapes
:	�*
dtype0
�
<Adam/v/transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*M
shared_name><Adam/v/transformer_encoder/multi_head_attention/query/kernel
�
PAdam/v/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp<Adam/v/transformer_encoder/multi_head_attention/query/kernel*$
_output_shapes
:��*
dtype0
�
<Adam/m/transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*M
shared_name><Adam/m/transformer_encoder/multi_head_attention/query/kernel
�
PAdam/m/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp<Adam/m/transformer_encoder/multi_head_attention/query/kernel*$
_output_shapes
:��*
dtype0
�
Adam/v/et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/et_time_dist_conv2d/bias
�
3Adam/v/et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/et_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
Adam/m/et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/et_time_dist_conv2d/bias
�
3Adam/m/et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/et_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
!Adam/v/et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**2
shared_name#!Adam/v/et_time_dist_conv2d/kernel
�
5Adam/v/et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/et_time_dist_conv2d/kernel*&
_output_shapes
:**
dtype0
�
!Adam/m/et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**2
shared_name#!Adam/m/et_time_dist_conv2d/kernel
�
5Adam/m/et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/et_time_dist_conv2d/kernel*&
_output_shapes
:**
dtype0
�
 Adam/v/swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/swe_time_dist_conv2d/bias
�
4Adam/v/swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/v/swe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
 Adam/m/swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/swe_time_dist_conv2d/bias
�
4Adam/m/swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/m/swe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
"Adam/v/swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:{�*3
shared_name$"Adam/v/swe_time_dist_conv2d/kernel
�
6Adam/v/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/swe_time_dist_conv2d/kernel*'
_output_shapes
:{�*
dtype0
�
"Adam/m/swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:{�*3
shared_name$"Adam/m/swe_time_dist_conv2d/kernel
�
6Adam/m/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/swe_time_dist_conv2d/kernel*'
_output_shapes
:{�*
dtype0
�
#Adam/v/precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/precip_time_dist_conv2d/bias
�
7Adam/v/precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/v/precip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
#Adam/m/precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/precip_time_dist_conv2d/bias
�
7Adam/m/precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/m/precip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
%Adam/v/precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/precip_time_dist_conv2d/kernel
�
9Adam/v/precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/v/precip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
%Adam/m/precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/precip_time_dist_conv2d/kernel
�
9Adam/m/precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/m/precip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
!Adam/v/temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/temp_time_dist_conv2d/bias
�
5Adam/v/temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp!Adam/v/temp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
!Adam/m/temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/temp_time_dist_conv2d/bias
�
5Adam/m/temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp!Adam/m/temp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
#Adam/v/temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/temp_time_dist_conv2d/kernel
�
7Adam/v/temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp#Adam/v/temp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
#Adam/m/temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/temp_time_dist_conv2d/kernel
�
7Adam/m/temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp#Adam/m/temp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
 Adam/v/dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/dem_time_dist_conv2d/bias
�
4Adam/v/dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/v/dem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
 Adam/m/dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/dem_time_dist_conv2d/bias
�
4Adam/m/dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/m/dem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
"Adam/v/dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*3
shared_name$"Adam/v/dem_time_dist_conv2d/kernel
�
6Adam/v/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/dem_time_dist_conv2d/kernel*(
_output_shapes
:��*
dtype0
�
"Adam/m/dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*3
shared_name$"Adam/m/dem_time_dist_conv2d/kernel
�
6Adam/m/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/dem_time_dist_conv2d/kernel*(
_output_shapes
:��*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
.transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.transformer_encoder/layer_normalization_1/beta
�
Btransformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp.transformer_encoder/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/transformer_encoder/layer_normalization_1/gamma
�
Ctransformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp/transformer_encoder/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
,transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,transformer_encoder/layer_normalization/beta
�
@transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp,transformer_encoder/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
-transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-transformer_encoder/layer_normalization/gamma
�
Atransformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp-transformer_encoder/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
>transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>transformer_encoder/multi_head_attention/attention_output/bias
�
Rtransformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
@transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*Q
shared_nameB@transformer_encoder/multi_head_attention/attention_output/kernel
�
Ttransformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder/multi_head_attention/attention_output/kernel*$
_output_shapes
:��*
dtype0
�
3transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*D
shared_name53transformer_encoder/multi_head_attention/value/bias
�
Gtransformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/value/bias*
_output_shapes
:	�*
dtype0
�
5transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*F
shared_name75transformer_encoder/multi_head_attention/value/kernel
�
Itransformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_encoder/multi_head_attention/value/kernel*$
_output_shapes
:��*
dtype0
�
1transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*B
shared_name31transformer_encoder/multi_head_attention/key/bias
�
Etransformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp1transformer_encoder/multi_head_attention/key/bias*
_output_shapes
:	�*
dtype0
�
3transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*D
shared_name53transformer_encoder/multi_head_attention/key/kernel
�
Gtransformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/key/kernel*$
_output_shapes
:��*
dtype0
�
3transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*D
shared_name53transformer_encoder/multi_head_attention/query/bias
�
Gtransformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/query/bias*
_output_shapes
:	�*
dtype0
�
5transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*F
shared_name75transformer_encoder/multi_head_attention/query/kernel
�
Itransformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_encoder/multi_head_attention/query/kernel*$
_output_shapes
:��*
dtype0
�
et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameet_time_dist_conv2d/bias
�
,et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpet_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**+
shared_nameet_time_dist_conv2d/kernel
�
.et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpet_time_dist_conv2d/kernel*&
_output_shapes
:**
dtype0
�
swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameswe_time_dist_conv2d/bias
�
-swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpswe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:{�*,
shared_nameswe_time_dist_conv2d/kernel
�
/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpswe_time_dist_conv2d/kernel*'
_output_shapes
:{�*
dtype0
�
precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameprecip_time_dist_conv2d/bias
�
0precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpprecip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name precip_time_dist_conv2d/kernel
�
2precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpprecip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametemp_time_dist_conv2d/bias
�
.temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOptemp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametemp_time_dist_conv2d/kernel
�
0temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOptemp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
�
dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedem_time_dist_conv2d/bias
�
-dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpdem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
�
dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_namedem_time_dist_conv2d/kernel
�
/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpdem_time_dist_conv2d/kernel*(
_output_shapes
:��*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_dem_inputsPlaceholder*5
_output_shapes#
!:����������	�*
dtype0**
shape!:����������	�
�
serving_default_et_inputsPlaceholder*3
_output_shapes!
:���������Oj*
dtype0*(
shape:���������Oj
�
serving_default_precip_inputsPlaceholder*3
_output_shapes!
:���������*
dtype0*(
shape:���������
�
serving_default_swe_inputsPlaceholder*5
_output_shapes#
!:�����������*
dtype0**
shape!:�����������
�
serving_default_temp_inputsPlaceholder*3
_output_shapes!
:���������*
dtype0*(
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dem_inputsserving_default_et_inputsserving_default_precip_inputsserving_default_swe_inputsserving_default_temp_inputset_time_dist_conv2d/kernelet_time_dist_conv2d/biasswe_time_dist_conv2d/kernelswe_time_dist_conv2d/biasprecip_time_dist_conv2d/kernelprecip_time_dist_conv2d/biastemp_time_dist_conv2d/kerneltemp_time_dist_conv2d/biasdem_time_dist_conv2d/kerneldem_time_dist_conv2d/bias5transformer_encoder/multi_head_attention/query/kernel3transformer_encoder/multi_head_attention/query/bias3transformer_encoder/multi_head_attention/key/kernel1transformer_encoder/multi_head_attention/key/bias5transformer_encoder/multi_head_attention/value/kernel3transformer_encoder/multi_head_attention/value/bias@transformer_encoder/multi_head_attention/attention_output/kernel>transformer_encoder/multi_head_attention/attention_output/bias-transformer_encoder/layer_normalization/gamma,transformer_encoder/layer_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias/transformer_encoder/layer_normalization_1/gamma.transformer_encoder/layer_normalization_1/betadense_2/kerneldense_2/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_46226

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
* 
* 
* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2layer*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
	@layer*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Glayer* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
	Nlayer* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
	Ulayer* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer* 
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
	clayer* 
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p	attention
q
dense_proj
rlayernorm_1
slayernorm_2*
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdem_time_dist_conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdem_time_dist_conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtemp_time_dist_conv2d/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEtemp_time_dist_conv2d/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprecip_time_dist_conv2d/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprecip_time_dist_conv2d/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEswe_time_dist_conv2d/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEswe_time_dist_conv2d/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEet_time_dist_conv2d/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEet_time_dist_conv2d/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_encoder/multi_head_attention/query/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_encoder/multi_head_attention/query/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_encoder/multi_head_attention/key/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_encoder/multi_head_attention/key/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_encoder/multi_head_attention/value/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_encoder/multi_head_attention/value/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE@transformer_encoder/multi_head_attention/attention_output/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>transformer_encoder/multi_head_attention/attention_output/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_encoder/layer_normalization/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,transformer_encoder/layer_normalization/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_encoder/layer_normalization_1/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_encoder/layer_normalization_1/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
* 
* 
* 

$0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

+0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

20*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

90*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

@0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
	
G0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
	
N0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
	
U0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
	
\0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
	
c0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
 
p0
q1
r2
s3*
* 
* 
* 
* 
* 
* 
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
mg
VARIABLE_VALUE"Adam/m/dem_time_dist_conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/dem_time_dist_conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/dem_time_dist_conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/dem_time_dist_conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/temp_time_dist_conv2d/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/temp_time_dist_conv2d/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/temp_time_dist_conv2d/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/temp_time_dist_conv2d/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/precip_time_dist_conv2d/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/precip_time_dist_conv2d/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/precip_time_dist_conv2d/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/precip_time_dist_conv2d/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/swe_time_dist_conv2d/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/swe_time_dist_conv2d/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/swe_time_dist_conv2d/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/swe_time_dist_conv2d/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/et_time_dist_conv2d/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/et_time_dist_conv2d/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/et_time_dist_conv2d/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/et_time_dist_conv2d/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/m/transformer_encoder/multi_head_attention/query/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/v/transformer_encoder/multi_head_attention/query/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/query/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/query/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/key/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/key/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUE8Adam/m/transformer_encoder/multi_head_attention/key/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUE8Adam/v/transformer_encoder/multi_head_attention/key/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/m/transformer_encoder/multi_head_attention/value/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE<Adam/v/transformer_encoder/multi_head_attention/value/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/value/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/value/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEAdam/v/transformer_encoder/multi_head_attention/attention_output/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/m/transformer_encoder/layer_normalization/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/v/transformer_encoder/layer_normalization/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/transformer_encoder/layer_normalization/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/transformer_encoder/layer_normalization/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/m/transformer_encoder/layer_normalization_1/gamma2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/v/transformer_encoder/layer_normalization_1/gamma2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/transformer_encoder/layer_normalization_1/beta2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/transformer_encoder/layer_normalization_1/beta2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/dem_time_dist_conv2d/kernel/Read/ReadVariableOp-dem_time_dist_conv2d/bias/Read/ReadVariableOp0temp_time_dist_conv2d/kernel/Read/ReadVariableOp.temp_time_dist_conv2d/bias/Read/ReadVariableOp2precip_time_dist_conv2d/kernel/Read/ReadVariableOp0precip_time_dist_conv2d/bias/Read/ReadVariableOp/swe_time_dist_conv2d/kernel/Read/ReadVariableOp-swe_time_dist_conv2d/bias/Read/ReadVariableOp.et_time_dist_conv2d/kernel/Read/ReadVariableOp,et_time_dist_conv2d/bias/Read/ReadVariableOpItransformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpGtransformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpGtransformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpEtransformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpItransformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpGtransformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpTtransformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpRtransformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAtransformer_encoder/layer_normalization/gamma/Read/ReadVariableOp@transformer_encoder/layer_normalization/beta/Read/ReadVariableOpCtransformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpBtransformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/dem_time_dist_conv2d/kernel/Read/ReadVariableOp6Adam/v/dem_time_dist_conv2d/kernel/Read/ReadVariableOp4Adam/m/dem_time_dist_conv2d/bias/Read/ReadVariableOp4Adam/v/dem_time_dist_conv2d/bias/Read/ReadVariableOp7Adam/m/temp_time_dist_conv2d/kernel/Read/ReadVariableOp7Adam/v/temp_time_dist_conv2d/kernel/Read/ReadVariableOp5Adam/m/temp_time_dist_conv2d/bias/Read/ReadVariableOp5Adam/v/temp_time_dist_conv2d/bias/Read/ReadVariableOp9Adam/m/precip_time_dist_conv2d/kernel/Read/ReadVariableOp9Adam/v/precip_time_dist_conv2d/kernel/Read/ReadVariableOp7Adam/m/precip_time_dist_conv2d/bias/Read/ReadVariableOp7Adam/v/precip_time_dist_conv2d/bias/Read/ReadVariableOp6Adam/m/swe_time_dist_conv2d/kernel/Read/ReadVariableOp6Adam/v/swe_time_dist_conv2d/kernel/Read/ReadVariableOp4Adam/m/swe_time_dist_conv2d/bias/Read/ReadVariableOp4Adam/v/swe_time_dist_conv2d/bias/Read/ReadVariableOp5Adam/m/et_time_dist_conv2d/kernel/Read/ReadVariableOp5Adam/v/et_time_dist_conv2d/kernel/Read/ReadVariableOp3Adam/m/et_time_dist_conv2d/bias/Read/ReadVariableOp3Adam/v/et_time_dist_conv2d/bias/Read/ReadVariableOpPAdam/m/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpPAdam/v/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpNAdam/m/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpNAdam/v/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpNAdam/m/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpNAdam/v/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpLAdam/m/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpLAdam/v/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpPAdam/m/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpPAdam/v/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpNAdam/m/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpNAdam/v/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOp[Adam/m/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOp[Adam/v/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpYAdam/m/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpYAdam/v/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOpHAdam/m/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpHAdam/v/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpGAdam/m/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpGAdam/v/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpJAdam/m/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpJAdam/v/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpIAdam/m/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpIAdam/v/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOp)Adam/m/dense_2/kernel/Read/ReadVariableOp)Adam/v/dense_2/kernel/Read/ReadVariableOp'Adam/m/dense_2/bias/Read/ReadVariableOp'Adam/v/dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*e
Tin^
\2Z	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_48668
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdem_time_dist_conv2d/kerneldem_time_dist_conv2d/biastemp_time_dist_conv2d/kerneltemp_time_dist_conv2d/biasprecip_time_dist_conv2d/kernelprecip_time_dist_conv2d/biasswe_time_dist_conv2d/kernelswe_time_dist_conv2d/biaset_time_dist_conv2d/kernelet_time_dist_conv2d/bias5transformer_encoder/multi_head_attention/query/kernel3transformer_encoder/multi_head_attention/query/bias3transformer_encoder/multi_head_attention/key/kernel1transformer_encoder/multi_head_attention/key/bias5transformer_encoder/multi_head_attention/value/kernel3transformer_encoder/multi_head_attention/value/bias@transformer_encoder/multi_head_attention/attention_output/kernel>transformer_encoder/multi_head_attention/attention_output/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias-transformer_encoder/layer_normalization/gamma,transformer_encoder/layer_normalization/beta/transformer_encoder/layer_normalization_1/gamma.transformer_encoder/layer_normalization_1/beta	iterationlearning_rate"Adam/m/dem_time_dist_conv2d/kernel"Adam/v/dem_time_dist_conv2d/kernel Adam/m/dem_time_dist_conv2d/bias Adam/v/dem_time_dist_conv2d/bias#Adam/m/temp_time_dist_conv2d/kernel#Adam/v/temp_time_dist_conv2d/kernel!Adam/m/temp_time_dist_conv2d/bias!Adam/v/temp_time_dist_conv2d/bias%Adam/m/precip_time_dist_conv2d/kernel%Adam/v/precip_time_dist_conv2d/kernel#Adam/m/precip_time_dist_conv2d/bias#Adam/v/precip_time_dist_conv2d/bias"Adam/m/swe_time_dist_conv2d/kernel"Adam/v/swe_time_dist_conv2d/kernel Adam/m/swe_time_dist_conv2d/bias Adam/v/swe_time_dist_conv2d/bias!Adam/m/et_time_dist_conv2d/kernel!Adam/v/et_time_dist_conv2d/kernelAdam/m/et_time_dist_conv2d/biasAdam/v/et_time_dist_conv2d/bias<Adam/m/transformer_encoder/multi_head_attention/query/kernel<Adam/v/transformer_encoder/multi_head_attention/query/kernel:Adam/m/transformer_encoder/multi_head_attention/query/bias:Adam/v/transformer_encoder/multi_head_attention/query/bias:Adam/m/transformer_encoder/multi_head_attention/key/kernel:Adam/v/transformer_encoder/multi_head_attention/key/kernel8Adam/m/transformer_encoder/multi_head_attention/key/bias8Adam/v/transformer_encoder/multi_head_attention/key/bias<Adam/m/transformer_encoder/multi_head_attention/value/kernel<Adam/v/transformer_encoder/multi_head_attention/value/kernel:Adam/m/transformer_encoder/multi_head_attention/value/bias:Adam/v/transformer_encoder/multi_head_attention/value/biasGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernelGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernelEAdam/m/transformer_encoder/multi_head_attention/attention_output/biasEAdam/v/transformer_encoder/multi_head_attention/attention_output/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias4Adam/m/transformer_encoder/layer_normalization/gamma4Adam/v/transformer_encoder/layer_normalization/gamma3Adam/m/transformer_encoder/layer_normalization/beta3Adam/v/transformer_encoder/layer_normalization/beta6Adam/m/transformer_encoder/layer_normalization_1/gamma6Adam/v/transformer_encoder/layer_normalization_1/gamma5Adam/m/transformer_encoder/layer_normalization_1/beta5Adam/v/transformer_encoder/layer_normalization_1/betaAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotalcount*d
Tin]
[2Y*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_48942��&
�
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47310

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_47984

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44160

inputs(
conv2d_44148:��
conv2d_44150:
identity��conv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:����������	��
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_44148conv2d_44150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44106\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�
G
+__inference_dem_flatten_layer_call_fn_47293

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44565n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
5__inference_temp_time_dist_conv2d_layer_call_fn_47037

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44246�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48103

inputs8
conv2d_readvariableop_resource:*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������Oj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������Oj
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_44961
dense_input
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44937t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:���������$�
%
_user_specified_namedense_input
�
�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_47019

inputsA
%conv2d_conv2d_readvariableop_resource:��4
&conv2d_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:����������	��
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

���
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_46356
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4!
unknown:*
	unknown_0:$
	unknown_1:{�
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:��
	unknown_8:!
	unknown_9:��

unknown_10:	�"

unknown_11:��

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_45835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:����������	�
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:�����������
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:���������Oj
"
_user_specified_name
inputs_4
��
�"
@__inference_model_layer_call_and_return_conditional_losses_46953
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4U
;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:*J
<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:W
<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:{�K
=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:Y
?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:N
@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:W
=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:L
>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:V
:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:��I
;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:l
Ttransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:��]
Jtransformer_encoder_multi_head_attention_query_add_readvariableop_resource:	�j
Rtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:��[
Htransformer_encoder_multi_head_attention_key_add_readvariableop_resource:	�l
Ttransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:��]
Jtransformer_encoder_multi_head_attention_value_add_readvariableop_resource:	�w
_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��d
Utransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:	�T
Etransformer_encoder_layer_normalization_mul_3_readvariableop_resource:	�R
Ctransformer_encoder_layer_normalization_add_readvariableop_resource:	�Y
Ftransformer_encoder_sequential_dense_tensordot_readvariableop_resource:	�R
Dtransformer_encoder_sequential_dense_biasadd_readvariableop_resource:[
Htransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource:	�U
Ftransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:	�V
Gtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:	�T
Etransformer_encoder_layer_normalization_1_add_readvariableop_resource:	�9
&dense_2_matmul_readvariableop_resource:	�5
'dense_2_biasadd_readvariableop_resource:
identity��2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp�1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp�2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp�7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp�6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp�4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp�3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp�5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp�4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp�:transformer_encoder/layer_normalization/add/ReadVariableOp�<transformer_encoder/layer_normalization/mul_3/ReadVariableOp�<transformer_encoder/layer_normalization_1/add/ReadVariableOp�>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp�Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp�Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�?transformer_encoder/multi_head_attention/key/add/ReadVariableOp�Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Atransformer_encoder/multi_head_attention/query/add/ReadVariableOp�Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Atransformer_encoder/multi_head_attention/value/add/ReadVariableOp�Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp�;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp�=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp�=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp�?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpz
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
#et_time_dist_conv2d/conv2d_1/Conv2DConv2D$et_time_dist_conv2d/Reshape:output:0:et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd,et_time_dist_conv2d/conv2d_1/Conv2D:output:0;et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!et_time_dist_conv2d/conv2d_1/ReluRelu-et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:����������
#et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
et_time_dist_conv2d/Reshape_1Reshape/et_time_dist_conv2d/conv2d_1/Relu:activations:0,et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������|
#et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/Reshape_2Reshapeinputs_4,et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������Oj{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
$swe_time_dist_conv2d/conv2d_4/Conv2DConv2D%swe_time_dist_conv2d/Reshape:output:0;swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=S�
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd-swe_time_dist_conv2d/conv2d_4/Conv2D:output:0<swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"swe_time_dist_conv2d/conv2d_4/ReluRelu.swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
swe_time_dist_conv2d/Reshape_1Reshape0swe_time_dist_conv2d/conv2d_4/Relu:activations:0-swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������}
$swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/Reshape_2Reshapeinputs_3-swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:�����������~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
'precip_time_dist_conv2d/conv2d_3/Conv2DConv2D(precip_time_dist_conv2d/Reshape:output:0>precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
(precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd0precip_time_dist_conv2d/conv2d_3/Conv2D:output:0?precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%precip_time_dist_conv2d/conv2d_3/ReluRelu1precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
'precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
!precip_time_dist_conv2d/Reshape_1Reshape3precip_time_dist_conv2d/conv2d_3/Relu:activations:00precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
'precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
!precip_time_dist_conv2d/Reshape_2Reshapeinputs_20precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%temp_time_dist_conv2d/conv2d_2/Conv2DConv2D&temp_time_dist_conv2d/Reshape:output:0<temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd.temp_time_dist_conv2d/conv2d_2/Conv2D:output:0=temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
#temp_time_dist_conv2d/conv2d_2/ReluRelu/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:����������
%temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
temp_time_dist_conv2d/Reshape_1Reshape1temp_time_dist_conv2d/conv2d_2/Relu:activations:0.temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������~
%temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/Reshape_2Reshapeinputs_1.temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshapeinputs_0+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
"dem_time_dist_conv2d/conv2d/Conv2DConv2D%dem_time_dist_conv2d/Reshape:output:09dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

���
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOp;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#dem_time_dist_conv2d/conv2d/BiasAddBiasAdd+dem_time_dist_conv2d/conv2d/Conv2D:output:0:dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 dem_time_dist_conv2d/conv2d/ReluRelu,dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
dem_time_dist_conv2d/Reshape_1Reshape.dem_time_dist_conv2d/conv2d/Relu:activations:0-dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������}
$dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/Reshape_2Reshapeinputs_0-dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:����������	�r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape'dem_time_dist_conv2d/Reshape_1:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������j
dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
dem_flatten/flatten/ReshapeReshapedem_flatten/Reshape:output:0"dem_flatten/flatten/Const:output:0*
T0*(
_output_shapes
:����������p
dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
dem_flatten/Reshape_1Reshape$dem_flatten/flatten/Reshape:output:0$dem_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������t
dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/Reshape_2Reshape'dem_time_dist_conv2d/Reshape_1:output:0$dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape(temp_time_dist_conv2d/Reshape_1:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������m
temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
temp_flatten/flatten_2/ReshapeReshapetemp_flatten/Reshape:output:0%temp_flatten/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������q
temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
temp_flatten/Reshape_1Reshape'temp_flatten/flatten_2/Reshape:output:0%temp_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������u
temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/Reshape_2Reshape(temp_time_dist_conv2d/Reshape_1:output:0%temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape*precip_time_dist_conv2d/Reshape_1:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������o
precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
 precip_flatten/flatten_3/ReshapeReshapeprecip_flatten/Reshape:output:0'precip_flatten/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������s
precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
precip_flatten/Reshape_1Reshape)precip_flatten/flatten_3/Reshape:output:0'precip_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������w
precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/Reshape_2Reshape*precip_time_dist_conv2d/Reshape_1:output:0'precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape'swe_time_dist_conv2d/Reshape_1:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������l
swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
swe_flatten/flatten_4/ReshapeReshapeswe_flatten/Reshape:output:0$swe_flatten/flatten_4/Const:output:0*
T0*(
_output_shapes
:����������p
swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
swe_flatten/Reshape_1Reshape&swe_flatten/flatten_4/Reshape:output:0$swe_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������t
swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/Reshape_2Reshape'swe_time_dist_conv2d/Reshape_1:output:0$swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape&et_time_dist_conv2d/Reshape_1:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������k
et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
et_flatten/flatten_1/ReshapeReshapeet_flatten/Reshape:output:0#et_flatten/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������o
et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
et_flatten/Reshape_1Reshape%et_flatten/flatten_1/Reshape:output:0#et_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������s
et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/Reshape_2Reshape&et_time_dist_conv2d/Reshape_1:output:0#et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2dem_flatten/Reshape_1:output:0temp_flatten/Reshape_1:output:0!precip_flatten/Reshape_1:output:0swe_flatten/Reshape_1:output:0et_flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������$��
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_encoder/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_encoder/multi_head_attention/query/addAddV2Etransformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_encoder/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
?transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpHtransformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_encoder/multi_head_attention/key/addAddV2Ctransformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Gtransformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_encoder/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_encoder/multi_head_attention/value/addAddV2Etransformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�s
.transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_encoder/multi_head_attention/MulMul6transformer_encoder/multi_head_attention/query/add:z:07transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
6transformer_encoder/multi_head_attention/einsum/EinsumEinsum4transformer_encoder/multi_head_attention/key/add:z:00transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
8transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmax?transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
9transformer_encoder/multi_head_attention/dropout/IdentityIdentityBtransformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumBtransformer_encoder/multi_head_attention/dropout/Identity:output:06transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpUtransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_encoder/multi_head_attention/attention_output/addAddV2Ptransformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Ttransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
transformer_encoder/addAddV2concatenate/concat:output:0Atransformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�x
-transformer_encoder/layer_normalization/ShapeShapetransformer_encoder/add:z:0*
T0*
_output_shapes
:�
;transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5transformer_encoder/layer_normalization/strided_sliceStridedSlice6transformer_encoder/layer_normalization/Shape:output:0Dtransformer_encoder/layer_normalization/strided_slice/stack:output:0Ftransformer_encoder/layer_normalization/strided_slice/stack_1:output:0Ftransformer_encoder/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-transformer_encoder/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
+transformer_encoder/layer_normalization/mulMul6transformer_encoder/layer_normalization/mul/x:output:0>transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: �
=transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization/strided_slice_1StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_1/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-transformer_encoder/layer_normalization/mul_1Mul/transformer_encoder/layer_normalization/mul:z:0@transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: �
=transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization/strided_slice_2StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_2/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_2/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/transformer_encoder/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
-transformer_encoder/layer_normalization/mul_2Mul8transformer_encoder/layer_normalization/mul_2/x:output:0@transformer_encoder/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: y
7transformer_encoder/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :y
7transformer_encoder/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
5transformer_encoder/layer_normalization/Reshape/shapePack@transformer_encoder/layer_normalization/Reshape/shape/0:output:01transformer_encoder/layer_normalization/mul_1:z:01transformer_encoder/layer_normalization/mul_2:z:0@transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
/transformer_encoder/layer_normalization/ReshapeReshapetransformer_encoder/add:z:0>transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
3transformer_encoder/layer_normalization/ones/packedPack1transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:w
2transformer_encoder/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,transformer_encoder/layer_normalization/onesFill<transformer_encoder/layer_normalization/ones/packed:output:0;transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:����������
4transformer_encoder/layer_normalization/zeros/packedPack1transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:x
3transformer_encoder/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-transformer_encoder/layer_normalization/zerosFill=transformer_encoder/layer_normalization/zeros/packed:output:0<transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������p
-transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB r
/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
8transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV38transformer_encoder/layer_normalization/Reshape:output:05transformer_encoder/layer_normalization/ones:output:06transformer_encoder/layer_normalization/zeros:output:06transformer_encoder/layer_normalization/Const:output:08transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
1transformer_encoder/layer_normalization/Reshape_1Reshape<transformer_encoder/layer_normalization/FusedBatchNormV3:y:06transformer_encoder/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
<transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-transformer_encoder/layer_normalization/mul_3Mul:transformer_encoder/layer_normalization/Reshape_1:output:0Dtransformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
:transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpCtransformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+transformer_encoder/layer_normalization/addAddV21transformer_encoder/layer_normalization/mul_3:z:0Btransformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0}
3transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_encoder/sequential/dense/Tensordot/ShapeShape/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:~
<transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/free:output:0Etransformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense/Tensordot/GatherV2_1GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Gtransformer_encoder/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_encoder/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_encoder/sequential/dense/Tensordot/ProdProd@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0=transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
6transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_encoder/sequential/dense/Tensordot/Prod_1ProdBtransformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0?transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_encoder/sequential/dense/Tensordot/concatConcatV2<transformer_encoder/sequential/dense/Tensordot/free:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Ctransformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_encoder/sequential/dense/Tensordot/stackPack<transformer_encoder/sequential/dense/Tensordot/Prod:output:0>transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_encoder/sequential/dense/Tensordot/transpose	Transpose/transformer_encoder/layer_normalization/add:z:0>transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
6transformer_encoder/sequential/dense/Tensordot/ReshapeReshape<transformer_encoder/sequential/dense/Tensordot/transpose:y:0=transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_encoder/sequential/dense/Tensordot/MatMulMatMul?transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Etransformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Etransformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_encoder/sequential/dense/TensordotReshape?transformer_encoder/sequential/dense/Tensordot/MatMul:product:0@transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpDtransformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
,transformer_encoder/sequential/dense/BiasAddBiasAdd7transformer_encoder/sequential/dense/Tensordot:output:0Ctransformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$�
)transformer_encoder/sequential/dense/ReluRelu5transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpHtransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0
5transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_encoder/sequential/dense_1/Tensordot/ShapeShape7transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:�
>transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/free:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Itransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_encoder/sequential/dense_1/Tensordot/ProdProdBtransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
8transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdDtransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2>transformer_encoder/sequential/dense_1/Tensordot/free:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Etransformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_encoder/sequential/dense_1/Tensordot/stackPack>transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0@transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose7transformer_encoder/sequential/dense/Relu:activations:0@transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
8transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshape>transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0?transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulAtransformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Btransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_encoder/sequential/dense_1/TensordotReshapeAtransformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Btransformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_encoder/sequential/dense_1/BiasAddBiasAdd9transformer_encoder/sequential/dense_1/Tensordot:output:0Etransformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
transformer_encoder/add_1AddV2/transformer_encoder/layer_normalization/add:z:07transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�|
/transformer_encoder/layer_normalization_1/ShapeShapetransformer_encoder/add_1:z:0*
T0*
_output_shapes
:�
=transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization_1/strided_sliceStridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Ftransformer_encoder/layer_normalization_1/strided_slice/stack:output:0Htransformer_encoder/layer_normalization_1/strided_slice/stack_1:output:0Htransformer_encoder/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/transformer_encoder/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
-transformer_encoder/layer_normalization_1/mulMul8transformer_encoder/layer_normalization_1/mul/x:output:0@transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: �
?transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
/transformer_encoder/layer_normalization_1/mul_1Mul1transformer_encoder/layer_normalization_1/mul:z:0Btransformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: �
?transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9transformer_encoder/layer_normalization_1/strided_slice_2StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_2/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_2/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_encoder/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
/transformer_encoder/layer_normalization_1/mul_2Mul:transformer_encoder/layer_normalization_1/mul_2/x:output:0Btransformer_encoder/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: {
9transformer_encoder/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :{
9transformer_encoder/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
7transformer_encoder/layer_normalization_1/Reshape/shapePackBtransformer_encoder/layer_normalization_1/Reshape/shape/0:output:03transformer_encoder/layer_normalization_1/mul_1:z:03transformer_encoder/layer_normalization_1/mul_2:z:0Btransformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
1transformer_encoder/layer_normalization_1/ReshapeReshapetransformer_encoder/add_1:z:0@transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
5transformer_encoder/layer_normalization_1/ones/packedPack3transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:y
4transformer_encoder/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.transformer_encoder/layer_normalization_1/onesFill>transformer_encoder/layer_normalization_1/ones/packed:output:0=transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
6transformer_encoder/layer_normalization_1/zeros/packedPack3transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:z
5transformer_encoder/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/transformer_encoder/layer_normalization_1/zerosFill?transformer_encoder/layer_normalization_1/zeros/packed:output:0>transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������r
/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB t
1transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
:transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3:transformer_encoder/layer_normalization_1/Reshape:output:07transformer_encoder/layer_normalization_1/ones:output:08transformer_encoder/layer_normalization_1/zeros:output:08transformer_encoder/layer_normalization_1/Const:output:0:transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
3transformer_encoder/layer_normalization_1/Reshape_1Reshape>transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:08transformer_encoder/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpGtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/transformer_encoder/layer_normalization_1/mul_3Mul<transformer_encoder/layer_normalization_1/Reshape_1:output:0Ftransformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
<transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-transformer_encoder/layer_normalization_1/addAddV23transformer_encoder/layer_normalization_1/mul_3:z:0Dtransformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d/MaxMax1transformer_encoder/layer_normalization_1/add:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout/dropout/MulMul!global_max_pooling1d/Max:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������f
dropout/dropout/ShapeShape!global_max_pooling1d/Max:output:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_2/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2^dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp4^et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3^et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp8^precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7^precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp5^swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4^swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp6^temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5^temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp;^transformer_encoder/layer_normalization/add/ReadVariableOp=^transformer_encoder/layer_normalization/mul_3/ReadVariableOp=^transformer_encoder/layer_normalization_1/add/ReadVariableOp?^transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpM^transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpW^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp@^transformer_encoder/multi_head_attention/key/add/ReadVariableOpJ^transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/query/add/ReadVariableOpL^transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/value/add/ReadVariableOpL^transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp<^transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp>^transformer_encoder/sequential/dense/Tensordot/ReadVariableOp>^transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp@^transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2f
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2j
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp2h
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp2r
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp2p
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp2l
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp2j
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp2n
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp2l
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp2x
:transformer_encoder/layer_normalization/add/ReadVariableOp:transformer_encoder/layer_normalization/add/ReadVariableOp2|
<transformer_encoder/layer_normalization/mul_3/ReadVariableOp<transformer_encoder/layer_normalization/mul_3/ReadVariableOp2|
<transformer_encoder/layer_normalization_1/add/ReadVariableOp<transformer_encoder/layer_normalization_1/add/ReadVariableOp2�
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2�
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpLtransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2�
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpVtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
?transformer_encoder/multi_head_attention/key/add/ReadVariableOp?transformer_encoder/multi_head_attention/key/add/ReadVariableOp2�
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpItransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpAtransformer_encoder/multi_head_attention/query/add/ReadVariableOp2�
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpAtransformer_encoder/multi_head_attention/value/add/ReadVariableOp2�
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2z
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2~
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2�
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:_ [
5
_output_shapes#
!:����������	�
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:�����������
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:���������Oj
"
_user_specified_name
inputs_4
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_48023

inputs:
conv2d_readvariableop_resource:��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

��r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������	�
 
_user_specified_nameinputs
�
G
+__inference_swe_flatten_layer_call_fn_47425

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44736n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
J
.__inference_precip_flatten_layer_call_fn_47376

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44652n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47151

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:
identity��conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_3/Conv2DConv2DReshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_48158

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_transformer_encoder_layer_call_fn_47596

inputs
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45620t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
G
+__inference_swe_flatten_layer_call_fn_47420

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44709n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44377

inputs)
conv2d_4_44365:{�
conv2d_4_44367:
identity�� conv2d_4/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:������������
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_4_44365conv2d_4_44367*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44364\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_4/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�
�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44119

inputs(
conv2d_44107:��
conv2d_44109:
identity��conv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:����������	��
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_44107conv2d_44109*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44106\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�g
�
@__inference_model_layer_call_and_return_conditional_losses_45835

inputs
inputs_1
inputs_2
inputs_3
inputs_43
et_time_dist_conv2d_45743:*'
et_time_dist_conv2d_45745:5
swe_time_dist_conv2d_45750:{�(
swe_time_dist_conv2d_45752:7
precip_time_dist_conv2d_45757:+
precip_time_dist_conv2d_45759:5
temp_time_dist_conv2d_45764:)
temp_time_dist_conv2d_45766:6
dem_time_dist_conv2d_45771:��(
dem_time_dist_conv2d_45773:1
transformer_encoder_45794:��,
transformer_encoder_45796:	�1
transformer_encoder_45798:��,
transformer_encoder_45800:	�1
transformer_encoder_45802:��,
transformer_encoder_45804:	�1
transformer_encoder_45806:��(
transformer_encoder_45808:	�(
transformer_encoder_45810:	�(
transformer_encoder_45812:	�,
transformer_encoder_45814:	�'
transformer_encoder_45816:,
transformer_encoder_45818:	�(
transformer_encoder_45820:	�(
transformer_encoder_45822:	�(
transformer_encoder_45824:	� 
dense_2_45829:	�
dense_2_45831:
identity��,dem_time_dist_conv2d/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�+et_time_dist_conv2d/StatefulPartitionedCall�/precip_time_dist_conv2d/StatefulPartitionedCall�,swe_time_dist_conv2d/StatefulPartitionedCall�-temp_time_dist_conv2d/StatefulPartitionedCall�+transformer_encoder/StatefulPartitionedCall�
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_4et_time_dist_conv2d_45743et_time_dist_conv2d_45745*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44504z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_3swe_time_dist_conv2d_45750swe_time_dist_conv2d_45752*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44418{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_2precip_time_dist_conv2d_45757precip_time_dist_conv2d_45759*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44332~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1temp_time_dist_conv2d_45764temp_time_dist_conv2d_45766*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44246|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsdem_time_dist_conv2d_45771dem_time_dist_conv2d_45773*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44160{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshapeinputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44565r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44622s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44679u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44736r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44793q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_45077�
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_45794transformer_encoder_45796transformer_encoder_45798transformer_encoder_45800transformer_encoder_45802transformer_encoder_45804transformer_encoder_45806transformer_encoder_45808transformer_encoder_45810transformer_encoder_45812transformer_encoder_45814transformer_encoder_45816transformer_encoder_45818transformer_encoder_45820transformer_encoder_45822transformer_encoder_45824*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45620�
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999�
dropout/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45402�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_45829dense_2_45831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_45306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:] Y
5
_output_shapes#
!:����������	�
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������Oj
 
_user_specified_nameinputs
��
�1
__inference__traced_save_48668
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_dem_time_dist_conv2d_kernel_read_readvariableop8
4savev2_dem_time_dist_conv2d_bias_read_readvariableop;
7savev2_temp_time_dist_conv2d_kernel_read_readvariableop9
5savev2_temp_time_dist_conv2d_bias_read_readvariableop=
9savev2_precip_time_dist_conv2d_kernel_read_readvariableop;
7savev2_precip_time_dist_conv2d_bias_read_readvariableop:
6savev2_swe_time_dist_conv2d_kernel_read_readvariableop8
4savev2_swe_time_dist_conv2d_bias_read_readvariableop9
5savev2_et_time_dist_conv2d_kernel_read_readvariableop7
3savev2_et_time_dist_conv2d_bias_read_readvariableopT
Psavev2_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopR
Nsavev2_transformer_encoder_multi_head_attention_query_bias_read_readvariableopR
Nsavev2_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopP
Lsavev2_transformer_encoder_multi_head_attention_key_bias_read_readvariableopT
Psavev2_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopR
Nsavev2_transformer_encoder_multi_head_attention_value_bias_read_readvariableop_
[savev2_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableop]
Ysavev2_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableopL
Hsavev2_transformer_encoder_layer_normalization_gamma_read_readvariableopK
Gsavev2_transformer_encoder_layer_normalization_beta_read_readvariableopN
Jsavev2_transformer_encoder_layer_normalization_1_gamma_read_readvariableopM
Isavev2_transformer_encoder_layer_normalization_1_beta_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_dem_time_dist_conv2d_kernel_read_readvariableopA
=savev2_adam_v_dem_time_dist_conv2d_kernel_read_readvariableop?
;savev2_adam_m_dem_time_dist_conv2d_bias_read_readvariableop?
;savev2_adam_v_dem_time_dist_conv2d_bias_read_readvariableopB
>savev2_adam_m_temp_time_dist_conv2d_kernel_read_readvariableopB
>savev2_adam_v_temp_time_dist_conv2d_kernel_read_readvariableop@
<savev2_adam_m_temp_time_dist_conv2d_bias_read_readvariableop@
<savev2_adam_v_temp_time_dist_conv2d_bias_read_readvariableopD
@savev2_adam_m_precip_time_dist_conv2d_kernel_read_readvariableopD
@savev2_adam_v_precip_time_dist_conv2d_kernel_read_readvariableopB
>savev2_adam_m_precip_time_dist_conv2d_bias_read_readvariableopB
>savev2_adam_v_precip_time_dist_conv2d_bias_read_readvariableopA
=savev2_adam_m_swe_time_dist_conv2d_kernel_read_readvariableopA
=savev2_adam_v_swe_time_dist_conv2d_kernel_read_readvariableop?
;savev2_adam_m_swe_time_dist_conv2d_bias_read_readvariableop?
;savev2_adam_v_swe_time_dist_conv2d_bias_read_readvariableop@
<savev2_adam_m_et_time_dist_conv2d_kernel_read_readvariableop@
<savev2_adam_v_et_time_dist_conv2d_kernel_read_readvariableop>
:savev2_adam_m_et_time_dist_conv2d_bias_read_readvariableop>
:savev2_adam_v_et_time_dist_conv2d_bias_read_readvariableop[
Wsavev2_adam_m_transformer_encoder_multi_head_attention_query_kernel_read_readvariableop[
Wsavev2_adam_v_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopY
Usavev2_adam_m_transformer_encoder_multi_head_attention_query_bias_read_readvariableopY
Usavev2_adam_v_transformer_encoder_multi_head_attention_query_bias_read_readvariableopY
Usavev2_adam_m_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopY
Usavev2_adam_v_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopW
Ssavev2_adam_m_transformer_encoder_multi_head_attention_key_bias_read_readvariableopW
Ssavev2_adam_v_transformer_encoder_multi_head_attention_key_bias_read_readvariableop[
Wsavev2_adam_m_transformer_encoder_multi_head_attention_value_kernel_read_readvariableop[
Wsavev2_adam_v_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopY
Usavev2_adam_m_transformer_encoder_multi_head_attention_value_bias_read_readvariableopY
Usavev2_adam_v_transformer_encoder_multi_head_attention_value_bias_read_readvariableopf
bsavev2_adam_m_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopf
bsavev2_adam_v_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopd
`savev2_adam_m_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableopd
`savev2_adam_v_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableopS
Osavev2_adam_m_transformer_encoder_layer_normalization_gamma_read_readvariableopS
Osavev2_adam_v_transformer_encoder_layer_normalization_gamma_read_readvariableopR
Nsavev2_adam_m_transformer_encoder_layer_normalization_beta_read_readvariableopR
Nsavev2_adam_v_transformer_encoder_layer_normalization_beta_read_readvariableopU
Qsavev2_adam_m_transformer_encoder_layer_normalization_1_gamma_read_readvariableopU
Qsavev2_adam_v_transformer_encoder_layer_normalization_1_gamma_read_readvariableopT
Psavev2_adam_m_transformer_encoder_layer_normalization_1_beta_read_readvariableopT
Psavev2_adam_v_transformer_encoder_layer_normalization_1_beta_read_readvariableop4
0savev2_adam_m_dense_2_kernel_read_readvariableop4
0savev2_adam_v_dense_2_kernel_read_readvariableop2
.savev2_adam_m_dense_2_bias_read_readvariableop2
.savev2_adam_v_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�!
value�!B�!YB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �0
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_dem_time_dist_conv2d_kernel_read_readvariableop4savev2_dem_time_dist_conv2d_bias_read_readvariableop7savev2_temp_time_dist_conv2d_kernel_read_readvariableop5savev2_temp_time_dist_conv2d_bias_read_readvariableop9savev2_precip_time_dist_conv2d_kernel_read_readvariableop7savev2_precip_time_dist_conv2d_bias_read_readvariableop6savev2_swe_time_dist_conv2d_kernel_read_readvariableop4savev2_swe_time_dist_conv2d_bias_read_readvariableop5savev2_et_time_dist_conv2d_kernel_read_readvariableop3savev2_et_time_dist_conv2d_bias_read_readvariableopPsavev2_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_query_bias_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopLsavev2_transformer_encoder_multi_head_attention_key_bias_read_readvariableopPsavev2_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_value_bias_read_readvariableop[savev2_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopYsavev2_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopHsavev2_transformer_encoder_layer_normalization_gamma_read_readvariableopGsavev2_transformer_encoder_layer_normalization_beta_read_readvariableopJsavev2_transformer_encoder_layer_normalization_1_gamma_read_readvariableopIsavev2_transformer_encoder_layer_normalization_1_beta_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_dem_time_dist_conv2d_kernel_read_readvariableop=savev2_adam_v_dem_time_dist_conv2d_kernel_read_readvariableop;savev2_adam_m_dem_time_dist_conv2d_bias_read_readvariableop;savev2_adam_v_dem_time_dist_conv2d_bias_read_readvariableop>savev2_adam_m_temp_time_dist_conv2d_kernel_read_readvariableop>savev2_adam_v_temp_time_dist_conv2d_kernel_read_readvariableop<savev2_adam_m_temp_time_dist_conv2d_bias_read_readvariableop<savev2_adam_v_temp_time_dist_conv2d_bias_read_readvariableop@savev2_adam_m_precip_time_dist_conv2d_kernel_read_readvariableop@savev2_adam_v_precip_time_dist_conv2d_kernel_read_readvariableop>savev2_adam_m_precip_time_dist_conv2d_bias_read_readvariableop>savev2_adam_v_precip_time_dist_conv2d_bias_read_readvariableop=savev2_adam_m_swe_time_dist_conv2d_kernel_read_readvariableop=savev2_adam_v_swe_time_dist_conv2d_kernel_read_readvariableop;savev2_adam_m_swe_time_dist_conv2d_bias_read_readvariableop;savev2_adam_v_swe_time_dist_conv2d_bias_read_readvariableop<savev2_adam_m_et_time_dist_conv2d_kernel_read_readvariableop<savev2_adam_v_et_time_dist_conv2d_kernel_read_readvariableop:savev2_adam_m_et_time_dist_conv2d_bias_read_readvariableop:savev2_adam_v_et_time_dist_conv2d_bias_read_readvariableopWsavev2_adam_m_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopWsavev2_adam_v_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_query_bias_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_query_bias_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopSsavev2_adam_m_transformer_encoder_multi_head_attention_key_bias_read_readvariableopSsavev2_adam_v_transformer_encoder_multi_head_attention_key_bias_read_readvariableopWsavev2_adam_m_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopWsavev2_adam_v_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_value_bias_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_value_bias_read_readvariableopbsavev2_adam_m_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopbsavev2_adam_v_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableop`savev2_adam_m_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop`savev2_adam_v_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableopOsavev2_adam_m_transformer_encoder_layer_normalization_gamma_read_readvariableopOsavev2_adam_v_transformer_encoder_layer_normalization_gamma_read_readvariableopNsavev2_adam_m_transformer_encoder_layer_normalization_beta_read_readvariableopNsavev2_adam_v_transformer_encoder_layer_normalization_beta_read_readvariableopQsavev2_adam_m_transformer_encoder_layer_normalization_1_gamma_read_readvariableopQsavev2_adam_v_transformer_encoder_layer_normalization_1_gamma_read_readvariableopPsavev2_adam_m_transformer_encoder_layer_normalization_1_beta_read_readvariableopPsavev2_adam_v_transformer_encoder_layer_normalization_1_beta_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *g
dtypes]
[2Y	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�::��::::::{�::*::��:	�:��:	�:��:	�:��:�:	�::	�:�:�:�:�:�: : :��:��:::::::::::{�:{�:::*:*:::��:��:	�:	�:��:��:	�:	�:��:��:	�:	�:��:��:�:�:	�:	�:::	�:	�:�:�:�:�:�:�:�:�:�:�:	�:	�::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::.*
(
_output_shapes
:��: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-	)
'
_output_shapes
:{�: 


_output_shapes
::,(
&
_output_shapes
:*: 

_output_shapes
::*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:��:. *
(
_output_shapes
:��: !

_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::-+)
'
_output_shapes
:{�:-,)
'
_output_shapes
:{�: -

_output_shapes
:: .

_output_shapes
::,/(
&
_output_shapes
:*:,0(
&
_output_shapes
:*: 1

_output_shapes
:: 2

_output_shapes
::*3&
$
_output_shapes
:��:*4&
$
_output_shapes
:��:%5!

_output_shapes
:	�:%6!

_output_shapes
:	�:*7&
$
_output_shapes
:��:*8&
$
_output_shapes
:��:%9!

_output_shapes
:	�:%:!

_output_shapes
:	�:*;&
$
_output_shapes
:��:*<&
$
_output_shapes
:��:%=!

_output_shapes
:	�:%>!

_output_shapes
:	�:*?&
$
_output_shapes
:��:*@&
$
_output_shapes
:��:!A

_output_shapes	
:�:!B

_output_shapes	
:�:%C!

_output_shapes
:	�:%D!

_output_shapes
:	�: E

_output_shapes
:: F

_output_shapes
::%G!

_output_shapes
:	�:%H!

_output_shapes
:	�:!I

_output_shapes	
:�:!J

_output_shapes	
:�:!K

_output_shapes	
:�:!L

_output_shapes	
:�:!M

_output_shapes	
:�:!N

_output_shapes	
:�:!O

_output_shapes	
:�:!P

_output_shapes	
:�:!Q

_output_shapes	
:�:!R

_output_shapes	
:�:%S!

_output_shapes
:	�:%T!

_output_shapes
:	�: U

_output_shapes
:: V

_output_shapes
::W

_output_shapes
: :X

_output_shapes
: :Y

_output_shapes
: 
�
�
'__inference_dense_2_layer_call_fn_47993

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_45306o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_46291
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4!
unknown:*
	unknown_0:$
	unknown_1:{�
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:��
	unknown_8:!
	unknown_9:��

unknown_10:	�"

unknown_11:��

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_45313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:����������	�
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:�����������
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:���������Oj
"
_user_specified_name
inputs_4
�
`
'__inference_dropout_layer_call_fn_47967

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45402p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_4_layer_call_fn_48072

inputs"
unknown:{�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44364w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_47503

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47259

inputsA
'conv2d_1_conv2d_readvariableop_resource:*6
(conv2d_1_biasadd_readvariableop_resource:
identity��conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_45959

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs!
unknown:*
	unknown_0:$
	unknown_1:{�
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:��
	unknown_8:!
	unknown_9:��

unknown_10:	�"

unknown_11:��

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_45835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs
�
�
*__inference_sequential_layer_call_fn_44888
dense_input
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44877t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:���������$�
%
_user_specified_namedense_input
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_44870

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������$�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������$
 
_user_specified_nameinputs
�
E
)__inference_flatten_3_layer_call_fn_48130

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_44645a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
+__inference_concatenate_layer_call_fn_47512
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_45077e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������$�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_4
��
�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45620

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�@
1layer_normalization_mul_3_readvariableop_resource:	�>
/layer_normalization_add_readvariableop_resource:	�E
2sequential_dense_tensordot_readvariableop_resource:	�>
0sequential_dense_biasadd_readvariableop_resource:G
4sequential_dense_1_tensordot_readvariableop_resource:	�A
2sequential_dense_1_biasadd_readvariableop_resource:	�B
3layer_normalization_1_mul_3_readvariableop_resource:	�@
1layer_normalization_1_add_readvariableop_resource:	�
identity��&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�z
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�P
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:����������t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
 sequential/dense/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�T
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�q
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44679

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_3/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_44645\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_3/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_47486

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
��
�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47946

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�@
1layer_normalization_mul_3_readvariableop_resource:	�>
/layer_normalization_add_readvariableop_resource:	�E
2sequential_dense_tensordot_readvariableop_resource:	�>
0sequential_dense_biasadd_readvariableop_resource:G
4sequential_dense_1_tensordot_readvariableop_resource:	�A
2sequential_dense_1_biasadd_readvariableop_resource:	�B
3layer_normalization_1_mul_3_readvariableop_resource:	�@
1layer_normalization_1_add_readvariableop_resource:	�
identity��&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�z
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�P
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:����������t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
 sequential/dense/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�T
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�q
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
��
�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45254

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�@
1layer_normalization_mul_3_readvariableop_resource:	�>
/layer_normalization_add_readvariableop_resource:	�E
2sequential_dense_tensordot_readvariableop_resource:	�>
0sequential_dense_biasadd_readvariableop_resource:G
4sequential_dense_1_tensordot_readvariableop_resource:	�A
2sequential_dense_1_biasadd_readvariableop_resource:	�B
3layer_normalization_1_mul_3_readvariableop_resource:	�@
1layer_normalization_1_add_readvariableop_resource:	�
identity��&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�z
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�P
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:����������t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
 sequential/dense/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�T
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�q
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
��
�$
 __inference__wrapped_model_44081

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs[
Amodel_et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:*P
Bmodel_et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:]
Bmodel_swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:{�Q
Cmodel_swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:_
Emodel_precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:T
Fmodel_precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:]
Cmodel_temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:R
Dmodel_temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:\
@model_dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:��O
Amodel_dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:r
Zmodel_transformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:��c
Pmodel_transformer_encoder_multi_head_attention_query_add_readvariableop_resource:	�p
Xmodel_transformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:��a
Nmodel_transformer_encoder_multi_head_attention_key_add_readvariableop_resource:	�r
Zmodel_transformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:��c
Pmodel_transformer_encoder_multi_head_attention_value_add_readvariableop_resource:	�}
emodel_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��j
[model_transformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:	�Z
Kmodel_transformer_encoder_layer_normalization_mul_3_readvariableop_resource:	�X
Imodel_transformer_encoder_layer_normalization_add_readvariableop_resource:	�_
Lmodel_transformer_encoder_sequential_dense_tensordot_readvariableop_resource:	�X
Jmodel_transformer_encoder_sequential_dense_biasadd_readvariableop_resource:a
Nmodel_transformer_encoder_sequential_dense_1_tensordot_readvariableop_resource:	�[
Lmodel_transformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:	�\
Mmodel_transformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:	�Z
Kmodel_transformer_encoder_layer_normalization_1_add_readvariableop_resource:	�?
,model_dense_2_matmul_readvariableop_resource:	�;
-model_dense_2_biasadd_readvariableop_resource:
identity��8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp�7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp�8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp�=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp�<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp�:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp�9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp�;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp�:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp�@model/transformer_encoder/layer_normalization/add/ReadVariableOp�Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp�Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp�Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp�Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp�\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp�Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp�Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp�Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp�Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp�Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp�Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp�Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp�
'model/et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
!model/et_time_dist_conv2d/ReshapeReshape	et_inputs0model/et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAmodel_et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
)model/et_time_dist_conv2d/conv2d_1/Conv2DConv2D*model/et_time_dist_conv2d/Reshape:output:0@model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBmodel_et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
*model/et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd2model/et_time_dist_conv2d/conv2d_1/Conv2D:output:0Amodel/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
'model/et_time_dist_conv2d/conv2d_1/ReluRelu3model/et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:����������
)model/et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
#model/et_time_dist_conv2d/Reshape_1Reshape5model/et_time_dist_conv2d/conv2d_1/Relu:activations:02model/et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
)model/et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
#model/et_time_dist_conv2d/Reshape_2Reshape	et_inputs2model/et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������Oj�
(model/swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
"model/swe_time_dist_conv2d/ReshapeReshape
swe_inputs1model/swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOpBmodel_swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
*model/swe_time_dist_conv2d/conv2d_4/Conv2DConv2D+model/swe_time_dist_conv2d/Reshape:output:0Amodel/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=S�
:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpCmodel_swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+model/swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd3model/swe_time_dist_conv2d/conv2d_4/Conv2D:output:0Bmodel/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
(model/swe_time_dist_conv2d/conv2d_4/ReluRelu4model/swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:����������
*model/swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
$model/swe_time_dist_conv2d/Reshape_1Reshape6model/swe_time_dist_conv2d/conv2d_4/Relu:activations:03model/swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
*model/swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
$model/swe_time_dist_conv2d/Reshape_2Reshape
swe_inputs3model/swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:������������
+model/precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
%model/precip_time_dist_conv2d/ReshapeReshapeprecip_inputs4model/precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOpEmodel_precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
-model/precip_time_dist_conv2d/conv2d_3/Conv2DConv2D.model/precip_time_dist_conv2d/Reshape:output:0Dmodel/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpFmodel_precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
.model/precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd6model/precip_time_dist_conv2d/conv2d_3/Conv2D:output:0Emodel/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
+model/precip_time_dist_conv2d/conv2d_3/ReluRelu7model/precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
-model/precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
'model/precip_time_dist_conv2d/Reshape_1Reshape9model/precip_time_dist_conv2d/conv2d_3/Relu:activations:06model/precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
-model/precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
'model/precip_time_dist_conv2d/Reshape_2Reshapeprecip_inputs6model/precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:����������
)model/temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
#model/temp_time_dist_conv2d/ReshapeReshapetemp_inputs2model/temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOpCmodel_temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
+model/temp_time_dist_conv2d/conv2d_2/Conv2DConv2D,model/temp_time_dist_conv2d/Reshape:output:0Bmodel/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpDmodel_temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
,model/temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd4model/temp_time_dist_conv2d/conv2d_2/Conv2D:output:0Cmodel/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
)model/temp_time_dist_conv2d/conv2d_2/ReluRelu5model/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:����������
+model/temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
%model/temp_time_dist_conv2d/Reshape_1Reshape7model/temp_time_dist_conv2d/conv2d_2/Relu:activations:04model/temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
+model/temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
%model/temp_time_dist_conv2d/Reshape_2Reshapetemp_inputs4model/temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:����������
(model/dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
"model/dem_time_dist_conv2d/ReshapeReshape
dem_inputs1model/dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp@model_dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
(model/dem_time_dist_conv2d/conv2d/Conv2DConv2D+model/dem_time_dist_conv2d/Reshape:output:0?model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

���
8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOpAmodel_dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)model/dem_time_dist_conv2d/conv2d/BiasAddBiasAdd1model/dem_time_dist_conv2d/conv2d/Conv2D:output:0@model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&model/dem_time_dist_conv2d/conv2d/ReluRelu2model/dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
*model/dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
$model/dem_time_dist_conv2d/Reshape_1Reshape4model/dem_time_dist_conv2d/conv2d/Relu:activations:03model/dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
*model/dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
$model/dem_time_dist_conv2d/Reshape_2Reshape
dem_inputs3model/dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:����������	�x
model/dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/dem_flatten/ReshapeReshape-model/dem_time_dist_conv2d/Reshape_1:output:0(model/dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������p
model/dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
!model/dem_flatten/flatten/ReshapeReshape"model/dem_flatten/Reshape:output:0(model/dem_flatten/flatten/Const:output:0*
T0*(
_output_shapes
:����������v
!model/dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
model/dem_flatten/Reshape_1Reshape*model/dem_flatten/flatten/Reshape:output:0*model/dem_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������z
!model/dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/dem_flatten/Reshape_2Reshape-model/dem_time_dist_conv2d/Reshape_1:output:0*model/dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������y
 model/temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/temp_flatten/ReshapeReshape.model/temp_time_dist_conv2d/Reshape_1:output:0)model/temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������s
"model/temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
$model/temp_flatten/flatten_2/ReshapeReshape#model/temp_flatten/Reshape:output:0+model/temp_flatten/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������w
"model/temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
model/temp_flatten/Reshape_1Reshape-model/temp_flatten/flatten_2/Reshape:output:0+model/temp_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������{
"model/temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/temp_flatten/Reshape_2Reshape.model/temp_time_dist_conv2d/Reshape_1:output:0+model/temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������{
"model/precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/precip_flatten/ReshapeReshape0model/precip_time_dist_conv2d/Reshape_1:output:0+model/precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������u
$model/precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
&model/precip_flatten/flatten_3/ReshapeReshape%model/precip_flatten/Reshape:output:0-model/precip_flatten/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������y
$model/precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
model/precip_flatten/Reshape_1Reshape/model/precip_flatten/flatten_3/Reshape:output:0-model/precip_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������}
$model/precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/precip_flatten/Reshape_2Reshape0model/precip_time_dist_conv2d/Reshape_1:output:0-model/precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������x
model/swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/swe_flatten/ReshapeReshape-model/swe_time_dist_conv2d/Reshape_1:output:0(model/swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������r
!model/swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
#model/swe_flatten/flatten_4/ReshapeReshape"model/swe_flatten/Reshape:output:0*model/swe_flatten/flatten_4/Const:output:0*
T0*(
_output_shapes
:����������v
!model/swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
model/swe_flatten/Reshape_1Reshape,model/swe_flatten/flatten_4/Reshape:output:0*model/swe_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������z
!model/swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/swe_flatten/Reshape_2Reshape-model/swe_time_dist_conv2d/Reshape_1:output:0*model/swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������w
model/et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/et_flatten/ReshapeReshape,model/et_time_dist_conv2d/Reshape_1:output:0'model/et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������q
 model/et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"model/et_flatten/flatten_1/ReshapeReshape!model/et_flatten/Reshape:output:0)model/et_flatten/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������u
 model/et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
model/et_flatten/Reshape_1Reshape+model/et_flatten/flatten_1/Reshape:output:0)model/et_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������y
 model/et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
model/et_flatten/Reshape_2Reshape,model/et_time_dist_conv2d/Reshape_1:output:0)model/et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2$model/dem_flatten/Reshape_1:output:0%model/temp_flatten/Reshape_1:output:0'model/precip_flatten/Reshape_1:output:0$model/swe_flatten/Reshape_1:output:0#model/et_flatten/Reshape_1:output:0&model/concatenate/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������$��
Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_encoder/multi_head_attention/query/einsum/EinsumEinsum!model/concatenate/concat:output:0Ymodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpPmodel_transformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_encoder/multi_head_attention/query/addAddV2Kmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Omodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_encoder/multi_head_attention/key/einsum/EinsumEinsum!model/concatenate/concat:output:0Wmodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpNmodel_transformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_encoder/multi_head_attention/key/addAddV2Imodel/transformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Mmodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_encoder/multi_head_attention/value/einsum/EinsumEinsum!model/concatenate/concat:output:0Ymodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpPmodel_transformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_encoder/multi_head_attention/value/addAddV2Kmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Omodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�y
4model/transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
2model/transformer_encoder/multi_head_attention/MulMul<model/transformer_encoder/multi_head_attention/query/add:z:0=model/transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
<model/transformer_encoder/multi_head_attention/einsum/EinsumEinsum:model/transformer_encoder/multi_head_attention/key/add:z:06model/transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
>model/transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmaxEmodel/transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
?model/transformer_encoder/multi_head_attention/dropout/IdentityIdentityHmodel/transformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
>model/transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumHmodel/transformer_encoder/multi_head_attention/dropout/Identity:output:0<model/transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpemodel_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Mmodel/transformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumGmodel/transformer_encoder/multi_head_attention/einsum_1/Einsum:output:0dmodel/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp[model_transformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_encoder/multi_head_attention/attention_output/addAddV2Vmodel/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Zmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
model/transformer_encoder/addAddV2!model/concatenate/concat:output:0Gmodel/transformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$��
3model/transformer_encoder/layer_normalization/ShapeShape!model/transformer_encoder/add:z:0*
T0*
_output_shapes
:�
Amodel/transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel/transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel/transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_encoder/layer_normalization/strided_sliceStridedSlice<model/transformer_encoder/layer_normalization/Shape:output:0Jmodel/transformer_encoder/layer_normalization/strided_slice/stack:output:0Lmodel/transformer_encoder/layer_normalization/strided_slice/stack_1:output:0Lmodel/transformer_encoder/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3model/transformer_encoder/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
1model/transformer_encoder/layer_normalization/mulMul<model/transformer_encoder/layer_normalization/mul/x:output:0Dmodel/transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: �
Cmodel/transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Emodel/transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel/transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model/transformer_encoder/layer_normalization/strided_slice_1StridedSlice<model/transformer_encoder/layer_normalization/Shape:output:0Lmodel/transformer_encoder/layer_normalization/strided_slice_1/stack:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3model/transformer_encoder/layer_normalization/mul_1Mul5model/transformer_encoder/layer_normalization/mul:z:0Fmodel/transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: �
Cmodel/transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Emodel/transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel/transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model/transformer_encoder/layer_normalization/strided_slice_2StridedSlice<model/transformer_encoder/layer_normalization/Shape:output:0Lmodel/transformer_encoder/layer_normalization/strided_slice_2/stack:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_2/stack_1:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5model/transformer_encoder/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
3model/transformer_encoder/layer_normalization/mul_2Mul>model/transformer_encoder/layer_normalization/mul_2/x:output:0Fmodel/transformer_encoder/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 
=model/transformer_encoder/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :
=model/transformer_encoder/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
;model/transformer_encoder/layer_normalization/Reshape/shapePackFmodel/transformer_encoder/layer_normalization/Reshape/shape/0:output:07model/transformer_encoder/layer_normalization/mul_1:z:07model/transformer_encoder/layer_normalization/mul_2:z:0Fmodel/transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
5model/transformer_encoder/layer_normalization/ReshapeReshape!model/transformer_encoder/add:z:0Dmodel/transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
9model/transformer_encoder/layer_normalization/ones/packedPack7model/transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:}
8model/transformer_encoder/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2model/transformer_encoder/layer_normalization/onesFillBmodel/transformer_encoder/layer_normalization/ones/packed:output:0Amodel/transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:����������
:model/transformer_encoder/layer_normalization/zeros/packedPack7model/transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:~
9model/transformer_encoder/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
3model/transformer_encoder/layer_normalization/zerosFillCmodel/transformer_encoder/layer_normalization/zeros/packed:output:0Bmodel/transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������v
3model/transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB x
5model/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
>model/transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV3>model/transformer_encoder/layer_normalization/Reshape:output:0;model/transformer_encoder/layer_normalization/ones:output:0<model/transformer_encoder/layer_normalization/zeros:output:0<model/transformer_encoder/layer_normalization/Const:output:0>model/transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
7model/transformer_encoder/layer_normalization/Reshape_1ReshapeBmodel/transformer_encoder/layer_normalization/FusedBatchNormV3:y:0<model/transformer_encoder/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpKmodel_transformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/transformer_encoder/layer_normalization/mul_3Mul@model/transformer_encoder/layer_normalization/Reshape_1:output:0Jmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
@model/transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpImodel_transformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model/transformer_encoder/layer_normalization/addAddV27model/transformer_encoder/layer_normalization/mul_3:z:0Hmodel/transformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
9model/transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_encoder/sequential/dense/Tensordot/ShapeShape5model/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2Cmodel/transformer_encoder/sequential/dense/Tensordot/Shape:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/free:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_encoder/sequential/dense/Tensordot/GatherV2_1GatherV2Cmodel/transformer_encoder/sequential/dense/Tensordot/Shape:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/axes:output:0Mmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_encoder/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_encoder/sequential/dense/Tensordot/ProdProdFmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0Cmodel/transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_encoder/sequential/dense/Tensordot/Prod_1ProdHmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0Emodel/transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_encoder/sequential/dense/Tensordot/concatConcatV2Bmodel/transformer_encoder/sequential/dense/Tensordot/free:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/axes:output:0Imodel/transformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_encoder/sequential/dense/Tensordot/stackPackBmodel/transformer_encoder/sequential/dense/Tensordot/Prod:output:0Dmodel/transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_encoder/sequential/dense/Tensordot/transpose	Transpose5model/transformer_encoder/layer_normalization/add:z:0Dmodel/transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
<model/transformer_encoder/sequential/dense/Tensordot/ReshapeReshapeBmodel/transformer_encoder/sequential/dense/Tensordot/transpose:y:0Cmodel/transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_encoder/sequential/dense/Tensordot/MatMulMatMulEmodel/transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<model/transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2Fmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0Emodel/transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_encoder/sequential/dense/TensordotReshapeEmodel/transformer_encoder/sequential/dense/Tensordot/MatMul:product:0Fmodel/transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2model/transformer_encoder/sequential/dense/BiasAddBiasAdd=model/transformer_encoder/sequential/dense/Tensordot:output:0Imodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$�
/model/transformer_encoder/sequential/dense/ReluRelu;model/transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpNmodel_transformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
;model/transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<model/transformer_encoder/sequential/dense_1/Tensordot/ShapeShape=model/transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:�
Dmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2Emodel/transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/free:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Fmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2Emodel/transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Omodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<model/transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_encoder/sequential/dense_1/Tensordot/ProdProdHmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Emodel/transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
>model/transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model/transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdJmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Gmodel/transformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Bmodel/transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2Dmodel/transformer_encoder/sequential/dense_1/Tensordot/free:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Kmodel/transformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_encoder/sequential/dense_1/Tensordot/stackPackDmodel/transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0Fmodel/transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@model/transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose=model/transformer_encoder/sequential/dense/Relu:activations:0Fmodel/transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
>model/transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshapeDmodel/transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0Emodel/transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
=model/transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulGmodel/transformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>model/transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Hmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Gmodel/transformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6model/transformer_encoder/sequential/dense_1/TensordotReshapeGmodel/transformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Hmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpLmodel_transformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4model/transformer_encoder/sequential/dense_1/BiasAddBiasAdd?model/transformer_encoder/sequential/dense_1/Tensordot:output:0Kmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
model/transformer_encoder/add_1AddV25model/transformer_encoder/layer_normalization/add:z:0=model/transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$��
5model/transformer_encoder/layer_normalization_1/ShapeShape#model/transformer_encoder/add_1:z:0*
T0*
_output_shapes
:�
Cmodel/transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Emodel/transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel/transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model/transformer_encoder/layer_normalization_1/strided_sliceStridedSlice>model/transformer_encoder/layer_normalization_1/Shape:output:0Lmodel/transformer_encoder/layer_normalization_1/strided_slice/stack:output:0Nmodel/transformer_encoder/layer_normalization_1/strided_slice/stack_1:output:0Nmodel/transformer_encoder/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5model/transformer_encoder/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
3model/transformer_encoder/layer_normalization_1/mulMul>model/transformer_encoder/layer_normalization_1/mul/x:output:0Fmodel/transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: �
Emodel/transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?model/transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice>model/transformer_encoder/layer_normalization_1/Shape:output:0Nmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
5model/transformer_encoder/layer_normalization_1/mul_1Mul7model/transformer_encoder/layer_normalization_1/mul:z:0Hmodel/transformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: �
Emodel/transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?model/transformer_encoder/layer_normalization_1/strided_slice_2StridedSlice>model/transformer_encoder/layer_normalization_1/Shape:output:0Nmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_1:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7model/transformer_encoder/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
5model/transformer_encoder/layer_normalization_1/mul_2Mul@model/transformer_encoder/layer_normalization_1/mul_2/x:output:0Hmodel/transformer_encoder/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: �
?model/transformer_encoder/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
?model/transformer_encoder/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
=model/transformer_encoder/layer_normalization_1/Reshape/shapePackHmodel/transformer_encoder/layer_normalization_1/Reshape/shape/0:output:09model/transformer_encoder/layer_normalization_1/mul_1:z:09model/transformer_encoder/layer_normalization_1/mul_2:z:0Hmodel/transformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
7model/transformer_encoder/layer_normalization_1/ReshapeReshape#model/transformer_encoder/add_1:z:0Fmodel/transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
;model/transformer_encoder/layer_normalization_1/ones/packedPack9model/transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:
:model/transformer_encoder/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4model/transformer_encoder/layer_normalization_1/onesFillDmodel/transformer_encoder/layer_normalization_1/ones/packed:output:0Cmodel/transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
<model/transformer_encoder/layer_normalization_1/zeros/packedPack9model/transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:�
;model/transformer_encoder/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
5model/transformer_encoder/layer_normalization_1/zerosFillEmodel/transformer_encoder/layer_normalization_1/zeros/packed:output:0Dmodel/transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������x
5model/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB z
7model/transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
@model/transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3@model/transformer_encoder/layer_normalization_1/Reshape:output:0=model/transformer_encoder/layer_normalization_1/ones:output:0>model/transformer_encoder/layer_normalization_1/zeros:output:0>model/transformer_encoder/layer_normalization_1/Const:output:0@model/transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
9model/transformer_encoder/layer_normalization_1/Reshape_1ReshapeDmodel/transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:0>model/transformer_encoder/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpMmodel_transformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5model/transformer_encoder/layer_normalization_1/mul_3MulBmodel/transformer_encoder/layer_normalization_1/Reshape_1:output:0Lmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpKmodel_transformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/transformer_encoder/layer_normalization_1/addAddV29model/transformer_encoder/layer_normalization_1/mul_3:z:0Jmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�r
0model/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model/global_max_pooling1d/MaxMax7model/transformer_encoder/layer_normalization_1/add:z:09model/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������~
model/dropout/IdentityIdentity'model/global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_2/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp8^model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp:^model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp9^model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp>^model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp=^model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp;^model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:^model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp<^model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp;^model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpA^model/transformer_encoder/layer_normalization/add/ReadVariableOpC^model/transformer_encoder/layer_normalization/mul_3/ReadVariableOpC^model/transformer_encoder/layer_normalization_1/add/ReadVariableOpE^model/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpS^model/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp]^model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpF^model/transformer_encoder/multi_head_attention/key/add/ReadVariableOpP^model/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpH^model/transformer_encoder/multi_head_attention/query/add/ReadVariableOpR^model/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpH^model/transformer_encoder/multi_head_attention/value/add/ReadVariableOpR^model/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpB^model/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpD^model/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpD^model/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpF^model/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2r
7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2v
9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp2t
8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp2~
=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp2|
<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp2x
:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp2v
9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp2z
;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp2x
:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp2�
@model/transformer_encoder/layer_normalization/add/ReadVariableOp@model/transformer_encoder/layer_normalization/add/ReadVariableOp2�
Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOpBmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp2�
Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOpBmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp2�
Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpDmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2�
Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpRmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2�
\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOpEmodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp2�
Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpOmodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOpGmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp2�
Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpQmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOpGmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp2�
Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpQmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpAmodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2�
Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpCmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2�
Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpCmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2�
Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpEmodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs
�
F
*__inference_et_flatten_layer_call_fn_47464

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44766n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48063

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47327

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
E
)__inference_flatten_2_layer_call_fn_48119

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_44588a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44291

inputs(
conv2d_3_44279:
conv2d_3_44281:
identity�� conv2d_3/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_3_44279conv2d_3_44281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44278\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_47972

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_precip_flatten_layer_call_fn_47381

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44679n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44736

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_4/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_44702\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_4/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
��
�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47771

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�@
1layer_normalization_mul_3_readvariableop_resource:	�>
/layer_normalization_add_readvariableop_resource:	�E
2sequential_dense_tensordot_readvariableop_resource:	�>
0sequential_dense_biasadd_readvariableop_resource:G
4sequential_dense_1_tensordot_readvariableop_resource:	�A
2sequential_dense_1_biasadd_readvariableop_resource:	�B
3layer_normalization_1_mul_3_readvariableop_resource:	�@
1layer_normalization_1_add_readvariableop_resource:	�
identity��&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�z
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�P
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:����������t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
 sequential/dense/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�T
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�q
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44418

inputs)
conv2d_4_44406:{�
conv2d_4_44408:
identity�� conv2d_4/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:������������
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_4_44406conv2d_4_44408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44364\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_4/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48083

inputs9
conv2d_readvariableop_resource:{�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=Sr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
7__inference_precip_time_dist_conv2d_layer_call_fn_47094

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44291�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_45402

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44246

inputs(
conv2d_2_44234:
conv2d_2_44236:
identity�� conv2d_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_2_44234conv2d_2_44236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44192\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44595

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_44588\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44538

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44531\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
E
)__inference_flatten_4_layer_call_fn_48141

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_44702a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47442

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_4/ReshapeReshapeReshape:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_4/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_47962

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45294a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47415

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_3/ReshapeReshapeReshape:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_3/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_45372

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs!
unknown:*
	unknown_0:$
	unknown_1:{�
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:��
	unknown_8:!
	unknown_9:��

unknown_10:	�"

unknown_11:��

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_45313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs
�
�
4__inference_swe_time_dist_conv2d_layer_call_fn_47169

inputs"
unknown:{�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44418�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�
�
3__inference_et_time_dist_conv2d_layer_call_fn_47226

inputs!
unknown:*
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44463�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47061

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2DConv2DReshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_46995

inputsA
%conv2d_conv2d_readvariableop_resource:��4
&conv2d_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:����������	��
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

���
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_48147

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44364

inputs9
conv2d_readvariableop_resource:{�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=Sr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
4__inference_dem_time_dist_conv2d_layer_call_fn_46962

inputs#
unknown:��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44119�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_44937

inputs
dense_44926:	�
dense_44928: 
dense_1_44931:	�
dense_1_44933:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_44926dense_44928*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44834�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44931dense_1_44933*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44870|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_48347

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44870t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44278

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_48003

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_48114

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_47957

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44709

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_4/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_44702\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_4/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_48307

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44834s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������$�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_44531

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_44834

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������$z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������$�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�=
�
E__inference_sequential_layer_call_and_return_conditional_losses_48298

inputs:
'dense_tensordot_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:<
)dense_1_tensordot_readvariableop_resource:	�6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�l
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44450

inputs8
conv2d_readvariableop_resource:*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������Oj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������Oj
 
_user_specified_nameinputs
�
�
7__inference_precip_time_dist_conv2d_layer_call_fn_47103

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44332�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
4__inference_swe_time_dist_conv2d_layer_call_fn_47160

inputs"
unknown:{�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44377�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�
�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44463

inputs(
conv2d_1_44451:*
conv2d_1_44453:
identity�� conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_44451conv2d_1_44453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44450\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_48108

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44531a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_2_layer_call_fn_48032

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44192w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_44766

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_44759\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_44989
dense_input
dense_44978:	�
dense_44980: 
dense_1_44983:	�
dense_1_44985:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_44978dense_44980*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44834�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44983dense_1_44985*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44870|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:���������$�
%
_user_specified_namedense_input
�
P
4__inference_global_max_pooling1d_layer_call_fn_47951

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_44793

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_44759\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�h
�
@__inference_model_layer_call_and_return_conditional_losses_46157

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs3
et_time_dist_conv2d_46065:*'
et_time_dist_conv2d_46067:5
swe_time_dist_conv2d_46072:{�(
swe_time_dist_conv2d_46074:7
precip_time_dist_conv2d_46079:+
precip_time_dist_conv2d_46081:5
temp_time_dist_conv2d_46086:)
temp_time_dist_conv2d_46088:6
dem_time_dist_conv2d_46093:��(
dem_time_dist_conv2d_46095:1
transformer_encoder_46116:��,
transformer_encoder_46118:	�1
transformer_encoder_46120:��,
transformer_encoder_46122:	�1
transformer_encoder_46124:��,
transformer_encoder_46126:	�1
transformer_encoder_46128:��(
transformer_encoder_46130:	�(
transformer_encoder_46132:	�(
transformer_encoder_46134:	�,
transformer_encoder_46136:	�'
transformer_encoder_46138:,
transformer_encoder_46140:	�(
transformer_encoder_46142:	�(
transformer_encoder_46144:	�(
transformer_encoder_46146:	� 
dense_2_46151:	�
dense_2_46153:
identity��,dem_time_dist_conv2d/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�+et_time_dist_conv2d/StatefulPartitionedCall�/precip_time_dist_conv2d/StatefulPartitionedCall�,swe_time_dist_conv2d/StatefulPartitionedCall�-temp_time_dist_conv2d/StatefulPartitionedCall�+transformer_encoder/StatefulPartitionedCall�
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall	et_inputset_time_dist_conv2d_46065et_time_dist_conv2d_46067*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44504z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshape	et_inputs*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
swe_inputsswe_time_dist_conv2d_46072swe_time_dist_conv2d_46074*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44418{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshape
swe_inputs+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallprecip_inputsprecip_time_dist_conv2d_46079precip_time_dist_conv2d_46081*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44332~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeprecip_inputs.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCalltemp_inputstemp_time_dist_conv2d_46086temp_time_dist_conv2d_46088*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44246|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapetemp_inputs,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
dem_inputsdem_time_dist_conv2d_46093dem_time_dist_conv2d_46095*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44160{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshape
dem_inputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44565r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44622s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44679u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44736r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44793q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_45077�
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_46116transformer_encoder_46118transformer_encoder_46120transformer_encoder_46122transformer_encoder_46124transformer_encoder_46126transformer_encoder_46128transformer_encoder_46130transformer_encoder_46132transformer_encoder_46134transformer_encoder_46136transformer_encoder_46138transformer_encoder_46140transformer_encoder_46142transformer_encoder_46144transformer_encoder_46146*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45620�
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999�
dropout/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45402�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_46151dense_2_46153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_45306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs
�
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_44702

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�E
!__inference__traced_restore_48942
file_prefix2
assignvariableop_dense_2_kernel:	�-
assignvariableop_1_dense_2_bias:J
.assignvariableop_2_dem_time_dist_conv2d_kernel:��:
,assignvariableop_3_dem_time_dist_conv2d_bias:I
/assignvariableop_4_temp_time_dist_conv2d_kernel:;
-assignvariableop_5_temp_time_dist_conv2d_bias:K
1assignvariableop_6_precip_time_dist_conv2d_kernel:=
/assignvariableop_7_precip_time_dist_conv2d_bias:I
.assignvariableop_8_swe_time_dist_conv2d_kernel:{�:
,assignvariableop_9_swe_time_dist_conv2d_bias:H
.assignvariableop_10_et_time_dist_conv2d_kernel:*:
,assignvariableop_11_et_time_dist_conv2d_bias:a
Iassignvariableop_12_transformer_encoder_multi_head_attention_query_kernel:��Z
Gassignvariableop_13_transformer_encoder_multi_head_attention_query_bias:	�_
Gassignvariableop_14_transformer_encoder_multi_head_attention_key_kernel:��X
Eassignvariableop_15_transformer_encoder_multi_head_attention_key_bias:	�a
Iassignvariableop_16_transformer_encoder_multi_head_attention_value_kernel:��Z
Gassignvariableop_17_transformer_encoder_multi_head_attention_value_bias:	�l
Tassignvariableop_18_transformer_encoder_multi_head_attention_attention_output_kernel:��a
Rassignvariableop_19_transformer_encoder_multi_head_attention_attention_output_bias:	�3
 assignvariableop_20_dense_kernel:	�,
assignvariableop_21_dense_bias:5
"assignvariableop_22_dense_1_kernel:	�/
 assignvariableop_23_dense_1_bias:	�P
Aassignvariableop_24_transformer_encoder_layer_normalization_gamma:	�O
@assignvariableop_25_transformer_encoder_layer_normalization_beta:	�R
Cassignvariableop_26_transformer_encoder_layer_normalization_1_gamma:	�Q
Bassignvariableop_27_transformer_encoder_layer_normalization_1_beta:	�'
assignvariableop_28_iteration:	 +
!assignvariableop_29_learning_rate: R
6assignvariableop_30_adam_m_dem_time_dist_conv2d_kernel:��R
6assignvariableop_31_adam_v_dem_time_dist_conv2d_kernel:��B
4assignvariableop_32_adam_m_dem_time_dist_conv2d_bias:B
4assignvariableop_33_adam_v_dem_time_dist_conv2d_bias:Q
7assignvariableop_34_adam_m_temp_time_dist_conv2d_kernel:Q
7assignvariableop_35_adam_v_temp_time_dist_conv2d_kernel:C
5assignvariableop_36_adam_m_temp_time_dist_conv2d_bias:C
5assignvariableop_37_adam_v_temp_time_dist_conv2d_bias:S
9assignvariableop_38_adam_m_precip_time_dist_conv2d_kernel:S
9assignvariableop_39_adam_v_precip_time_dist_conv2d_kernel:E
7assignvariableop_40_adam_m_precip_time_dist_conv2d_bias:E
7assignvariableop_41_adam_v_precip_time_dist_conv2d_bias:Q
6assignvariableop_42_adam_m_swe_time_dist_conv2d_kernel:{�Q
6assignvariableop_43_adam_v_swe_time_dist_conv2d_kernel:{�B
4assignvariableop_44_adam_m_swe_time_dist_conv2d_bias:B
4assignvariableop_45_adam_v_swe_time_dist_conv2d_bias:O
5assignvariableop_46_adam_m_et_time_dist_conv2d_kernel:*O
5assignvariableop_47_adam_v_et_time_dist_conv2d_kernel:*A
3assignvariableop_48_adam_m_et_time_dist_conv2d_bias:A
3assignvariableop_49_adam_v_et_time_dist_conv2d_bias:h
Passignvariableop_50_adam_m_transformer_encoder_multi_head_attention_query_kernel:��h
Passignvariableop_51_adam_v_transformer_encoder_multi_head_attention_query_kernel:��a
Nassignvariableop_52_adam_m_transformer_encoder_multi_head_attention_query_bias:	�a
Nassignvariableop_53_adam_v_transformer_encoder_multi_head_attention_query_bias:	�f
Nassignvariableop_54_adam_m_transformer_encoder_multi_head_attention_key_kernel:��f
Nassignvariableop_55_adam_v_transformer_encoder_multi_head_attention_key_kernel:��_
Lassignvariableop_56_adam_m_transformer_encoder_multi_head_attention_key_bias:	�_
Lassignvariableop_57_adam_v_transformer_encoder_multi_head_attention_key_bias:	�h
Passignvariableop_58_adam_m_transformer_encoder_multi_head_attention_value_kernel:��h
Passignvariableop_59_adam_v_transformer_encoder_multi_head_attention_value_kernel:��a
Nassignvariableop_60_adam_m_transformer_encoder_multi_head_attention_value_bias:	�a
Nassignvariableop_61_adam_v_transformer_encoder_multi_head_attention_value_bias:	�s
[assignvariableop_62_adam_m_transformer_encoder_multi_head_attention_attention_output_kernel:��s
[assignvariableop_63_adam_v_transformer_encoder_multi_head_attention_attention_output_kernel:��h
Yassignvariableop_64_adam_m_transformer_encoder_multi_head_attention_attention_output_bias:	�h
Yassignvariableop_65_adam_v_transformer_encoder_multi_head_attention_attention_output_bias:	�:
'assignvariableop_66_adam_m_dense_kernel:	�:
'assignvariableop_67_adam_v_dense_kernel:	�3
%assignvariableop_68_adam_m_dense_bias:3
%assignvariableop_69_adam_v_dense_bias:<
)assignvariableop_70_adam_m_dense_1_kernel:	�<
)assignvariableop_71_adam_v_dense_1_kernel:	�6
'assignvariableop_72_adam_m_dense_1_bias:	�6
'assignvariableop_73_adam_v_dense_1_bias:	�W
Hassignvariableop_74_adam_m_transformer_encoder_layer_normalization_gamma:	�W
Hassignvariableop_75_adam_v_transformer_encoder_layer_normalization_gamma:	�V
Gassignvariableop_76_adam_m_transformer_encoder_layer_normalization_beta:	�V
Gassignvariableop_77_adam_v_transformer_encoder_layer_normalization_beta:	�Y
Jassignvariableop_78_adam_m_transformer_encoder_layer_normalization_1_gamma:	�Y
Jassignvariableop_79_adam_v_transformer_encoder_layer_normalization_1_gamma:	�X
Iassignvariableop_80_adam_m_transformer_encoder_layer_normalization_1_beta:	�X
Iassignvariableop_81_adam_v_transformer_encoder_layer_normalization_1_beta:	�<
)assignvariableop_82_adam_m_dense_2_kernel:	�<
)assignvariableop_83_adam_v_dense_2_kernel:	�5
'assignvariableop_84_adam_m_dense_2_bias:5
'assignvariableop_85_adam_v_dense_2_bias:#
assignvariableop_86_total: #
assignvariableop_87_count: 
identity_89��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_9�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�!
value�!B�!YB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*�
value�B�YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_dem_time_dist_conv2d_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_dem_time_dist_conv2d_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_temp_time_dist_conv2d_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_temp_time_dist_conv2d_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_precip_time_dist_conv2d_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp/assignvariableop_7_precip_time_dist_conv2d_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_swe_time_dist_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_swe_time_dist_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_et_time_dist_conv2d_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_et_time_dist_conv2d_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_encoder_multi_head_attention_query_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpGassignvariableop_13_transformer_encoder_multi_head_attention_query_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpGassignvariableop_14_transformer_encoder_multi_head_attention_key_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpEassignvariableop_15_transformer_encoder_multi_head_attention_key_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpIassignvariableop_16_transformer_encoder_multi_head_attention_value_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpGassignvariableop_17_transformer_encoder_multi_head_attention_value_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpTassignvariableop_18_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpRassignvariableop_19_transformer_encoder_multi_head_attention_attention_output_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpAassignvariableop_24_transformer_encoder_layer_normalization_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp@assignvariableop_25_transformer_encoder_layer_normalization_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_encoder_layer_normalization_1_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpBassignvariableop_27_transformer_encoder_layer_normalization_1_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_iterationIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_m_dem_time_dist_conv2d_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_v_dem_time_dist_conv2d_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_m_dem_time_dist_conv2d_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_v_dem_time_dist_conv2d_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_m_temp_time_dist_conv2d_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_v_temp_time_dist_conv2d_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_m_temp_time_dist_conv2d_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_v_temp_time_dist_conv2d_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp9assignvariableop_38_adam_m_precip_time_dist_conv2d_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp9assignvariableop_39_adam_v_precip_time_dist_conv2d_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_m_precip_time_dist_conv2d_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_v_precip_time_dist_conv2d_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_m_swe_time_dist_conv2d_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_v_swe_time_dist_conv2d_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_m_swe_time_dist_conv2d_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_v_swe_time_dist_conv2d_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_m_et_time_dist_conv2d_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_v_et_time_dist_conv2d_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_m_et_time_dist_conv2d_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_v_et_time_dist_conv2d_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpPassignvariableop_50_adam_m_transformer_encoder_multi_head_attention_query_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpPassignvariableop_51_adam_v_transformer_encoder_multi_head_attention_query_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpNassignvariableop_52_adam_m_transformer_encoder_multi_head_attention_query_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpNassignvariableop_53_adam_v_transformer_encoder_multi_head_attention_query_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpNassignvariableop_54_adam_m_transformer_encoder_multi_head_attention_key_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpNassignvariableop_55_adam_v_transformer_encoder_multi_head_attention_key_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpLassignvariableop_56_adam_m_transformer_encoder_multi_head_attention_key_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpLassignvariableop_57_adam_v_transformer_encoder_multi_head_attention_key_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_m_transformer_encoder_multi_head_attention_value_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpPassignvariableop_59_adam_v_transformer_encoder_multi_head_attention_value_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpNassignvariableop_60_adam_m_transformer_encoder_multi_head_attention_value_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpNassignvariableop_61_adam_v_transformer_encoder_multi_head_attention_value_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp[assignvariableop_62_adam_m_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp[assignvariableop_63_adam_v_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpYassignvariableop_64_adam_m_transformer_encoder_multi_head_attention_attention_output_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpYassignvariableop_65_adam_v_transformer_encoder_multi_head_attention_attention_output_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_m_dense_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_v_dense_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_m_dense_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp%assignvariableop_69_adam_v_dense_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_m_dense_1_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_v_dense_1_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_m_dense_1_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_v_dense_1_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpHassignvariableop_74_adam_m_transformer_encoder_layer_normalization_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpHassignvariableop_75_adam_v_transformer_encoder_layer_normalization_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpGassignvariableop_76_adam_m_transformer_encoder_layer_normalization_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpGassignvariableop_77_adam_v_transformer_encoder_layer_normalization_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpJassignvariableop_78_adam_m_transformer_encoder_layer_normalization_1_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpJassignvariableop_79_adam_v_transformer_encoder_layer_normalization_1_gammaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpIassignvariableop_80_adam_m_transformer_encoder_layer_normalization_1_betaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpIassignvariableop_81_adam_v_transformer_encoder_layer_normalization_1_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_m_dense_2_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_v_dense_2_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_m_dense_2_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_v_dense_2_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_totalIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpassignvariableop_87_countIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44652

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_3/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_44645\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_3/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
F__inference_concatenate_layer_call_and_return_conditional_losses_47522
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*,
_output_shapes
:���������$�\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������$�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_4
�
k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_48092

inputs!
unknown:*
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44450w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������Oj: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������Oj
 
_user_specified_nameinputs
�
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47354

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_2/ReshapeReshapeReshape:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_2/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_44588

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_temp_flatten_layer_call_fn_47337

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44622n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47193

inputsB
'conv2d_4_conv2d_readvariableop_resource:{�6
(conv2d_4_biasadd_readvariableop_resource:
identity��conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:������������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
conv2d_4/Conv2DConv2DReshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=S�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_4/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_45306

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47085

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2DConv2DReshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�=
�
E__inference_sequential_layer_call_and_return_conditional_losses_48241

inputs:
'dense_tensordot_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:<
)dense_1_tensordot_readvariableop_resource:	�6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�l
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44332

inputs(
conv2d_3_44320:
conv2d_3_44322:
identity�� conv2d_3/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_3_44320conv2d_3_44322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44278\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�g
�
@__inference_model_layer_call_and_return_conditional_losses_46058

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs3
et_time_dist_conv2d_45966:*'
et_time_dist_conv2d_45968:5
swe_time_dist_conv2d_45973:{�(
swe_time_dist_conv2d_45975:7
precip_time_dist_conv2d_45980:+
precip_time_dist_conv2d_45982:5
temp_time_dist_conv2d_45987:)
temp_time_dist_conv2d_45989:6
dem_time_dist_conv2d_45994:��(
dem_time_dist_conv2d_45996:1
transformer_encoder_46017:��,
transformer_encoder_46019:	�1
transformer_encoder_46021:��,
transformer_encoder_46023:	�1
transformer_encoder_46025:��,
transformer_encoder_46027:	�1
transformer_encoder_46029:��(
transformer_encoder_46031:	�(
transformer_encoder_46033:	�(
transformer_encoder_46035:	�,
transformer_encoder_46037:	�'
transformer_encoder_46039:,
transformer_encoder_46041:	�(
transformer_encoder_46043:	�(
transformer_encoder_46045:	�(
transformer_encoder_46047:	� 
dense_2_46052:	�
dense_2_46054:
identity��,dem_time_dist_conv2d/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+et_time_dist_conv2d/StatefulPartitionedCall�/precip_time_dist_conv2d/StatefulPartitionedCall�,swe_time_dist_conv2d/StatefulPartitionedCall�-temp_time_dist_conv2d/StatefulPartitionedCall�+transformer_encoder/StatefulPartitionedCall�
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall	et_inputset_time_dist_conv2d_45966et_time_dist_conv2d_45968*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44463z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshape	et_inputs*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
swe_inputsswe_time_dist_conv2d_45973swe_time_dist_conv2d_45975*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44377{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshape
swe_inputs+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallprecip_inputsprecip_time_dist_conv2d_45980precip_time_dist_conv2d_45982*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44291~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeprecip_inputs.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCalltemp_inputstemp_time_dist_conv2d_45987temp_time_dist_conv2d_45989*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44205|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapetemp_inputs,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
dem_inputsdem_time_dist_conv2d_45994dem_time_dist_conv2d_45996*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44119{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshape
dem_inputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44538r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44595s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44652u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44709r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44766q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_45077�
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_46017transformer_encoder_46019transformer_encoder_46021transformer_encoder_46023transformer_encoder_46025transformer_encoder_46027transformer_encoder_46029transformer_encoder_46031transformer_encoder_46033transformer_encoder_46035transformer_encoder_46037transformer_encoder_46039transformer_encoder_46041transformer_encoder_46043transformer_encoder_46045transformer_encoder_46047*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45254�
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999�
dropout/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45294�
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_46052dense_2_46054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_45306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_48136

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47127

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:
identity��conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_3/Conv2DConv2DReshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_46226

dem_inputs
	et_inputs
precip_inputs

swe_inputs
temp_inputs!
unknown:*
	unknown_0:$
	unknown_1:{�
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:��
	unknown_8:!
	unknown_9:��

unknown_10:	�"

unknown_11:��

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_44081o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������Oj:���������:�����������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:����������	�
$
_user_specified_name
dem_inputs:^Z
3
_output_shapes!
:���������Oj
#
_user_specified_name	et_inputs:b^
3
_output_shapes!
:���������
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:�����������
$
_user_specified_name
swe_inputs:`\
3
_output_shapes!
:���������
%
_user_specified_nametemp_inputs
�
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44565

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_44531\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
&__inference_conv2d_layer_call_fn_48012

inputs#
unknown:��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_44106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������	�: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:����������	�
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_48171

inputs
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44877t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
3__inference_transformer_encoder_layer_call_fn_47559

inputs
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45254t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������$�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
5__inference_temp_time_dist_conv2d_layer_call_fn_47028

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44205�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_48377

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������$�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_44106

inputs:
conv2d_readvariableop_resource:��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

��r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������	�
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_44877

inputs
dense_44835:	�
dense_44837: 
dense_1_44871:	�
dense_1_44873:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_44835dense_44837*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44834�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44871dense_1_44873*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44870|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�f
�
@__inference_model_layer_call_and_return_conditional_losses_45313

inputs
inputs_1
inputs_2
inputs_3
inputs_43
et_time_dist_conv2d_45017:*'
et_time_dist_conv2d_45019:5
swe_time_dist_conv2d_45024:{�(
swe_time_dist_conv2d_45026:7
precip_time_dist_conv2d_45031:+
precip_time_dist_conv2d_45033:5
temp_time_dist_conv2d_45038:)
temp_time_dist_conv2d_45040:6
dem_time_dist_conv2d_45045:��(
dem_time_dist_conv2d_45047:1
transformer_encoder_45255:��,
transformer_encoder_45257:	�1
transformer_encoder_45259:��,
transformer_encoder_45261:	�1
transformer_encoder_45263:��,
transformer_encoder_45265:	�1
transformer_encoder_45267:��(
transformer_encoder_45269:	�(
transformer_encoder_45271:	�(
transformer_encoder_45273:	�,
transformer_encoder_45275:	�'
transformer_encoder_45277:,
transformer_encoder_45279:	�(
transformer_encoder_45281:	�(
transformer_encoder_45283:	�(
transformer_encoder_45285:	� 
dense_2_45307:	�
dense_2_45309:
identity��,dem_time_dist_conv2d/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+et_time_dist_conv2d/StatefulPartitionedCall�/precip_time_dist_conv2d/StatefulPartitionedCall�,swe_time_dist_conv2d/StatefulPartitionedCall�-temp_time_dist_conv2d/StatefulPartitionedCall�+transformer_encoder/StatefulPartitionedCall�
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_4et_time_dist_conv2d_45017et_time_dist_conv2d_45019*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44463z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_3swe_time_dist_conv2d_45024swe_time_dist_conv2d_45026*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_44377{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_2precip_time_dist_conv2d_45031precip_time_dist_conv2d_45033*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_44291~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1temp_time_dist_conv2d_45038temp_time_dist_conv2d_45040*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44205|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsdem_time_dist_conv2d_45045dem_time_dist_conv2d_45047*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44119{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshapeinputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44538r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44595s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_44652u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_44709r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44766q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_45077�
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_45255transformer_encoder_45257transformer_encoder_45259transformer_encoder_45261transformer_encoder_45263transformer_encoder_45265transformer_encoder_45267transformer_encoder_45269transformer_encoder_45271transformer_encoder_45273transformer_encoder_45275transformer_encoder_45277transformer_encoder_45279transformer_encoder_45281transformer_encoder_45283transformer_encoder_45285*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_45254�
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_44999�
dropout/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_45294�
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_45307dense_2_45309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_45306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:] Y
5
_output_shapes#
!:����������	�
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:�����������
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������Oj
 
_user_specified_nameinputs
�
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47398

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_3/ReshapeReshapeReshape:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_3/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_44759

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_44975
dense_input
dense_44964:	�
dense_44966: 
dense_1_44969:	�
dense_1_44971:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_44964dense_44966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44834�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_44969dense_1_44971*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44870|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:���������$�
%
_user_specified_namedense_input
�
�
(__inference_conv2d_3_layer_call_fn_48052

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44278w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_dem_flatten_layer_call_fn_47288

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_44538n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
3__inference_et_time_dist_conv2d_layer_call_fn_47235

inputs!
unknown:*
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44504�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
E
)__inference_flatten_1_layer_call_fn_48152

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_44759a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_44205

inputs(
conv2d_2_44193:
conv2d_2_44195:
identity�� conv2d_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_2_44193conv2d_2_44195*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44192\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47283

inputsA
'conv2d_1_conv2d_readvariableop_resource:*6
(conv2d_1_biasadd_readvariableop_resource:
identity��conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_48338

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������$z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������$�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48043

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47217

inputsB
'conv2d_4_conv2d_readvariableop_resource:{�6
(conv2d_4_biasadd_readvariableop_resource:
identity��conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:������������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
conv2d_4/Conv2DConv2DReshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=S�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeconv2d_4/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&�������������������
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(��������������������: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(��������������������
 
_user_specified_nameinputs
�
�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_44504

inputs(
conv2d_1_44492:*
conv2d_1_44494:
identity�� conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_44492conv2d_1_44494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44450\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&������������������v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&������������������Oj: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&������������������Oj
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_45294

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44622

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:����������
flatten_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_44588\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape"flatten_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_44192

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_44645

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_48184

inputs
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������$�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44937t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������$�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������$�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������$�
 
_user_specified_nameinputs
�
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_48125

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47459

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_4/ReshapeReshapeReshape:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_4/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
��
�"
@__inference_model_layer_call_and_return_conditional_losses_46651
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4U
;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:*J
<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:W
<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:{�K
=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:Y
?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:N
@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:W
=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:L
>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:V
:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:��I
;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:l
Ttransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:��]
Jtransformer_encoder_multi_head_attention_query_add_readvariableop_resource:	�j
Rtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:��[
Htransformer_encoder_multi_head_attention_key_add_readvariableop_resource:	�l
Ttransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:��]
Jtransformer_encoder_multi_head_attention_value_add_readvariableop_resource:	�w
_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��d
Utransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:	�T
Etransformer_encoder_layer_normalization_mul_3_readvariableop_resource:	�R
Ctransformer_encoder_layer_normalization_add_readvariableop_resource:	�Y
Ftransformer_encoder_sequential_dense_tensordot_readvariableop_resource:	�R
Dtransformer_encoder_sequential_dense_biasadd_readvariableop_resource:[
Htransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource:	�U
Ftransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:	�V
Gtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:	�T
Etransformer_encoder_layer_normalization_1_add_readvariableop_resource:	�9
&dense_2_matmul_readvariableop_resource:	�5
'dense_2_biasadd_readvariableop_resource:
identity��2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp�1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp�2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp�7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp�6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp�4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp�3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp�5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp�4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp�:transformer_encoder/layer_normalization/add/ReadVariableOp�<transformer_encoder/layer_normalization/mul_3/ReadVariableOp�<transformer_encoder/layer_normalization_1/add/ReadVariableOp�>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp�Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp�Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�?transformer_encoder/multi_head_attention/key/add/ReadVariableOp�Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Atransformer_encoder/multi_head_attention/query/add/ReadVariableOp�Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Atransformer_encoder/multi_head_attention/value/add/ReadVariableOp�Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp�;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp�=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp�=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp�?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpz
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Oj�
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:**
dtype0�
#et_time_dist_conv2d/conv2d_1/Conv2DConv2D$et_time_dist_conv2d/Reshape:output:0:et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd,et_time_dist_conv2d/conv2d_1/Conv2D:output:0;et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!et_time_dist_conv2d/conv2d_1/ReluRelu-et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:����������
#et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
et_time_dist_conv2d/Reshape_1Reshape/et_time_dist_conv2d/conv2d_1/Relu:activations:0,et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������|
#et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����O   j      �
et_time_dist_conv2d/Reshape_2Reshapeinputs_4,et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������Oj{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:������������
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:{�*
dtype0�
$swe_time_dist_conv2d/conv2d_4/Conv2DConv2D%swe_time_dist_conv2d/Reshape:output:0;swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
=S�
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd-swe_time_dist_conv2d/conv2d_4/Conv2D:output:0<swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"swe_time_dist_conv2d/conv2d_4/ReluRelu.swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
swe_time_dist_conv2d/Reshape_1Reshape0swe_time_dist_conv2d/conv2d_4/Relu:activations:0-swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������}
$swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����5  �     �
swe_time_dist_conv2d/Reshape_2Reshapeinputs_3-swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:�����������~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
'precip_time_dist_conv2d/conv2d_3/Conv2DConv2D(precip_time_dist_conv2d/Reshape:output:0>precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
(precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd0precip_time_dist_conv2d/conv2d_3/Conv2D:output:0?precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
%precip_time_dist_conv2d/conv2d_3/ReluRelu1precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
'precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
!precip_time_dist_conv2d/Reshape_1Reshape3precip_time_dist_conv2d/conv2d_3/Relu:activations:00precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:����������
'precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
!precip_time_dist_conv2d/Reshape_2Reshapeinputs_20precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%temp_time_dist_conv2d/conv2d_2/Conv2DConv2D&temp_time_dist_conv2d/Reshape:output:0<temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd.temp_time_dist_conv2d/conv2d_2/Conv2D:output:0=temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
#temp_time_dist_conv2d/conv2d_2/ReluRelu/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:����������
%temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
temp_time_dist_conv2d/Reshape_1Reshape1temp_time_dist_conv2d/conv2d_2/Relu:activations:0.temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������~
%temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_time_dist_conv2d/Reshape_2Reshapeinputs_1.temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/ReshapeReshapeinputs_0+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:����������	��
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
"dem_time_dist_conv2d/conv2d/Conv2DConv2D%dem_time_dist_conv2d/Reshape:output:09dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides

���
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOp;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#dem_time_dist_conv2d/conv2d/BiasAddBiasAdd+dem_time_dist_conv2d/conv2d/Conv2D:output:0:dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 dem_time_dist_conv2d/conv2d/ReluRelu,dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"����            �
dem_time_dist_conv2d/Reshape_1Reshape.dem_time_dist_conv2d/conv2d/Relu:activations:0-dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:���������}
$dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"�����  `     �
dem_time_dist_conv2d/Reshape_2Reshapeinputs_0-dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:����������	�r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/ReshapeReshape'dem_time_dist_conv2d/Reshape_1:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������j
dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
dem_flatten/flatten/ReshapeReshapedem_flatten/Reshape:output:0"dem_flatten/flatten/Const:output:0*
T0*(
_output_shapes
:����������p
dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
dem_flatten/Reshape_1Reshape$dem_flatten/flatten/Reshape:output:0$dem_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������t
dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
dem_flatten/Reshape_2Reshape'dem_time_dist_conv2d/Reshape_1:output:0$dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/ReshapeReshape(temp_time_dist_conv2d/Reshape_1:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������m
temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
temp_flatten/flatten_2/ReshapeReshapetemp_flatten/Reshape:output:0%temp_flatten/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������q
temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
temp_flatten/Reshape_1Reshape'temp_flatten/flatten_2/Reshape:output:0%temp_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������u
temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
temp_flatten/Reshape_2Reshape(temp_time_dist_conv2d/Reshape_1:output:0%temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/ReshapeReshape*precip_time_dist_conv2d/Reshape_1:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������o
precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
 precip_flatten/flatten_3/ReshapeReshapeprecip_flatten/Reshape:output:0'precip_flatten/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������s
precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
precip_flatten/Reshape_1Reshape)precip_flatten/flatten_3/Reshape:output:0'precip_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������w
precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
precip_flatten/Reshape_2Reshape*precip_time_dist_conv2d/Reshape_1:output:0'precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/ReshapeReshape'swe_time_dist_conv2d/Reshape_1:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������l
swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
swe_flatten/flatten_4/ReshapeReshapeswe_flatten/Reshape:output:0$swe_flatten/flatten_4/Const:output:0*
T0*(
_output_shapes
:����������p
swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
swe_flatten/Reshape_1Reshape&swe_flatten/flatten_4/Reshape:output:0$swe_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������t
swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
swe_flatten/Reshape_2Reshape'swe_time_dist_conv2d/Reshape_1:output:0$swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/ReshapeReshape&et_time_dist_conv2d/Reshape_1:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:���������k
et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
et_flatten/flatten_1/ReshapeReshapeet_flatten/Reshape:output:0#et_flatten/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������o
et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   �   �
et_flatten/Reshape_1Reshape%et_flatten/flatten_1/Reshape:output:0#et_flatten/Reshape_1/shape:output:0*
T0*,
_output_shapes
:����������s
et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
et_flatten/Reshape_2Reshape&et_time_dist_conv2d/Reshape_1:output:0#et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2dem_flatten/Reshape_1:output:0temp_flatten/Reshape_1:output:0!precip_flatten/Reshape_1:output:0swe_flatten/Reshape_1:output:0et_flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������$��
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_encoder/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_encoder/multi_head_attention/query/addAddV2Etransformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_encoder/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
?transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpHtransformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_encoder/multi_head_attention/key/addAddV2Ctransformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Gtransformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$��
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_encoder/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������$�*
equationabc,cde->abde�
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_encoder/multi_head_attention/value/addAddV2Etransformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������$�s
.transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_encoder/multi_head_attention/MulMul6transformer_encoder/multi_head_attention/query/add:z:07transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������$��
6transformer_encoder/multi_head_attention/einsum/EinsumEinsum4transformer_encoder/multi_head_attention/key/add:z:00transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������$$*
equationaecd,abcd->acbe�
8transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmax?transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������$$�
9transformer_encoder/multi_head_attention/dropout/IdentityIdentityBtransformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������$$�
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumBtransformer_encoder/multi_head_attention/dropout/Identity:output:06transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������$�*
equationacbe,aecd->abcd�
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������$�*
equationabcd,cde->abe�
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpUtransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_encoder/multi_head_attention/attention_output/addAddV2Ptransformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Ttransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
transformer_encoder/addAddV2concatenate/concat:output:0Atransformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������$�x
-transformer_encoder/layer_normalization/ShapeShapetransformer_encoder/add:z:0*
T0*
_output_shapes
:�
;transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5transformer_encoder/layer_normalization/strided_sliceStridedSlice6transformer_encoder/layer_normalization/Shape:output:0Dtransformer_encoder/layer_normalization/strided_slice/stack:output:0Ftransformer_encoder/layer_normalization/strided_slice/stack_1:output:0Ftransformer_encoder/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-transformer_encoder/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
+transformer_encoder/layer_normalization/mulMul6transformer_encoder/layer_normalization/mul/x:output:0>transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: �
=transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization/strided_slice_1StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_1/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-transformer_encoder/layer_normalization/mul_1Mul/transformer_encoder/layer_normalization/mul:z:0@transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: �
=transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization/strided_slice_2StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_2/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_2/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/transformer_encoder/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
-transformer_encoder/layer_normalization/mul_2Mul8transformer_encoder/layer_normalization/mul_2/x:output:0@transformer_encoder/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: y
7transformer_encoder/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :y
7transformer_encoder/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
5transformer_encoder/layer_normalization/Reshape/shapePack@transformer_encoder/layer_normalization/Reshape/shape/0:output:01transformer_encoder/layer_normalization/mul_1:z:01transformer_encoder/layer_normalization/mul_2:z:0@transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
/transformer_encoder/layer_normalization/ReshapeReshapetransformer_encoder/add:z:0>transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
3transformer_encoder/layer_normalization/ones/packedPack1transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:w
2transformer_encoder/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,transformer_encoder/layer_normalization/onesFill<transformer_encoder/layer_normalization/ones/packed:output:0;transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:����������
4transformer_encoder/layer_normalization/zeros/packedPack1transformer_encoder/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:x
3transformer_encoder/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-transformer_encoder/layer_normalization/zerosFill=transformer_encoder/layer_normalization/zeros/packed:output:0<transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������p
-transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB r
/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
8transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV38transformer_encoder/layer_normalization/Reshape:output:05transformer_encoder/layer_normalization/ones:output:06transformer_encoder/layer_normalization/zeros:output:06transformer_encoder/layer_normalization/Const:output:08transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
1transformer_encoder/layer_normalization/Reshape_1Reshape<transformer_encoder/layer_normalization/FusedBatchNormV3:y:06transformer_encoder/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:���������$��
<transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-transformer_encoder/layer_normalization/mul_3Mul:transformer_encoder/layer_normalization/Reshape_1:output:0Dtransformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
:transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpCtransformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+transformer_encoder/layer_normalization/addAddV21transformer_encoder/layer_normalization/mul_3:z:0Btransformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0}
3transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_encoder/sequential/dense/Tensordot/ShapeShape/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:~
<transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/free:output:0Etransformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense/Tensordot/GatherV2_1GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Gtransformer_encoder/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_encoder/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_encoder/sequential/dense/Tensordot/ProdProd@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0=transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
6transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_encoder/sequential/dense/Tensordot/Prod_1ProdBtransformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0?transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_encoder/sequential/dense/Tensordot/concatConcatV2<transformer_encoder/sequential/dense/Tensordot/free:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Ctransformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_encoder/sequential/dense/Tensordot/stackPack<transformer_encoder/sequential/dense/Tensordot/Prod:output:0>transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_encoder/sequential/dense/Tensordot/transpose	Transpose/transformer_encoder/layer_normalization/add:z:0>transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������$��
6transformer_encoder/sequential/dense/Tensordot/ReshapeReshape<transformer_encoder/sequential/dense/Tensordot/transpose:y:0=transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_encoder/sequential/dense/Tensordot/MatMulMatMul?transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Etransformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Etransformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_encoder/sequential/dense/TensordotReshape?transformer_encoder/sequential/dense/Tensordot/MatMul:product:0@transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������$�
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpDtransformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
,transformer_encoder/sequential/dense/BiasAddBiasAdd7transformer_encoder/sequential/dense/Tensordot:output:0Ctransformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������$�
)transformer_encoder/sequential/dense/ReluRelu5transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������$�
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpHtransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0
5transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_encoder/sequential/dense_1/Tensordot/ShapeShape7transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:�
>transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/free:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Itransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_encoder/sequential/dense_1/Tensordot/ProdProdBtransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
8transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdDtransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2>transformer_encoder/sequential/dense_1/Tensordot/free:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Etransformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_encoder/sequential/dense_1/Tensordot/stackPack>transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0@transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose7transformer_encoder/sequential/dense/Relu:activations:0@transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������$�
8transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshape>transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0?transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulAtransformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Btransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_encoder/sequential/dense_1/TensordotReshapeAtransformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Btransformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������$��
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_encoder/sequential/dense_1/BiasAddBiasAdd9transformer_encoder/sequential/dense_1/Tensordot:output:0Etransformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
transformer_encoder/add_1AddV2/transformer_encoder/layer_normalization/add:z:07transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������$�|
/transformer_encoder/layer_normalization_1/ShapeShapetransformer_encoder/add_1:z:0*
T0*
_output_shapes
:�
=transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7transformer_encoder/layer_normalization_1/strided_sliceStridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Ftransformer_encoder/layer_normalization_1/strided_slice/stack:output:0Htransformer_encoder/layer_normalization_1/strided_slice/stack_1:output:0Htransformer_encoder/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/transformer_encoder/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
-transformer_encoder/layer_normalization_1/mulMul8transformer_encoder/layer_normalization_1/mul/x:output:0@transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: �
?transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
/transformer_encoder/layer_normalization_1/mul_1Mul1transformer_encoder/layer_normalization_1/mul:z:0Btransformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: �
?transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9transformer_encoder/layer_normalization_1/strided_slice_2StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_2/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_2/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_encoder/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
/transformer_encoder/layer_normalization_1/mul_2Mul:transformer_encoder/layer_normalization_1/mul_2/x:output:0Btransformer_encoder/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: {
9transformer_encoder/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :{
9transformer_encoder/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
7transformer_encoder/layer_normalization_1/Reshape/shapePackBtransformer_encoder/layer_normalization_1/Reshape/shape/0:output:03transformer_encoder/layer_normalization_1/mul_1:z:03transformer_encoder/layer_normalization_1/mul_2:z:0Btransformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
1transformer_encoder/layer_normalization_1/ReshapeReshapetransformer_encoder/add_1:z:0@transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
5transformer_encoder/layer_normalization_1/ones/packedPack3transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:y
4transformer_encoder/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.transformer_encoder/layer_normalization_1/onesFill>transformer_encoder/layer_normalization_1/ones/packed:output:0=transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
6transformer_encoder/layer_normalization_1/zeros/packedPack3transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:z
5transformer_encoder/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/transformer_encoder/layer_normalization_1/zerosFill?transformer_encoder/layer_normalization_1/zeros/packed:output:0>transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������r
/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB t
1transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
:transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3:transformer_encoder/layer_normalization_1/Reshape:output:07transformer_encoder/layer_normalization_1/ones:output:08transformer_encoder/layer_normalization_1/zeros:output:08transformer_encoder/layer_normalization_1/Const:output:0:transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
3transformer_encoder/layer_normalization_1/Reshape_1Reshape>transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:08transformer_encoder/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:���������$��
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpGtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/transformer_encoder/layer_normalization_1/mul_3Mul<transformer_encoder/layer_normalization_1/Reshape_1:output:0Ftransformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$��
<transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-transformer_encoder/layer_normalization_1/addAddV23transformer_encoder/layer_normalization_1/mul_3:z:0Dtransformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������$�l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d/MaxMax1transformer_encoder/layer_normalization_1/add:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������r
dropout/IdentityIdentity!global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_2/MatMulMatMuldropout/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2^dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp4^et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3^et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp8^precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7^precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp5^swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4^swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp6^temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5^temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp;^transformer_encoder/layer_normalization/add/ReadVariableOp=^transformer_encoder/layer_normalization/mul_3/ReadVariableOp=^transformer_encoder/layer_normalization_1/add/ReadVariableOp?^transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpM^transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpW^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp@^transformer_encoder/multi_head_attention/key/add/ReadVariableOpJ^transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/query/add/ReadVariableOpL^transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/value/add/ReadVariableOpL^transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp<^transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp>^transformer_encoder/sequential/dense/Tensordot/ReadVariableOp>^transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp@^transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������	�:���������:���������:�����������:���������Oj: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2f
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2j
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp2h
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp2r
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp2p
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp2l
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp2j
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp2n
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp2l
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp2x
:transformer_encoder/layer_normalization/add/ReadVariableOp:transformer_encoder/layer_normalization/add/ReadVariableOp2|
<transformer_encoder/layer_normalization/mul_3/ReadVariableOp<transformer_encoder/layer_normalization/mul_3/ReadVariableOp2|
<transformer_encoder/layer_normalization_1/add/ReadVariableOp<transformer_encoder/layer_normalization_1/add/ReadVariableOp2�
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2�
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpLtransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2�
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpVtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
?transformer_encoder/multi_head_attention/key/add/ReadVariableOp?transformer_encoder/multi_head_attention/key/add/ReadVariableOp2�
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpItransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpAtransformer_encoder/multi_head_attention/query/add/ReadVariableOp2�
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpAtransformer_encoder/multi_head_attention/value/add/ReadVariableOp2�
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2z
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2~
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2�
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:_ [
5
_output_shapes#
!:����������	�
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:���������
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:�����������
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:���������Oj
"
_user_specified_name
inputs_4
�
H
,__inference_temp_flatten_layer_call_fn_47332

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_44595n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
4__inference_dem_time_dist_conv2d_layer_call_fn_46971

inputs#
unknown:��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_44160�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(�������������������	�: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(�������������������	�
 
_user_specified_nameinputs
�
F
*__inference_et_flatten_layer_call_fn_47469

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_44793n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47371

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   {
flatten_2/ReshapeReshapeReshape:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapeflatten_2/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&������������������:d `
<
_output_shapes*
(:&������������������
 
_user_specified_nameinputs
�
�
F__inference_concatenate_layer_call_and_return_conditional_losses_45077

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*,
_output_shapes
:���������$�\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������$�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x:����������:����������:����������:����������:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O

dem_inputsA
serving_default_dem_inputs:0����������	�
K
	et_inputs>
serving_default_et_inputs:0���������Oj
S
precip_inputsB
serving_default_precip_inputs:0���������
O

swe_inputsA
serving_default_swe_inputs:0�����������
O
temp_inputs@
serving_default_temp_inputs:0���������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2layer"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
	@layer"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Glayer"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
	Nlayer"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
	Ulayer"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
	clayer"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p	attention
q
dense_proj
rlayernorm_1
slayernorm_2"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_45372
%__inference_model_layer_call_fn_46291
%__inference_model_layer_call_fn_46356
%__inference_model_layer_call_fn_45959�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_46651
@__inference_model_layer_call_and_return_conditional_losses_46953
@__inference_model_layer_call_and_return_conditional_losses_46058
@__inference_model_layer_call_and_return_conditional_losses_46157�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_44081
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_dem_time_dist_conv2d_layer_call_fn_46962
4__inference_dem_time_dist_conv2d_layer_call_fn_46971�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_46995
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_47019�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_temp_time_dist_conv2d_layer_call_fn_47028
5__inference_temp_time_dist_conv2d_layer_call_fn_47037�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47061
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47085�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_precip_time_dist_conv2d_layer_call_fn_47094
7__inference_precip_time_dist_conv2d_layer_call_fn_47103�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47127
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47151�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_swe_time_dist_conv2d_layer_call_fn_47160
4__inference_swe_time_dist_conv2d_layer_call_fn_47169�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47193
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47217�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_et_time_dist_conv2d_layer_call_fn_47226
3__inference_et_time_dist_conv2d_layer_call_fn_47235�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47259
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47283�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dem_flatten_layer_call_fn_47288
+__inference_dem_flatten_layer_call_fn_47293�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47310
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47327�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_temp_flatten_layer_call_fn_47332
,__inference_temp_flatten_layer_call_fn_47337�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47354
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47371�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_precip_flatten_layer_call_fn_47376
.__inference_precip_flatten_layer_call_fn_47381�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47398
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47415�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_swe_flatten_layer_call_fn_47420
+__inference_swe_flatten_layer_call_fn_47425�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47442
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47459�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_et_flatten_layer_call_fn_47464
*__inference_et_flatten_layer_call_fn_47469�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_et_flatten_layer_call_and_return_conditional_losses_47486
E__inference_et_flatten_layer_call_and_return_conditional_losses_47503�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_concatenate_layer_call_fn_47512�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_concatenate_layer_call_and_return_conditional_losses_47522�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_transformer_encoder_layer_call_fn_47559
3__inference_transformer_encoder_layer_call_fn_47596�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47771
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47946�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_global_max_pooling1d_layer_call_fn_47951�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_47957�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_dropout_layer_call_fn_47962
'__inference_dropout_layer_call_fn_47967�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_dropout_layer_call_and_return_conditional_losses_47972
B__inference_dropout_layer_call_and_return_conditional_losses_47984�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_2_layer_call_fn_47993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_2_layer_call_and_return_conditional_losses_48003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�2dense_2/kernel
:2dense_2/bias
7:5��2dem_time_dist_conv2d/kernel
':%2dem_time_dist_conv2d/bias
6:42temp_time_dist_conv2d/kernel
(:&2temp_time_dist_conv2d/bias
8:62precip_time_dist_conv2d/kernel
*:(2precip_time_dist_conv2d/bias
6:4{�2swe_time_dist_conv2d/kernel
':%2swe_time_dist_conv2d/bias
4:2*2et_time_dist_conv2d/kernel
&:$2et_time_dist_conv2d/bias
M:K��25transformer_encoder/multi_head_attention/query/kernel
F:D	�23transformer_encoder/multi_head_attention/query/bias
K:I��23transformer_encoder/multi_head_attention/key/kernel
D:B	�21transformer_encoder/multi_head_attention/key/bias
M:K��25transformer_encoder/multi_head_attention/value/kernel
F:D	�23transformer_encoder/multi_head_attention/value/bias
X:V��2@transformer_encoder/multi_head_attention/attention_output/kernel
M:K�2>transformer_encoder/multi_head_attention/attention_output/bias
:	�2dense/kernel
:2
dense/bias
!:	�2dense_1/kernel
:�2dense_1/bias
<::�2-transformer_encoder/layer_normalization/gamma
;:9�2,transformer_encoder/layer_normalization/beta
>:<�2/transformer_encoder/layer_normalization_1/gamma
=:;�2.transformer_encoder/layer_normalization_1/beta
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_45372
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_46291inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_46356inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_45959
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_46651inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_46953inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_46058
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_46157
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_46226
dem_inputs	et_inputsprecip_inputs
swe_inputstemp_inputs"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_dem_time_dist_conv2d_layer_call_fn_46962inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_dem_time_dist_conv2d_layer_call_fn_46971inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_46995inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_47019inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv2d_layer_call_fn_48012�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv2d_layer_call_and_return_conditional_losses_48023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_temp_time_dist_conv2d_layer_call_fn_47028inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_temp_time_dist_conv2d_layer_call_fn_47037inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47061inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47085inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_2_layer_call_fn_48032�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_precip_time_dist_conv2d_layer_call_fn_47094inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_precip_time_dist_conv2d_layer_call_fn_47103inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47127inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47151inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_3_layer_call_fn_48052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48063�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_swe_time_dist_conv2d_layer_call_fn_47160inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_swe_time_dist_conv2d_layer_call_fn_47169inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47193inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47217inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_4_layer_call_fn_48072�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_et_time_dist_conv2d_layer_call_fn_47226inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_et_time_dist_conv2d_layer_call_fn_47235inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47259inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47283inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_1_layer_call_fn_48092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48103�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dem_flatten_layer_call_fn_47288inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dem_flatten_layer_call_fn_47293inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47310inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47327inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_48108�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_48114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_temp_flatten_layer_call_fn_47332inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_temp_flatten_layer_call_fn_47337inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47354inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47371inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_2_layer_call_fn_48119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_2_layer_call_and_return_conditional_losses_48125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
U0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_precip_flatten_layer_call_fn_47376inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_precip_flatten_layer_call_fn_47381inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47398inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47415inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_3_layer_call_fn_48130�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_3_layer_call_and_return_conditional_losses_48136�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_swe_flatten_layer_call_fn_47420inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_swe_flatten_layer_call_fn_47425inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47442inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47459inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_4_layer_call_fn_48141�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_4_layer_call_and_return_conditional_losses_48147�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_et_flatten_layer_call_fn_47464inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_et_flatten_layer_call_fn_47469inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_et_flatten_layer_call_and_return_conditional_losses_47486inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_et_flatten_layer_call_and_return_conditional_losses_47503inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_1_layer_call_fn_48152�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_1_layer_call_and_return_conditional_losses_48158�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_concatenate_layer_call_fn_47512inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_concatenate_layer_call_and_return_conditional_losses_47522inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_transformer_encoder_layer_call_fn_47559inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_transformer_encoder_layer_call_fn_47596inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47771inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47946inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
*__inference_sequential_layer_call_fn_44888
*__inference_sequential_layer_call_fn_48171
*__inference_sequential_layer_call_fn_48184
*__inference_sequential_layer_call_fn_44961�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_48241
E__inference_sequential_layer_call_and_return_conditional_losses_48298
E__inference_sequential_layer_call_and_return_conditional_losses_44975
E__inference_sequential_layer_call_and_return_conditional_losses_44989�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_global_max_pooling1d_layer_call_fn_47951inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_47957inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dropout_layer_call_fn_47962inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_47967inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_47972inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_47984inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_2_layer_call_fn_47993inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_2_layer_call_and_return_conditional_losses_48003inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
<::��2"Adam/m/dem_time_dist_conv2d/kernel
<::��2"Adam/v/dem_time_dist_conv2d/kernel
,:*2 Adam/m/dem_time_dist_conv2d/bias
,:*2 Adam/v/dem_time_dist_conv2d/bias
;:92#Adam/m/temp_time_dist_conv2d/kernel
;:92#Adam/v/temp_time_dist_conv2d/kernel
-:+2!Adam/m/temp_time_dist_conv2d/bias
-:+2!Adam/v/temp_time_dist_conv2d/bias
=:;2%Adam/m/precip_time_dist_conv2d/kernel
=:;2%Adam/v/precip_time_dist_conv2d/kernel
/:-2#Adam/m/precip_time_dist_conv2d/bias
/:-2#Adam/v/precip_time_dist_conv2d/bias
;:9{�2"Adam/m/swe_time_dist_conv2d/kernel
;:9{�2"Adam/v/swe_time_dist_conv2d/kernel
,:*2 Adam/m/swe_time_dist_conv2d/bias
,:*2 Adam/v/swe_time_dist_conv2d/bias
9:7*2!Adam/m/et_time_dist_conv2d/kernel
9:7*2!Adam/v/et_time_dist_conv2d/kernel
+:)2Adam/m/et_time_dist_conv2d/bias
+:)2Adam/v/et_time_dist_conv2d/bias
R:P��2<Adam/m/transformer_encoder/multi_head_attention/query/kernel
R:P��2<Adam/v/transformer_encoder/multi_head_attention/query/kernel
K:I	�2:Adam/m/transformer_encoder/multi_head_attention/query/bias
K:I	�2:Adam/v/transformer_encoder/multi_head_attention/query/bias
P:N��2:Adam/m/transformer_encoder/multi_head_attention/key/kernel
P:N��2:Adam/v/transformer_encoder/multi_head_attention/key/kernel
I:G	�28Adam/m/transformer_encoder/multi_head_attention/key/bias
I:G	�28Adam/v/transformer_encoder/multi_head_attention/key/bias
R:P��2<Adam/m/transformer_encoder/multi_head_attention/value/kernel
R:P��2<Adam/v/transformer_encoder/multi_head_attention/value/kernel
K:I	�2:Adam/m/transformer_encoder/multi_head_attention/value/bias
K:I	�2:Adam/v/transformer_encoder/multi_head_attention/value/bias
]:[��2GAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel
]:[��2GAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel
R:P�2EAdam/m/transformer_encoder/multi_head_attention/attention_output/bias
R:P�2EAdam/v/transformer_encoder/multi_head_attention/attention_output/bias
$:"	�2Adam/m/dense/kernel
$:"	�2Adam/v/dense/kernel
:2Adam/m/dense/bias
:2Adam/v/dense/bias
&:$	�2Adam/m/dense_1/kernel
&:$	�2Adam/v/dense_1/kernel
 :�2Adam/m/dense_1/bias
 :�2Adam/v/dense_1/bias
A:?�24Adam/m/transformer_encoder/layer_normalization/gamma
A:?�24Adam/v/transformer_encoder/layer_normalization/gamma
@:>�23Adam/m/transformer_encoder/layer_normalization/beta
@:>�23Adam/v/transformer_encoder/layer_normalization/beta
C:A�26Adam/m/transformer_encoder/layer_normalization_1/gamma
C:A�26Adam/v/transformer_encoder/layer_normalization_1/gamma
B:@�25Adam/m/transformer_encoder/layer_normalization_1/beta
B:@�25Adam/v/transformer_encoder/layer_normalization_1/beta
&:$	�2Adam/m/dense_2/kernel
&:$	�2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv2d_layer_call_fn_48012inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv2d_layer_call_and_return_conditional_losses_48023inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_2_layer_call_fn_48032inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48043inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_3_layer_call_fn_48052inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48063inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_4_layer_call_fn_48072inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48083inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_1_layer_call_fn_48092inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48103inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_48108inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_48114inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_2_layer_call_fn_48119inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_2_layer_call_and_return_conditional_losses_48125inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_3_layer_call_fn_48130inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_3_layer_call_and_return_conditional_losses_48136inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_4_layer_call_fn_48141inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_4_layer_call_and_return_conditional_losses_48147inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_1_layer_call_fn_48152inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_1_layer_call_and_return_conditional_losses_48158inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_48307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_48338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_48347�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_48377�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_44888dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_48171inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_48184inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_44961dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_48241inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_48298inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_44975dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_44989dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_48307inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_48338inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_48347inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_48377inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_44081�8�������������������������������
���
���
2�/

dem_inputs����������	�
1�.
temp_inputs���������
3�0
precip_inputs���������
2�/

swe_inputs�����������
/�,
	et_inputs���������Oj
� "1�.
,
dense_2!�
dense_2����������
F__inference_concatenate_layer_call_and_return_conditional_losses_47522����
���
���
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
'�$
inputs_3����������
'�$
inputs_4����������
� "1�.
'�$
tensor_0���������$�
� �
+__inference_concatenate_layer_call_fn_47512����
���
���
'�$
inputs_0����������
'�$
inputs_1����������
'�$
inputs_2����������
'�$
inputs_3����������
'�$
inputs_4����������
� "&�#
unknown���������$��
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48103u��7�4
-�*
(�%
inputs���������Oj
� "4�1
*�'
tensor_0���������
� �
(__inference_conv2d_1_layer_call_fn_48092j��7�4
-�*
(�%
inputs���������Oj
� ")�&
unknown����������
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48043u��7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
(__inference_conv2d_2_layer_call_fn_48032j��7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
C__inference_conv2d_3_layer_call_and_return_conditional_losses_48063u��7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
(__inference_conv2d_3_layer_call_fn_48052j��7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
C__inference_conv2d_4_layer_call_and_return_conditional_losses_48083w��9�6
/�,
*�'
inputs�����������
� "4�1
*�'
tensor_0���������
� �
(__inference_conv2d_4_layer_call_fn_48072l��9�6
/�,
*�'
inputs�����������
� ")�&
unknown����������
A__inference_conv2d_layer_call_and_return_conditional_losses_48023w��9�6
/�,
*�'
inputs����������	�
� "4�1
*�'
tensor_0���������
� �
&__inference_conv2d_layer_call_fn_48012l��9�6
/�,
*�'
inputs����������	�
� ")�&
unknown����������
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47310�L�I
B�?
5�2
inputs&������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
F__inference_dem_flatten_layer_call_and_return_conditional_losses_47327�L�I
B�?
5�2
inputs&������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
+__inference_dem_flatten_layer_call_fn_47288L�I
B�?
5�2
inputs&������������������
p 

 
� "/�,
unknown��������������������
+__inference_dem_flatten_layer_call_fn_47293L�I
B�?
5�2
inputs&������������������
p

 
� "/�,
unknown��������������������
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_46995���N�K
D�A
7�4
inputs(�������������������	�
p 

 
� "A�>
7�4
tensor_0&������������������
� �
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_47019���N�K
D�A
7�4
inputs(�������������������	�
p

 
� "A�>
7�4
tensor_0&������������������
� �
4__inference_dem_time_dist_conv2d_layer_call_fn_46962���N�K
D�A
7�4
inputs(�������������������	�
p 

 
� "6�3
unknown&�������������������
4__inference_dem_time_dist_conv2d_layer_call_fn_46971���N�K
D�A
7�4
inputs(�������������������	�
p

 
� "6�3
unknown&�������������������
B__inference_dense_1_layer_call_and_return_conditional_losses_48377n��3�0
)�&
$�!
inputs���������$
� "1�.
'�$
tensor_0���������$�
� �
'__inference_dense_1_layer_call_fn_48347c��3�0
)�&
$�!
inputs���������$
� "&�#
unknown���������$��
B__inference_dense_2_layer_call_and_return_conditional_losses_48003f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
'__inference_dense_2_layer_call_fn_47993[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
@__inference_dense_layer_call_and_return_conditional_losses_48338n��4�1
*�'
%�"
inputs���������$�
� "0�-
&�#
tensor_0���������$
� �
%__inference_dense_layer_call_fn_48307c��4�1
*�'
%�"
inputs���������$�
� "%�"
unknown���������$�
B__inference_dropout_layer_call_and_return_conditional_losses_47972e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_47984e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
'__inference_dropout_layer_call_fn_47962Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
'__inference_dropout_layer_call_fn_47967Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
E__inference_et_flatten_layer_call_and_return_conditional_losses_47486�L�I
B�?
5�2
inputs&������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
E__inference_et_flatten_layer_call_and_return_conditional_losses_47503�L�I
B�?
5�2
inputs&������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
*__inference_et_flatten_layer_call_fn_47464L�I
B�?
5�2
inputs&������������������
p 

 
� "/�,
unknown��������������������
*__inference_et_flatten_layer_call_fn_47469L�I
B�?
5�2
inputs&������������������
p

 
� "/�,
unknown��������������������
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47259���L�I
B�?
5�2
inputs&������������������Oj
p 

 
� "A�>
7�4
tensor_0&������������������
� �
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_47283���L�I
B�?
5�2
inputs&������������������Oj
p

 
� "A�>
7�4
tensor_0&������������������
� �
3__inference_et_time_dist_conv2d_layer_call_fn_47226���L�I
B�?
5�2
inputs&������������������Oj
p 

 
� "6�3
unknown&�������������������
3__inference_et_time_dist_conv2d_layer_call_fn_47235���L�I
B�?
5�2
inputs&������������������Oj
p

 
� "6�3
unknown&�������������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_48158h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_flatten_1_layer_call_fn_48152]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
D__inference_flatten_2_layer_call_and_return_conditional_losses_48125h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_flatten_2_layer_call_fn_48119]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
D__inference_flatten_3_layer_call_and_return_conditional_losses_48136h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_flatten_3_layer_call_fn_48130]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
D__inference_flatten_4_layer_call_and_return_conditional_losses_48147h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_flatten_4_layer_call_fn_48141]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
B__inference_flatten_layer_call_and_return_conditional_losses_48114h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
'__inference_flatten_layer_call_fn_48108]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_47957~E�B
;�8
6�3
inputs'���������������������������
� "5�2
+�(
tensor_0������������������
� �
4__inference_global_max_pooling1d_layer_call_fn_47951sE�B
;�8
6�3
inputs'���������������������������
� "*�'
unknown�������������������
@__inference_model_layer_call_and_return_conditional_losses_46058�8�������������������������������
���
���
2�/

dem_inputs����������	�
1�.
temp_inputs���������
3�0
precip_inputs���������
2�/

swe_inputs�����������
/�,
	et_inputs���������Oj
p 

 
� ",�)
"�
tensor_0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_46157�8�������������������������������
���
���
2�/

dem_inputs����������	�
1�.
temp_inputs���������
3�0
precip_inputs���������
2�/

swe_inputs�����������
/�,
	et_inputs���������Oj
p

 
� ",�)
"�
tensor_0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_46651�8�������������������������������
���
���
0�-
inputs_0����������	�
.�+
inputs_1���������
.�+
inputs_2���������
0�-
inputs_3�����������
.�+
inputs_4���������Oj
p 

 
� ",�)
"�
tensor_0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_46953�8�������������������������������
���
���
0�-
inputs_0����������	�
.�+
inputs_1���������
.�+
inputs_2���������
0�-
inputs_3�����������
.�+
inputs_4���������Oj
p

 
� ",�)
"�
tensor_0���������
� �
%__inference_model_layer_call_fn_45372�8�������������������������������
���
���
2�/

dem_inputs����������	�
1�.
temp_inputs���������
3�0
precip_inputs���������
2�/

swe_inputs�����������
/�,
	et_inputs���������Oj
p 

 
� "!�
unknown����������
%__inference_model_layer_call_fn_45959�8�������������������������������
���
���
2�/

dem_inputs����������	�
1�.
temp_inputs���������
3�0
precip_inputs���������
2�/

swe_inputs�����������
/�,
	et_inputs���������Oj
p

 
� "!�
unknown����������
%__inference_model_layer_call_fn_46291�8�������������������������������
���
���
0�-
inputs_0����������	�
.�+
inputs_1���������
.�+
inputs_2���������
0�-
inputs_3�����������
.�+
inputs_4���������Oj
p 

 
� "!�
unknown����������
%__inference_model_layer_call_fn_46356�8�������������������������������
���
���
0�-
inputs_0����������	�
.�+
inputs_1���������
.�+
inputs_2���������
0�-
inputs_3�����������
.�+
inputs_4���������Oj
p

 
� "!�
unknown����������
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47398�L�I
B�?
5�2
inputs&������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
I__inference_precip_flatten_layer_call_and_return_conditional_losses_47415�L�I
B�?
5�2
inputs&������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
.__inference_precip_flatten_layer_call_fn_47376L�I
B�?
5�2
inputs&������������������
p 

 
� "/�,
unknown��������������������
.__inference_precip_flatten_layer_call_fn_47381L�I
B�?
5�2
inputs&������������������
p

 
� "/�,
unknown��������������������
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47127���L�I
B�?
5�2
inputs&������������������
p 

 
� "A�>
7�4
tensor_0&������������������
� �
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_47151���L�I
B�?
5�2
inputs&������������������
p

 
� "A�>
7�4
tensor_0&������������������
� �
7__inference_precip_time_dist_conv2d_layer_call_fn_47094���L�I
B�?
5�2
inputs&������������������
p 

 
� "6�3
unknown&�������������������
7__inference_precip_time_dist_conv2d_layer_call_fn_47103���L�I
B�?
5�2
inputs&������������������
p

 
� "6�3
unknown&�������������������
E__inference_sequential_layer_call_and_return_conditional_losses_44975�����A�>
7�4
*�'
dense_input���������$�
p 

 
� "1�.
'�$
tensor_0���������$�
� �
E__inference_sequential_layer_call_and_return_conditional_losses_44989�����A�>
7�4
*�'
dense_input���������$�
p

 
� "1�.
'�$
tensor_0���������$�
� �
E__inference_sequential_layer_call_and_return_conditional_losses_48241{����<�9
2�/
%�"
inputs���������$�
p 

 
� "1�.
'�$
tensor_0���������$�
� �
E__inference_sequential_layer_call_and_return_conditional_losses_48298{����<�9
2�/
%�"
inputs���������$�
p

 
� "1�.
'�$
tensor_0���������$�
� �
*__inference_sequential_layer_call_fn_44888u����A�>
7�4
*�'
dense_input���������$�
p 

 
� "&�#
unknown���������$��
*__inference_sequential_layer_call_fn_44961u����A�>
7�4
*�'
dense_input���������$�
p

 
� "&�#
unknown���������$��
*__inference_sequential_layer_call_fn_48171p����<�9
2�/
%�"
inputs���������$�
p 

 
� "&�#
unknown���������$��
*__inference_sequential_layer_call_fn_48184p����<�9
2�/
%�"
inputs���������$�
p

 
� "&�#
unknown���������$��
#__inference_signature_wrapper_46226�8�������������������������������
� 
���
@

dem_inputs2�/

dem_inputs����������	�
<
	et_inputs/�,
	et_inputs���������Oj
D
precip_inputs3�0
precip_inputs���������
@

swe_inputs2�/

swe_inputs�����������
@
temp_inputs1�.
temp_inputs���������"1�.
,
dense_2!�
dense_2����������
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47442�L�I
B�?
5�2
inputs&������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
F__inference_swe_flatten_layer_call_and_return_conditional_losses_47459�L�I
B�?
5�2
inputs&������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
+__inference_swe_flatten_layer_call_fn_47420L�I
B�?
5�2
inputs&������������������
p 

 
� "/�,
unknown��������������������
+__inference_swe_flatten_layer_call_fn_47425L�I
B�?
5�2
inputs&������������������
p

 
� "/�,
unknown��������������������
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47193���N�K
D�A
7�4
inputs(��������������������
p 

 
� "A�>
7�4
tensor_0&������������������
� �
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_47217���N�K
D�A
7�4
inputs(��������������������
p

 
� "A�>
7�4
tensor_0&������������������
� �
4__inference_swe_time_dist_conv2d_layer_call_fn_47160���N�K
D�A
7�4
inputs(��������������������
p 

 
� "6�3
unknown&�������������������
4__inference_swe_time_dist_conv2d_layer_call_fn_47169���N�K
D�A
7�4
inputs(��������������������
p

 
� "6�3
unknown&�������������������
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47354�L�I
B�?
5�2
inputs&������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
G__inference_temp_flatten_layer_call_and_return_conditional_losses_47371�L�I
B�?
5�2
inputs&������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
,__inference_temp_flatten_layer_call_fn_47332L�I
B�?
5�2
inputs&������������������
p 

 
� "/�,
unknown��������������������
,__inference_temp_flatten_layer_call_fn_47337L�I
B�?
5�2
inputs&������������������
p

 
� "/�,
unknown��������������������
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47061���L�I
B�?
5�2
inputs&������������������
p 

 
� "A�>
7�4
tensor_0&������������������
� �
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_47085���L�I
B�?
5�2
inputs&������������������
p

 
� "A�>
7�4
tensor_0&������������������
� �
5__inference_temp_time_dist_conv2d_layer_call_fn_47028���L�I
B�?
5�2
inputs&������������������
p 

 
� "6�3
unknown&�������������������
5__inference_temp_time_dist_conv2d_layer_call_fn_47037���L�I
B�?
5�2
inputs&������������������
p

 
� "6�3
unknown&�������������������
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47771� ����������������H�E
.�+
%�"
inputs���������$�

 
�

trainingp "1�.
'�$
tensor_0���������$�
� �
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_47946� ����������������H�E
.�+
%�"
inputs���������$�

 
�

trainingp"1�.
'�$
tensor_0���������$�
� �
3__inference_transformer_encoder_layer_call_fn_47559� ����������������H�E
.�+
%�"
inputs���������$�

 
�

trainingp "&�#
unknown���������$��
3__inference_transformer_encoder_layer_call_fn_47596� ����������������H�E
.�+
%�"
inputs���������$�

 
�

trainingp"&�#
unknown���������$�