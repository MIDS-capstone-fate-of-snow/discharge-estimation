ÁŽ2
Ç
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
ű
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
epsilonfloat%ˇŃ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
Ž
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

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.0-dev202210222v1.12.1-83490-gb1d4c35fba28¸´+
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

Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

:
*
dtype0

Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

:
*
dtype0
Â
5Adam/v/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*F
shared_name75Adam/v/transformer_encoder/layer_normalization_1/beta
ť
IAdam/v/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp5Adam/v/transformer_encoder/layer_normalization_1/beta*
_output_shapes
:
*
dtype0
Â
5Adam/m/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*F
shared_name75Adam/m/transformer_encoder/layer_normalization_1/beta
ť
IAdam/m/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp5Adam/m/transformer_encoder/layer_normalization_1/beta*
_output_shapes
:
*
dtype0
Ä
6Adam/v/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*G
shared_name86Adam/v/transformer_encoder/layer_normalization_1/gamma
˝
JAdam/v/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp6Adam/v/transformer_encoder/layer_normalization_1/gamma*
_output_shapes
:
*
dtype0
Ä
6Adam/m/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*G
shared_name86Adam/m/transformer_encoder/layer_normalization_1/gamma
˝
JAdam/m/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp6Adam/m/transformer_encoder/layer_normalization_1/gamma*
_output_shapes
:
*
dtype0
ž
3Adam/v/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53Adam/v/transformer_encoder/layer_normalization/beta
ˇ
GAdam/v/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp3Adam/v/transformer_encoder/layer_normalization/beta*
_output_shapes
:
*
dtype0
ž
3Adam/m/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53Adam/m/transformer_encoder/layer_normalization/beta
ˇ
GAdam/m/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp3Adam/m/transformer_encoder/layer_normalization/beta*
_output_shapes
:
*
dtype0
Ŕ
4Adam/v/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*E
shared_name64Adam/v/transformer_encoder/layer_normalization/gamma
š
HAdam/v/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp4Adam/v/transformer_encoder/layer_normalization/gamma*
_output_shapes
:
*
dtype0
Ŕ
4Adam/m/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*E
shared_name64Adam/m/transformer_encoder/layer_normalization/gamma
š
HAdam/m/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp4Adam/m/transformer_encoder/layer_normalization/gamma*
_output_shapes
:
*
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:
*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:
*
dtype0

Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

: 
*
dtype0

Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

: 
*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
: *
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
: *
dtype0

Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:
 *
dtype0

Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:
 *
dtype0
â
EAdam/v/transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*V
shared_nameGEAdam/v/transformer_encoder/multi_head_attention/attention_output/bias
Ű
YAdam/v/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpEAdam/v/transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes
:
*
dtype0
â
EAdam/m/transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*V
shared_nameGEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias
Ű
YAdam/m/transformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes
:
*
dtype0
î
GAdam/v/transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*X
shared_nameIGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel
ç
[Adam/v/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel*"
_output_shapes
:

*
dtype0
î
GAdam/m/transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*X
shared_nameIGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel
ç
[Adam/m/transformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel*"
_output_shapes
:

*
dtype0
Đ
:Adam/v/transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/value/bias
É
NAdam/v/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/value/bias*
_output_shapes

:
*
dtype0
Đ
:Adam/m/transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/value/bias
É
NAdam/m/transformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/value/bias*
_output_shapes

:
*
dtype0
Ř
<Adam/v/transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*M
shared_name><Adam/v/transformer_encoder/multi_head_attention/value/kernel
Ń
PAdam/v/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp<Adam/v/transformer_encoder/multi_head_attention/value/kernel*"
_output_shapes
:

*
dtype0
Ř
<Adam/m/transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*M
shared_name><Adam/m/transformer_encoder/multi_head_attention/value/kernel
Ń
PAdam/m/transformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp<Adam/m/transformer_encoder/multi_head_attention/value/kernel*"
_output_shapes
:

*
dtype0
Ě
8Adam/v/transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*I
shared_name:8Adam/v/transformer_encoder/multi_head_attention/key/bias
Ĺ
LAdam/v/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp8Adam/v/transformer_encoder/multi_head_attention/key/bias*
_output_shapes

:
*
dtype0
Ě
8Adam/m/transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*I
shared_name:8Adam/m/transformer_encoder/multi_head_attention/key/bias
Ĺ
LAdam/m/transformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp8Adam/m/transformer_encoder/multi_head_attention/key/bias*
_output_shapes

:
*
dtype0
Ô
:Adam/v/transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/key/kernel
Í
NAdam/v/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/key/kernel*"
_output_shapes
:

*
dtype0
Ô
:Adam/m/transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/key/kernel
Í
NAdam/m/transformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/key/kernel*"
_output_shapes
:

*
dtype0
Đ
:Adam/v/transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:Adam/v/transformer_encoder/multi_head_attention/query/bias
É
NAdam/v/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp:Adam/v/transformer_encoder/multi_head_attention/query/bias*
_output_shapes

:
*
dtype0
Đ
:Adam/m/transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:Adam/m/transformer_encoder/multi_head_attention/query/bias
É
NAdam/m/transformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp:Adam/m/transformer_encoder/multi_head_attention/query/bias*
_output_shapes

:
*
dtype0
Ř
<Adam/v/transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*M
shared_name><Adam/v/transformer_encoder/multi_head_attention/query/kernel
Ń
PAdam/v/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp<Adam/v/transformer_encoder/multi_head_attention/query/kernel*"
_output_shapes
:

*
dtype0
Ř
<Adam/m/transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*M
shared_name><Adam/m/transformer_encoder/multi_head_attention/query/kernel
Ń
PAdam/m/transformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp<Adam/m/transformer_encoder/multi_head_attention/query/kernel*"
_output_shapes
:

*
dtype0

Adam/v/et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/et_time_dist_conv2d/bias

3Adam/v/et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/et_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

Adam/m/et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/et_time_dist_conv2d/bias

3Adam/m/et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/et_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
Ś
!Adam/v/et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:!'*2
shared_name#!Adam/v/et_time_dist_conv2d/kernel

5Adam/v/et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/et_time_dist_conv2d/kernel*&
_output_shapes
:!'*
dtype0
Ś
!Adam/m/et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:!'*2
shared_name#!Adam/m/et_time_dist_conv2d/kernel

5Adam/m/et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/et_time_dist_conv2d/kernel*&
_output_shapes
:!'*
dtype0

 Adam/v/swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/swe_time_dist_conv2d/bias

4Adam/v/swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/v/swe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

 Adam/m/swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/swe_time_dist_conv2d/bias

4Adam/m/swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/m/swe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
Ş
"Adam/v/swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/swe_time_dist_conv2d/kernel
Ł
6Adam/v/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/swe_time_dist_conv2d/kernel*(
_output_shapes
:*
dtype0
Ş
"Adam/m/swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/swe_time_dist_conv2d/kernel
Ł
6Adam/m/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/swe_time_dist_conv2d/kernel*(
_output_shapes
:*
dtype0

#Adam/v/precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/precip_time_dist_conv2d/bias

7Adam/v/precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/v/precip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

#Adam/m/precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/precip_time_dist_conv2d/bias

7Adam/m/precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/m/precip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
Ž
%Adam/v/precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/precip_time_dist_conv2d/kernel
§
9Adam/v/precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/v/precip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
Ž
%Adam/m/precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/precip_time_dist_conv2d/kernel
§
9Adam/m/precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/m/precip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0

!Adam/v/temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/temp_time_dist_conv2d/bias

5Adam/v/temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp!Adam/v/temp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

!Adam/m/temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/temp_time_dist_conv2d/bias

5Adam/m/temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp!Adam/m/temp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
Ş
#Adam/v/temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/temp_time_dist_conv2d/kernel
Ł
7Adam/v/temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp#Adam/v/temp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0
Ş
#Adam/m/temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/temp_time_dist_conv2d/kernel
Ł
7Adam/m/temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp#Adam/m/temp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0

 Adam/v/dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/dem_time_dist_conv2d/bias

4Adam/v/dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/v/dem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

 Adam/m/dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/dem_time_dist_conv2d/bias

4Adam/m/dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/m/dem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
Ş
"Adam/v/dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:úŢ*3
shared_name$"Adam/v/dem_time_dist_conv2d/kernel
Ł
6Adam/v/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/dem_time_dist_conv2d/kernel*(
_output_shapes
:úŢ*
dtype0
Ş
"Adam/m/dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:úŢ*3
shared_name$"Adam/m/dem_time_dist_conv2d/kernel
Ł
6Adam/m/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/dem_time_dist_conv2d/kernel*(
_output_shapes
:úŢ*
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
´
.transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.transformer_encoder/layer_normalization_1/beta
­
Btransformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp.transformer_encoder/layer_normalization_1/beta*
_output_shapes
:
*
dtype0
ś
/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/transformer_encoder/layer_normalization_1/gamma
Ż
Ctransformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp/transformer_encoder/layer_normalization_1/gamma*
_output_shapes
:
*
dtype0
°
,transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,transformer_encoder/layer_normalization/beta
Š
@transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp,transformer_encoder/layer_normalization/beta*
_output_shapes
:
*
dtype0
˛
-transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-transformer_encoder/layer_normalization/gamma
Ť
Atransformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp-transformer_encoder/layer_normalization/gamma*
_output_shapes
:
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: 
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
 *
dtype0
Ô
>transformer_encoder/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*O
shared_name@>transformer_encoder/multi_head_attention/attention_output/bias
Í
Rtransformer_encoder/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_encoder/multi_head_attention/attention_output/bias*
_output_shapes
:
*
dtype0
ŕ
@transformer_encoder/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*Q
shared_nameB@transformer_encoder/multi_head_attention/attention_output/kernel
Ů
Ttransformer_encoder/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_encoder/multi_head_attention/attention_output/kernel*"
_output_shapes
:

*
dtype0
Â
3transformer_encoder/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*D
shared_name53transformer_encoder/multi_head_attention/value/bias
ť
Gtransformer_encoder/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/value/bias*
_output_shapes

:
*
dtype0
Ę
5transformer_encoder/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*F
shared_name75transformer_encoder/multi_head_attention/value/kernel
Ă
Itransformer_encoder/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_encoder/multi_head_attention/value/kernel*"
_output_shapes
:

*
dtype0
ž
1transformer_encoder/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31transformer_encoder/multi_head_attention/key/bias
ˇ
Etransformer_encoder/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp1transformer_encoder/multi_head_attention/key/bias*
_output_shapes

:
*
dtype0
Ć
3transformer_encoder/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*D
shared_name53transformer_encoder/multi_head_attention/key/kernel
ż
Gtransformer_encoder/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/key/kernel*"
_output_shapes
:

*
dtype0
Â
3transformer_encoder/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*D
shared_name53transformer_encoder/multi_head_attention/query/bias
ť
Gtransformer_encoder/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp3transformer_encoder/multi_head_attention/query/bias*
_output_shapes

:
*
dtype0
Ę
5transformer_encoder/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*F
shared_name75transformer_encoder/multi_head_attention/query/kernel
Ă
Itransformer_encoder/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_encoder/multi_head_attention/query/kernel*"
_output_shapes
:

*
dtype0

et_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameet_time_dist_conv2d/bias

,et_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpet_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

et_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:!'*+
shared_nameet_time_dist_conv2d/kernel

.et_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpet_time_dist_conv2d/kernel*&
_output_shapes
:!'*
dtype0

swe_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameswe_time_dist_conv2d/bias

-swe_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpswe_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

swe_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameswe_time_dist_conv2d/kernel

/swe_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpswe_time_dist_conv2d/kernel*(
_output_shapes
:*
dtype0

precip_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameprecip_time_dist_conv2d/bias

0precip_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpprecip_time_dist_conv2d/bias*
_output_shapes
:*
dtype0
 
precip_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name precip_time_dist_conv2d/kernel

2precip_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpprecip_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0

temp_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametemp_time_dist_conv2d/bias

.temp_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOptemp_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

temp_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametemp_time_dist_conv2d/kernel

0temp_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOptemp_time_dist_conv2d/kernel*&
_output_shapes
:*
dtype0

dem_time_dist_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedem_time_dist_conv2d/bias

-dem_time_dist_conv2d/bias/Read/ReadVariableOpReadVariableOpdem_time_dist_conv2d/bias*
_output_shapes
:*
dtype0

dem_time_dist_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:úŢ*,
shared_namedem_time_dist_conv2d/kernel

/dem_time_dist_conv2d/kernel/Read/ReadVariableOpReadVariableOpdem_time_dist_conv2d/kernel*(
_output_shapes
:úŢ*
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
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
*
dtype0

serving_default_dem_inputsPlaceholder*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷*
dtype0**
shape!:˙˙˙˙˙˙˙˙˙÷

serving_default_et_inputsPlaceholder*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w*
dtype0*(
shape:˙˙˙˙˙˙˙˙˙2w

serving_default_precip_inputsPlaceholder*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
dtype0*(
shape:˙˙˙˙˙˙˙˙˙

serving_default_swe_inputsPlaceholder*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ*
dtype0**
shape!:˙˙˙˙˙˙˙˙˙ĂÓ

serving_default_temp_inputsPlaceholder*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
dtype0*(
shape:˙˙˙˙˙˙˙˙˙
×
StatefulPartitionedCallStatefulPartitionedCallserving_default_dem_inputsserving_default_et_inputsserving_default_precip_inputsserving_default_swe_inputsserving_default_temp_inputset_time_dist_conv2d/kernelet_time_dist_conv2d/biasswe_time_dist_conv2d/kernelswe_time_dist_conv2d/biasprecip_time_dist_conv2d/kernelprecip_time_dist_conv2d/biastemp_time_dist_conv2d/kerneltemp_time_dist_conv2d/biasdem_time_dist_conv2d/kerneldem_time_dist_conv2d/bias5transformer_encoder/multi_head_attention/query/kernel3transformer_encoder/multi_head_attention/query/bias3transformer_encoder/multi_head_attention/key/kernel1transformer_encoder/multi_head_attention/key/bias5transformer_encoder/multi_head_attention/value/kernel3transformer_encoder/multi_head_attention/value/bias@transformer_encoder/multi_head_attention/attention_output/kernel>transformer_encoder/multi_head_attention/attention_output/bias-transformer_encoder/layer_normalization/gamma,transformer_encoder/layer_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias/transformer_encoder/layer_normalization_1/gamma.transformer_encoder/layer_normalization_1/betadense_2/kerneldense_2/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_57638

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ŕ
valueľBą BŠ

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

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer*

%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer*

,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2layer*

3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer*

:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
	@layer*

A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Glayer* 

H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
	Nlayer* 

O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
	Ulayer* 

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer* 

]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
	clayer* 

d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
Ń
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

t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
Ś
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
_random_generator* 
Ž
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
ö
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
Ą24
˘25
26
27*
ö
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
Ą24
˘25
26
27*
* 
ľ
Łnon_trainable_variables
¤layers
Ľmetrics
 Ślayer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
¨trace_0
Štrace_1
Ştrace_2
Ťtrace_3* 
:
Źtrace_0
­trace_1
Žtrace_2
Żtrace_3* 
* 

°
_variables
ą_iterations
˛_learning_rate
ł_index_dict
´
_momentums
ľ_velocities
ś_update_step_xla*

ˇserving_default* 

0
1*

0
1*
* 

¸non_trainable_variables
šlayers
şmetrics
 ťlayer_regularization_losses
źlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

˝trace_0
žtrace_1* 

żtrace_0
Ŕtrace_1* 
Ń
Á	variables
Âtrainable_variables
Ăregularization_losses
Ä	keras_api
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
kernel
	bias
!Ç_jit_compiled_convolution_op*

0
1*

0
1*
* 

Čnon_trainable_variables
Élayers
Ęmetrics
 Ëlayer_regularization_losses
Ělayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Ítrace_0
Îtrace_1* 

Ďtrace_0
Đtrace_1* 
Ń
Ń	variables
Ňtrainable_variables
Óregularization_losses
Ô	keras_api
Ő__call__
+Ö&call_and_return_all_conditional_losses
kernel
	bias
!×_jit_compiled_convolution_op*

0
1*

0
1*
* 

Řnon_trainable_variables
Ůlayers
Úmetrics
 Űlayer_regularization_losses
Ülayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Ýtrace_0
Ţtrace_1* 

ßtrace_0
ŕtrace_1* 
Ń
á	variables
âtrainable_variables
ăregularization_losses
ä	keras_api
ĺ__call__
+ć&call_and_return_all_conditional_losses
kernel
	bias
!ç_jit_compiled_convolution_op*

0
1*

0
1*
* 

čnon_trainable_variables
élayers
ęmetrics
 ëlayer_regularization_losses
ělayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

ítrace_0
îtrace_1* 

ďtrace_0
đtrace_1* 
Ń
ń	variables
ňtrainable_variables
óregularization_losses
ô	keras_api
ő__call__
+ö&call_and_return_all_conditional_losses
kernel
	bias
!÷_jit_compiled_convolution_op*

0
1*

0
1*
* 

řnon_trainable_variables
ůlayers
úmetrics
 űlayer_regularization_losses
ülayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

ýtrace_0
ţtrace_1* 

˙trace_0
trace_1* 
Ń
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

 	variables
Ątrainable_variables
˘regularization_losses
Ł	keras_api
¤__call__
+Ľ&call_and_return_all_conditional_losses* 
* 
* 
* 

Śnon_trainable_variables
§layers
¨metrics
 Šlayer_regularization_losses
Şlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

Ťtrace_0
Źtrace_1* 

­trace_0
Žtrace_1* 

Ż	variables
°trainable_variables
ąregularization_losses
˛	keras_api
ł__call__
+´&call_and_return_all_conditional_losses* 
* 
* 
* 

ľnon_trainable_variables
ślayers
ˇmetrics
 ¸layer_regularization_losses
šlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

ştrace_0
ťtrace_1* 

źtrace_0
˝trace_1* 

ž	variables
żtrainable_variables
Ŕregularization_losses
Á	keras_api
Â__call__
+Ă&call_and_return_all_conditional_losses* 
* 
* 
* 

Änon_trainable_variables
Ĺlayers
Ćmetrics
 Çlayer_regularization_losses
Člayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

Étrace_0
Ętrace_1* 

Ëtrace_0
Ětrace_1* 

Í	variables
Îtrainable_variables
Ďregularization_losses
Đ	keras_api
Ń__call__
+Ň&call_and_return_all_conditional_losses* 
* 
* 
* 

Ónon_trainable_variables
Ôlayers
Őmetrics
 Ölayer_regularization_losses
×layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

Řtrace_0* 

Ůtrace_0* 

0
1
2
3
4
5
6
7
8
9
10
11
12
 13
Ą14
˘15*

0
1
2
3
4
5
6
7
8
9
10
11
12
 13
Ą14
˘15*
* 

Únon_trainable_variables
Űlayers
Ümetrics
 Ýlayer_regularization_losses
Ţlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ßtrace_0
ŕtrace_1* 

átrace_0
âtrace_1* 

ă	variables
ätrainable_variables
ĺregularization_losses
ć	keras_api
ç__call__
+č&call_and_return_all_conditional_losses
é_query_dense
ę
_key_dense
ë_value_dense
ě_softmax
í_dropout_layer
î_output_dense*
č
ďlayer_with_weights-0
ďlayer-0
đlayer_with_weights-1
đlayer-1
ń	variables
ňtrainable_variables
óregularization_losses
ô	keras_api
ő__call__
+ö&call_and_return_all_conditional_losses*
¸
÷	variables
řtrainable_variables
ůregularization_losses
ú	keras_api
ű__call__
+ü&call_and_return_all_conditional_losses
	ýaxis

gamma
	 beta*
¸
ţ	variables
˙trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

Ągamma
	˘beta*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
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
{
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

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

0*
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
ű
ą0
1
2
3
 4
Ą5
˘6
Ł7
¤8
Ľ9
Ś10
§11
¨12
Š13
Ş14
Ť15
Ź16
­17
Ž18
Ż19
°20
ą21
˛22
ł23
´24
ľ25
ś26
ˇ27
¸28
š29
ş30
ť31
ź32
˝33
ž34
ż35
Ŕ36
Á37
Â38
Ă39
Ä40
Ĺ41
Ć42
Ç43
Č44
É45
Ę46
Ë47
Ě48
Í49
Î50
Ď51
Đ52
Ń53
Ň54
Ó55
Ô56*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
ö
0
1
Ą2
Ł3
Ľ4
§5
Š6
Ť7
­8
Ż9
ą10
ł11
ľ12
ˇ13
š14
ť15
˝16
ż17
Á18
Ă19
Ĺ20
Ç21
É22
Ë23
Í24
Ď25
Ń26
Ó27*
ö
0
 1
˘2
¤3
Ś4
¨5
Ş6
Ź7
Ž8
°9
˛10
´11
ś12
¸13
ş14
ź15
ž16
Ŕ17
Â18
Ä19
Ć20
Č21
Ę22
Ě23
Î24
Đ25
Ň26
Ô27*
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
0
1*

0
1*
* 

Őnon_trainable_variables
Ölayers
×metrics
 Řlayer_regularization_losses
Ůlayer_metrics
Á	variables
Âtrainable_variables
Ăregularization_losses
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses*

Útrace_0* 

Űtrace_0* 
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
0
1*

0
1*
* 

Ünon_trainable_variables
Ýlayers
Ţmetrics
 ßlayer_regularization_losses
ŕlayer_metrics
Ń	variables
Ňtrainable_variables
Óregularization_losses
Ő__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses*

átrace_0* 

âtrace_0* 
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
0
1*

0
1*
* 

ănon_trainable_variables
älayers
ĺmetrics
 ćlayer_regularization_losses
çlayer_metrics
á	variables
âtrainable_variables
ăregularization_losses
ĺ__call__
+ć&call_and_return_all_conditional_losses
'ć"call_and_return_conditional_losses*

čtrace_0* 

étrace_0* 
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
0
1*

0
1*
* 

ęnon_trainable_variables
ëlayers
ěmetrics
 ílayer_regularization_losses
îlayer_metrics
ń	variables
ňtrainable_variables
óregularization_losses
ő__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*

ďtrace_0* 

đtrace_0* 
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
0
1*

0
1*
* 

ńnon_trainable_variables
ňlayers
ómetrics
 ôlayer_regularization_losses
őlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ötrace_0* 

÷trace_0* 
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

řnon_trainable_variables
ůlayers
úmetrics
 űlayer_regularization_losses
ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ýtrace_0* 

ţtrace_0* 
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

˙non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
Ątrainable_variables
˘regularization_losses
¤__call__
+Ľ&call_and_return_all_conditional_losses
'Ľ"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ż	variables
°trainable_variables
ąregularization_losses
ł__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ž	variables
żtrainable_variables
Ŕregularization_losses
Â__call__
+Ă&call_and_return_all_conditional_losses
'Ă"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Í	variables
Îtrainable_variables
Ďregularization_losses
Ń__call__
+Ň&call_and_return_all_conditional_losses
'Ň"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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
0
1
2
3
4
5
6
7*
D
0
1
2
3
4
5
6
7*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ă	variables
ätrainable_variables
ĺregularization_losses
ç__call__
+č&call_and_return_all_conditional_losses
'č"call_and_return_conditional_losses*
* 
* 
á
 	variables
Ątrainable_variables
˘regularization_losses
Ł	keras_api
¤__call__
+Ľ&call_and_return_all_conditional_losses
Śpartial_output_shape
§full_output_shape
kernel
	bias*
á
¨	variables
Štrainable_variables
Şregularization_losses
Ť	keras_api
Ź__call__
+­&call_and_return_all_conditional_losses
Žpartial_output_shape
Żfull_output_shape
kernel
	bias*
á
°	variables
ątrainable_variables
˛regularization_losses
ł	keras_api
´__call__
+ľ&call_and_return_all_conditional_losses
śpartial_output_shape
ˇfull_output_shape
kernel
	bias*

¸	variables
štrainable_variables
şregularization_losses
ť	keras_api
ź__call__
+˝&call_and_return_all_conditional_losses* 
Ź
ž	variables
żtrainable_variables
Ŕregularization_losses
Á	keras_api
Â__call__
+Ă&call_and_return_all_conditional_losses
Ä_random_generator* 
á
Ĺ	variables
Ćtrainable_variables
Çregularization_losses
Č	keras_api
É__call__
+Ę&call_and_return_all_conditional_losses
Ëpartial_output_shape
Ěfull_output_shape
kernel
	bias*
Ž
Í	variables
Îtrainable_variables
Ďregularization_losses
Đ	keras_api
Ń__call__
+Ň&call_and_return_all_conditional_losses
kernel
	bias*
Ž
Ó	variables
Ôtrainable_variables
Őregularization_losses
Ö	keras_api
×__call__
+Ř&call_and_return_all_conditional_losses
kernel
	bias*
$
0
1
2
3*
$
0
1
2
3*
* 

Ůnon_trainable_variables
Úlayers
Űmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
ń	variables
ňtrainable_variables
óregularization_losses
ő__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*
:
Ţtrace_0
ßtrace_1
ŕtrace_2
átrace_3* 
:
âtrace_0
ătrace_1
ätrace_2
ĺtrace_3* 

0
 1*

0
 1*
* 

ćnon_trainable_variables
çlayers
čmetrics
 élayer_regularization_losses
ęlayer_metrics
÷	variables
řtrainable_variables
ůregularization_losses
ű__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses*
* 
* 
* 

Ą0
˘1*

Ą0
˘1*
* 

ënon_trainable_variables
ělayers
ímetrics
 îlayer_regularization_losses
ďlayer_metrics
ţ	variables
˙trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
đ	variables
ń	keras_api

ňtotal

ócount*
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

VARIABLE_VALUE<Adam/m/transformer_encoder/multi_head_attention/query/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/v/transformer_encoder/multi_head_attention/query/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/query/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/query/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/key/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/key/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE8Adam/m/transformer_encoder/multi_head_attention/key/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE8Adam/v/transformer_encoder/multi_head_attention/key/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/m/transformer_encoder/multi_head_attention/value/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/v/transformer_encoder/multi_head_attention/value/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/m/transformer_encoder/multi_head_attention/value/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/v/transformer_encoder/multi_head_attention/value/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEGAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEGAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEEAdam/m/transformer_encoder/multi_head_attention/attention_output/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*

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
z
VARIABLE_VALUE4Adam/m/transformer_encoder/layer_normalization/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE4Adam/v/transformer_encoder/layer_normalization/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/transformer_encoder/layer_normalization/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/transformer_encoder/layer_normalization/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE6Adam/m/transformer_encoder/layer_normalization_1/gamma2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE6Adam/v/transformer_encoder/layer_normalization_1/gamma2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE5Adam/m/transformer_encoder/layer_normalization_1/beta2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
{
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
é0
ę1
ë2
ě3
í4
î5*
* 
* 
* 

0
1*

0
1*
* 

ônon_trainable_variables
őlayers
ömetrics
 ÷layer_regularization_losses
řlayer_metrics
 	variables
Ątrainable_variables
˘regularization_losses
¤__call__
+Ľ&call_and_return_all_conditional_losses
'Ľ"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

ůnon_trainable_variables
úlayers
űmetrics
 ülayer_regularization_losses
ýlayer_metrics
¨	variables
Štrainable_variables
Şregularization_losses
Ź__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

ţnon_trainable_variables
˙layers
metrics
 layer_regularization_losses
layer_metrics
°	variables
ątrainable_variables
˛regularization_losses
´__call__
+ľ&call_and_return_all_conditional_losses
'ľ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
štrainable_variables
şregularization_losses
ź__call__
+˝&call_and_return_all_conditional_losses
'˝"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ž	variables
żtrainable_variables
Ŕregularization_losses
Â__call__
+Ă&call_and_return_all_conditional_losses
'Ă"call_and_return_conditional_losses* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ĺ	variables
Ćtrainable_variables
Çregularization_losses
É__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Í	variables
Îtrainable_variables
Ďregularization_losses
Ń__call__
+Ň&call_and_return_all_conditional_losses
'Ň"call_and_return_conditional_losses*

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Őregularization_losses
×__call__
+Ř&call_and_return_all_conditional_losses
'Ř"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

ď0
đ1*
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
ň0
ó1*

đ	variables*
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
Ţ+
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
GPU 2J 8 *'
f"R 
__inference__traced_save_60080
ů
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_60354čă&
ˇ
Ň
*__inference_sequential_layer_call_fn_56373
dense_input
unknown:
 
	unknown_0: 
	unknown_1: 

	unknown_2:

identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_56349s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

%
_user_specified_namedense_input
­
E
)__inference_flatten_2_layer_call_fn_59531

inputs
identityŻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_56000`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ë
J
.__inference_precip_flatten_layer_call_fn_58788

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56064m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ž

'__inference_dense_2_layer_call_fn_59405

inputs
unknown:

	unknown_0:
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_56718o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ő
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_58898

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ŕg
ă
@__inference_model_layer_call_and_return_conditional_losses_57247

inputs
inputs_1
inputs_2
inputs_3
inputs_43
et_time_dist_conv2d_57155:!''
et_time_dist_conv2d_57157:6
swe_time_dist_conv2d_57162:(
swe_time_dist_conv2d_57164:7
precip_time_dist_conv2d_57169:+
precip_time_dist_conv2d_57171:5
temp_time_dist_conv2d_57176:)
temp_time_dist_conv2d_57178:6
dem_time_dist_conv2d_57183:úŢ(
dem_time_dist_conv2d_57185:/
transformer_encoder_57206:

+
transformer_encoder_57208:
/
transformer_encoder_57210:

+
transformer_encoder_57212:
/
transformer_encoder_57214:

+
transformer_encoder_57216:
/
transformer_encoder_57218:

'
transformer_encoder_57220:
'
transformer_encoder_57222:
'
transformer_encoder_57224:
+
transformer_encoder_57226:
 '
transformer_encoder_57228: +
transformer_encoder_57230: 
'
transformer_encoder_57232:
'
transformer_encoder_57234:
'
transformer_encoder_57236:

dense_2_57241:

dense_2_57243:
identity˘,dem_time_dist_conv2d/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘+et_time_dist_conv2d/StatefulPartitionedCall˘/precip_time_dist_conv2d/StatefulPartitionedCall˘,swe_time_dist_conv2d/StatefulPartitionedCall˘-temp_time_dist_conv2d/StatefulPartitionedCall˘+transformer_encoder/StatefulPartitionedCall§
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_4et_time_dist_conv2d_57155et_time_dist_conv2d_57157*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55916z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2wŤ
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_3swe_time_dist_conv2d_57162swe_time_dist_conv2d_57164*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55830{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓˇ
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_2precip_time_dist_conv2d_57169precip_time_dist_conv2d_57171*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55744~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1temp_time_dist_conv2d_57176temp_time_dist_conv2d_57178*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55658|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsdem_time_dist_conv2d_57183dem_time_dist_conv2d_57185*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55572{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshapeinputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷đ
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55977r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ó
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56034s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ś
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ů
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56091u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ź
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56148r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙í
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56205q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         °
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ţ
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_56489Ń
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_57206transformer_encoder_57208transformer_encoder_57210transformer_encoder_57212transformer_encoder_57214transformer_encoder_57216transformer_encoder_57218transformer_encoder_57220transformer_encoder_57222transformer_encoder_57224transformer_encoder_57226transformer_encoder_57228transformer_encoder_57230transformer_encoder_57232transformer_encoder_57234transformer_encoder_57236*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_57032ý
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411ě
dropout/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56814
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_57241dense_2_57243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_56718w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
!:˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_59515

inputs8
conv2d_readvariableop_resource:!'-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2w: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ĺ	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_59415

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ĺ
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_56178

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_56171\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_59374

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56706`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
­
E
)__inference_flatten_3_layer_call_fn_59542

inputs
identityŻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_56057`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
×
Ű
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55830

inputs*
conv2d_4_55818:
conv2d_4_55820:
identity˘ conv2d_4/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙Ă   Ó     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ˙
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_4_55818conv2d_4_55820*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55776\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_4/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
Î
Ż
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58563

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:
identity˘conv2d_3/BiasAdd/ReadVariableOp˘conv2d_3/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ś
conv2d_3/Conv2DConv2DReshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę

%__inference_dense_layer_call_fn_59719

inputs
unknown:
 
	unknown_0: 
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_56246s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ĺ
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_56205

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_56171\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
é
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56064

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_3/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_56057\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_3/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55604

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŐŢ
­
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59358

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_query_add_readvariableop_resource:
T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:

F
4multi_head_attention_key_add_readvariableop_resource:
V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_value_add_readvariableop_resource:
a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:

O
Amulti_head_attention_attention_output_add_readvariableop_resource:
?
1layer_normalization_mul_3_readvariableop_resource:
=
/layer_normalization_add_readvariableop_resource:
D
2sequential_dense_tensordot_readvariableop_resource:
 >
0sequential_dense_biasadd_readvariableop_resource: F
4sequential_dense_1_tensordot_readvariableop_resource: 
@
2sequential_dense_1_biasadd_readvariableop_resource:
A
3layer_normalization_1_mul_3_readvariableop_resource:
?
1layer_normalization_1_add_readvariableop_resource:

identity˘&layer_normalization/add/ReadVariableOp˘(layer_normalization/mul_3/ReadVariableOp˘(layer_normalization_1/add/ReadVariableOp˘*layer_normalization_1/mul_3/ReadVariableOp˘8multi_head_attention/attention_output/add/ReadVariableOp˘Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘+multi_head_attention/key/add/ReadVariableOp˘5multi_head_attention/key/einsum/Einsum/ReadVariableOp˘-multi_head_attention/query/add/ReadVariableOp˘7multi_head_attention/query/einsum/Einsum/ReadVariableOp˘-multi_head_attention/value/add/ReadVariableOp˘7multi_head_attention/value/einsum/Einsum/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOpź
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0×
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ĺ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>˘
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdŇ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeś
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0č
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
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
valueB:ľ
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
value	B :
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
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:˝
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
value	B :
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
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
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
 *  ?Ą
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙u
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
 *    ¤
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:Ź
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0°
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ľ
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ˙
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
value	B : 
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
valueB: Ą
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ŕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ź
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˝
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ś
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ż
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙  
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : 
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
value	B : 
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
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:˛
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ź
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ă
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ź
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ľ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
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
valueB:ż
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
value	B :
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
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ç
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
value	B :
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
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
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
 *  ?§
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙y
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
 *    Ş
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB §
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:˛
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ś
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ť
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
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
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ä
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_59559

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Ş
5__inference_temp_time_dist_conv2d_layer_call_fn_58449

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55658
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_59396

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ś
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ě
­
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58473

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity˘conv2d_2/BiasAdd/ReadVariableOp˘conv2d_2/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ś
conv2d_2/Conv2DConv2DReshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
Ű
%__inference_model_layer_call_fn_57703
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4!
unknown:!'
	unknown_0:%
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:úŢ
	unknown_8:
	unknown_9:



unknown_10:
 

unknown_11:



unknown_12:
 

unknown_13:



unknown_14:
 

unknown_15:



unknown_16:


unknown_17:


unknown_18:


unknown_19:
 

unknown_20: 

unknown_21: 


unknown_22:


unknown_23:


unknown_24:


unknown_25:


unknown_26:
identity˘StatefulPartitionedCallć
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
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_56725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
"
_user_specified_name
inputs_4
÷

E__inference_sequential_layer_call_and_return_conditional_losses_56289

inputs
dense_56247:
 
dense_56249: 
dense_1_56283: 

dense_1_56285:

identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallĺ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56247dense_56249*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_56246
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56283dense_1_56285*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_56282{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
­
E
)__inference_flatten_1_layer_call_fn_59564

inputs
identityŻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_56171`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ţ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55776

inputs:
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AMr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
ĺ
G
+__inference_dem_flatten_layer_call_fn_58700

inputs
identityž
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55950m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_59369

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
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:e a
=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55862

inputs8
conv2d_readvariableop_resource:!'-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2w: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_59548

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411

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
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:e a
=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
Ű
%__inference_model_layer_call_fn_57768
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4!
unknown:!'
	unknown_0:%
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:úŢ
	unknown_8:
	unknown_9:



unknown_10:
 

unknown_11:



unknown_12:
 

unknown_13:



unknown_14:
 

unknown_15:



unknown_16:


unknown_17:


unknown_18:


unknown_19:
 

unknown_20: 

unknown_21: 


unknown_22:


unknown_23:


unknown_24:


unknown_25:


unknown_26:
identity˘StatefulPartitionedCallć
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
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_57247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
"
_user_specified_name
inputs_4
Ť
Ś
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58407

inputsA
%conv2d_conv2d_readvariableop_resource:úŢ4
&conv2d_biasadd_readvariableop_resource:
identity˘conv2d/BiasAdd/ReadVariableOp˘conv2d/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙÷       n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0´
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
ł
¨
3__inference_et_time_dist_conv2d_layer_call_fn_58647

inputs!
unknown:!'
	unknown_0:
identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55916
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ň
Ü
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55744

inputs(
conv2d_3_55732:
conv2d_3_55734:
identity˘ conv2d_3/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˙
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_3_55732conv2d_3_55734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55690\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ë
`
'__inference_dropout_layer_call_fn_59379

inputs
identity˘StatefulPartitionedCall˝
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56814o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
É
ů
B__inference_dense_1_layer_call_and_return_conditional_losses_59789

inputs3
!tensordot_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : ť
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
value	B : ż
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
value	B : 
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
:˙˙˙˙˙˙˙˙˙ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
÷

E__inference_sequential_layer_call_and_return_conditional_losses_56349

inputs
dense_56338:
 
dense_56340: 
dense_1_56343: 

dense_1_56345:

identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallĺ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56338dense_56340*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_56246
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56343dense_1_56345*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_56282{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ę

&__inference_conv2d_layer_call_fn_59424

inputs#
unknown:úŢ
	unknown_0:
identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_55518w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙÷: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
ć
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56121

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_4/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_56114\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_4/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
C__inference_conv2d_3_layer_call_and_return_conditional_losses_59475

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
Ř
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55875

inputs(
conv2d_1_55863:!'
conv2d_1_55865:
identity˘ conv2d_1/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙2   w      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w˙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_55863conv2d_1_55865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55862\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ţ

F__inference_concatenate_layer_call_and_return_conditional_losses_56489

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
value	B :
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesu
s:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:SO
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:SO
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:SO
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs:SO
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ő
a
E__inference_et_flatten_layer_call_and_return_conditional_losses_58915

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŕ
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55977

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_55943\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę
Ť
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58695

inputsA
'conv2d_1_conv2d_readvariableop_resource:!'6
(conv2d_1_biasadd_readvariableop_resource:
identity˘conv2d_1/BiasAdd/ReadVariableOp˘conv2d_1/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙2   w      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0ś
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
ŐŢ
­
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_56666

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_query_add_readvariableop_resource:
T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:

F
4multi_head_attention_key_add_readvariableop_resource:
V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_value_add_readvariableop_resource:
a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:

O
Amulti_head_attention_attention_output_add_readvariableop_resource:
?
1layer_normalization_mul_3_readvariableop_resource:
=
/layer_normalization_add_readvariableop_resource:
D
2sequential_dense_tensordot_readvariableop_resource:
 >
0sequential_dense_biasadd_readvariableop_resource: F
4sequential_dense_1_tensordot_readvariableop_resource: 
@
2sequential_dense_1_biasadd_readvariableop_resource:
A
3layer_normalization_1_mul_3_readvariableop_resource:
?
1layer_normalization_1_add_readvariableop_resource:

identity˘&layer_normalization/add/ReadVariableOp˘(layer_normalization/mul_3/ReadVariableOp˘(layer_normalization_1/add/ReadVariableOp˘*layer_normalization_1/mul_3/ReadVariableOp˘8multi_head_attention/attention_output/add/ReadVariableOp˘Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘+multi_head_attention/key/add/ReadVariableOp˘5multi_head_attention/key/einsum/Einsum/ReadVariableOp˘-multi_head_attention/query/add/ReadVariableOp˘7multi_head_attention/query/einsum/Einsum/ReadVariableOp˘-multi_head_attention/value/add/ReadVariableOp˘7multi_head_attention/value/einsum/Einsum/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOpź
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0×
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ĺ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>˘
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdŇ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeś
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0č
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
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
valueB:ľ
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
value	B :
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
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:˝
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
value	B :
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
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
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
 *  ?Ą
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙u
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
 *    ¤
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:Ź
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0°
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ľ
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ˙
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
value	B : 
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
valueB: Ą
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ŕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ź
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˝
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ś
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ż
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙  
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : 
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
value	B : 
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
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:˛
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ź
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ă
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ź
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ľ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
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
valueB:ż
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
value	B :
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
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ç
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
value	B :
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
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
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
 *  ?§
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙y
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
 *    Ş
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB §
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:˛
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ś
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ť
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
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
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ä
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_56171

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

(__inference_conv2d_3_layer_call_fn_59464

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55690w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă
ÔD
!__inference__traced_restore_60354
file_prefix1
assignvariableop_dense_2_kernel:
-
assignvariableop_1_dense_2_bias:J
.assignvariableop_2_dem_time_dist_conv2d_kernel:úŢ:
,assignvariableop_3_dem_time_dist_conv2d_bias:I
/assignvariableop_4_temp_time_dist_conv2d_kernel:;
-assignvariableop_5_temp_time_dist_conv2d_bias:K
1assignvariableop_6_precip_time_dist_conv2d_kernel:=
/assignvariableop_7_precip_time_dist_conv2d_bias:J
.assignvariableop_8_swe_time_dist_conv2d_kernel::
,assignvariableop_9_swe_time_dist_conv2d_bias:H
.assignvariableop_10_et_time_dist_conv2d_kernel:!':
,assignvariableop_11_et_time_dist_conv2d_bias:_
Iassignvariableop_12_transformer_encoder_multi_head_attention_query_kernel:

Y
Gassignvariableop_13_transformer_encoder_multi_head_attention_query_bias:
]
Gassignvariableop_14_transformer_encoder_multi_head_attention_key_kernel:

W
Eassignvariableop_15_transformer_encoder_multi_head_attention_key_bias:
_
Iassignvariableop_16_transformer_encoder_multi_head_attention_value_kernel:

Y
Gassignvariableop_17_transformer_encoder_multi_head_attention_value_bias:
j
Tassignvariableop_18_transformer_encoder_multi_head_attention_attention_output_kernel:

`
Rassignvariableop_19_transformer_encoder_multi_head_attention_attention_output_bias:
2
 assignvariableop_20_dense_kernel:
 ,
assignvariableop_21_dense_bias: 4
"assignvariableop_22_dense_1_kernel: 
.
 assignvariableop_23_dense_1_bias:
O
Aassignvariableop_24_transformer_encoder_layer_normalization_gamma:
N
@assignvariableop_25_transformer_encoder_layer_normalization_beta:
Q
Cassignvariableop_26_transformer_encoder_layer_normalization_1_gamma:
P
Bassignvariableop_27_transformer_encoder_layer_normalization_1_beta:
'
assignvariableop_28_iteration:	 +
!assignvariableop_29_learning_rate: R
6assignvariableop_30_adam_m_dem_time_dist_conv2d_kernel:úŢR
6assignvariableop_31_adam_v_dem_time_dist_conv2d_kernel:úŢB
4assignvariableop_32_adam_m_dem_time_dist_conv2d_bias:B
4assignvariableop_33_adam_v_dem_time_dist_conv2d_bias:Q
7assignvariableop_34_adam_m_temp_time_dist_conv2d_kernel:Q
7assignvariableop_35_adam_v_temp_time_dist_conv2d_kernel:C
5assignvariableop_36_adam_m_temp_time_dist_conv2d_bias:C
5assignvariableop_37_adam_v_temp_time_dist_conv2d_bias:S
9assignvariableop_38_adam_m_precip_time_dist_conv2d_kernel:S
9assignvariableop_39_adam_v_precip_time_dist_conv2d_kernel:E
7assignvariableop_40_adam_m_precip_time_dist_conv2d_bias:E
7assignvariableop_41_adam_v_precip_time_dist_conv2d_bias:R
6assignvariableop_42_adam_m_swe_time_dist_conv2d_kernel:R
6assignvariableop_43_adam_v_swe_time_dist_conv2d_kernel:B
4assignvariableop_44_adam_m_swe_time_dist_conv2d_bias:B
4assignvariableop_45_adam_v_swe_time_dist_conv2d_bias:O
5assignvariableop_46_adam_m_et_time_dist_conv2d_kernel:!'O
5assignvariableop_47_adam_v_et_time_dist_conv2d_kernel:!'A
3assignvariableop_48_adam_m_et_time_dist_conv2d_bias:A
3assignvariableop_49_adam_v_et_time_dist_conv2d_bias:f
Passignvariableop_50_adam_m_transformer_encoder_multi_head_attention_query_kernel:

f
Passignvariableop_51_adam_v_transformer_encoder_multi_head_attention_query_kernel:

`
Nassignvariableop_52_adam_m_transformer_encoder_multi_head_attention_query_bias:
`
Nassignvariableop_53_adam_v_transformer_encoder_multi_head_attention_query_bias:
d
Nassignvariableop_54_adam_m_transformer_encoder_multi_head_attention_key_kernel:

d
Nassignvariableop_55_adam_v_transformer_encoder_multi_head_attention_key_kernel:

^
Lassignvariableop_56_adam_m_transformer_encoder_multi_head_attention_key_bias:
^
Lassignvariableop_57_adam_v_transformer_encoder_multi_head_attention_key_bias:
f
Passignvariableop_58_adam_m_transformer_encoder_multi_head_attention_value_kernel:

f
Passignvariableop_59_adam_v_transformer_encoder_multi_head_attention_value_kernel:

`
Nassignvariableop_60_adam_m_transformer_encoder_multi_head_attention_value_bias:
`
Nassignvariableop_61_adam_v_transformer_encoder_multi_head_attention_value_bias:
q
[assignvariableop_62_adam_m_transformer_encoder_multi_head_attention_attention_output_kernel:

q
[assignvariableop_63_adam_v_transformer_encoder_multi_head_attention_attention_output_kernel:

g
Yassignvariableop_64_adam_m_transformer_encoder_multi_head_attention_attention_output_bias:
g
Yassignvariableop_65_adam_v_transformer_encoder_multi_head_attention_attention_output_bias:
9
'assignvariableop_66_adam_m_dense_kernel:
 9
'assignvariableop_67_adam_v_dense_kernel:
 3
%assignvariableop_68_adam_m_dense_bias: 3
%assignvariableop_69_adam_v_dense_bias: ;
)assignvariableop_70_adam_m_dense_1_kernel: 
;
)assignvariableop_71_adam_v_dense_1_kernel: 
5
'assignvariableop_72_adam_m_dense_1_bias:
5
'assignvariableop_73_adam_v_dense_1_bias:
V
Hassignvariableop_74_adam_m_transformer_encoder_layer_normalization_gamma:
V
Hassignvariableop_75_adam_v_transformer_encoder_layer_normalization_gamma:
U
Gassignvariableop_76_adam_m_transformer_encoder_layer_normalization_beta:
U
Gassignvariableop_77_adam_v_transformer_encoder_layer_normalization_beta:
X
Jassignvariableop_78_adam_m_transformer_encoder_layer_normalization_1_gamma:
X
Jassignvariableop_79_adam_v_transformer_encoder_layer_normalization_1_gamma:
W
Iassignvariableop_80_adam_m_transformer_encoder_layer_normalization_1_beta:
W
Iassignvariableop_81_adam_v_transformer_encoder_layer_normalization_1_beta:
;
)assignvariableop_82_adam_m_dense_2_kernel:
;
)assignvariableop_83_adam_v_dense_2_kernel:
5
'assignvariableop_84_adam_m_dense_2_bias:5
'assignvariableop_85_adam_v_dense_2_bias:#
assignvariableop_86_total: #
assignvariableop_87_count: 
identity_89˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_35˘AssignVariableOp_36˘AssignVariableOp_37˘AssignVariableOp_38˘AssignVariableOp_39˘AssignVariableOp_4˘AssignVariableOp_40˘AssignVariableOp_41˘AssignVariableOp_42˘AssignVariableOp_43˘AssignVariableOp_44˘AssignVariableOp_45˘AssignVariableOp_46˘AssignVariableOp_47˘AssignVariableOp_48˘AssignVariableOp_49˘AssignVariableOp_5˘AssignVariableOp_50˘AssignVariableOp_51˘AssignVariableOp_52˘AssignVariableOp_53˘AssignVariableOp_54˘AssignVariableOp_55˘AssignVariableOp_56˘AssignVariableOp_57˘AssignVariableOp_58˘AssignVariableOp_59˘AssignVariableOp_6˘AssignVariableOp_60˘AssignVariableOp_61˘AssignVariableOp_62˘AssignVariableOp_63˘AssignVariableOp_64˘AssignVariableOp_65˘AssignVariableOp_66˘AssignVariableOp_67˘AssignVariableOp_68˘AssignVariableOp_69˘AssignVariableOp_7˘AssignVariableOp_70˘AssignVariableOp_71˘AssignVariableOp_72˘AssignVariableOp_73˘AssignVariableOp_74˘AssignVariableOp_75˘AssignVariableOp_76˘AssignVariableOp_77˘AssignVariableOp_78˘AssignVariableOp_79˘AssignVariableOp_8˘AssignVariableOp_80˘AssignVariableOp_81˘AssignVariableOp_82˘AssignVariableOp_83˘AssignVariableOp_84˘AssignVariableOp_85˘AssignVariableOp_86˘AssignVariableOp_87˘AssignVariableOp_9Ĺ"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*ë!
valueá!BŢ!YB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHĽ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*Ç
value˝BşYB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ţ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ú
_output_shapesç
ä:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ś
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ĺ
AssignVariableOp_2AssignVariableOp.assignvariableop_2_dem_time_dist_conv2d_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_3AssignVariableOp,assignvariableop_3_dem_time_dist_conv2d_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ć
AssignVariableOp_4AssignVariableOp/assignvariableop_4_temp_time_dist_conv2d_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_5AssignVariableOp-assignvariableop_5_temp_time_dist_conv2d_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Č
AssignVariableOp_6AssignVariableOp1assignvariableop_6_precip_time_dist_conv2d_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ć
AssignVariableOp_7AssignVariableOp/assignvariableop_7_precip_time_dist_conv2d_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ĺ
AssignVariableOp_8AssignVariableOp.assignvariableop_8_swe_time_dist_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_9AssignVariableOp,assignvariableop_9_swe_time_dist_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_10AssignVariableOp.assignvariableop_10_et_time_dist_conv2d_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ĺ
AssignVariableOp_11AssignVariableOp,assignvariableop_11_et_time_dist_conv2d_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:â
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_encoder_multi_head_attention_query_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ŕ
AssignVariableOp_13AssignVariableOpGassignvariableop_13_transformer_encoder_multi_head_attention_query_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ŕ
AssignVariableOp_14AssignVariableOpGassignvariableop_14_transformer_encoder_multi_head_attention_key_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ţ
AssignVariableOp_15AssignVariableOpEassignvariableop_15_transformer_encoder_multi_head_attention_key_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:â
AssignVariableOp_16AssignVariableOpIassignvariableop_16_transformer_encoder_multi_head_attention_value_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ŕ
AssignVariableOp_17AssignVariableOpGassignvariableop_17_transformer_encoder_multi_head_attention_value_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:í
AssignVariableOp_18AssignVariableOpTassignvariableop_18_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ë
AssignVariableOp_19AssignVariableOpRassignvariableop_19_transformer_encoder_multi_head_attention_attention_output_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ť
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ú
AssignVariableOp_24AssignVariableOpAassignvariableop_24_transformer_encoder_layer_normalization_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ů
AssignVariableOp_25AssignVariableOp@assignvariableop_25_transformer_encoder_layer_normalization_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ü
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_encoder_layer_normalization_1_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ű
AssignVariableOp_27AssignVariableOpBassignvariableop_27_transformer_encoder_layer_normalization_1_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:ś
AssignVariableOp_28AssignVariableOpassignvariableop_28_iterationIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ş
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_m_dem_time_dist_conv2d_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_v_dem_time_dist_conv2d_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_m_dem_time_dist_conv2d_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_v_dem_time_dist_conv2d_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_m_temp_time_dist_conv2d_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_v_temp_time_dist_conv2d_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_m_temp_time_dist_conv2d_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_v_temp_time_dist_conv2d_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ň
AssignVariableOp_38AssignVariableOp9assignvariableop_38_adam_m_precip_time_dist_conv2d_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ň
AssignVariableOp_39AssignVariableOp9assignvariableop_39_adam_v_precip_time_dist_conv2d_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_m_precip_time_dist_conv2d_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_v_precip_time_dist_conv2d_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_m_swe_time_dist_conv2d_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_v_swe_time_dist_conv2d_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_m_swe_time_dist_conv2d_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_v_swe_time_dist_conv2d_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_m_et_time_dist_conv2d_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_v_et_time_dist_conv2d_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ě
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_m_et_time_dist_conv2d_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ě
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_v_et_time_dist_conv2d_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:é
AssignVariableOp_50AssignVariableOpPassignvariableop_50_adam_m_transformer_encoder_multi_head_attention_query_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:é
AssignVariableOp_51AssignVariableOpPassignvariableop_51_adam_v_transformer_encoder_multi_head_attention_query_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_52AssignVariableOpNassignvariableop_52_adam_m_transformer_encoder_multi_head_attention_query_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_53AssignVariableOpNassignvariableop_53_adam_v_transformer_encoder_multi_head_attention_query_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_54AssignVariableOpNassignvariableop_54_adam_m_transformer_encoder_multi_head_attention_key_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_55AssignVariableOpNassignvariableop_55_adam_v_transformer_encoder_multi_head_attention_key_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:ĺ
AssignVariableOp_56AssignVariableOpLassignvariableop_56_adam_m_transformer_encoder_multi_head_attention_key_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:ĺ
AssignVariableOp_57AssignVariableOpLassignvariableop_57_adam_v_transformer_encoder_multi_head_attention_key_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:é
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_m_transformer_encoder_multi_head_attention_value_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:é
AssignVariableOp_59AssignVariableOpPassignvariableop_59_adam_v_transformer_encoder_multi_head_attention_value_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_60AssignVariableOpNassignvariableop_60_adam_m_transformer_encoder_multi_head_attention_value_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ç
AssignVariableOp_61AssignVariableOpNassignvariableop_61_adam_v_transformer_encoder_multi_head_attention_value_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:ô
AssignVariableOp_62AssignVariableOp[assignvariableop_62_adam_m_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:ô
AssignVariableOp_63AssignVariableOp[assignvariableop_63_adam_v_transformer_encoder_multi_head_attention_attention_output_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:ň
AssignVariableOp_64AssignVariableOpYassignvariableop_64_adam_m_transformer_encoder_multi_head_attention_attention_output_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ň
AssignVariableOp_65AssignVariableOpYassignvariableop_65_adam_v_transformer_encoder_multi_head_attention_attention_output_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_m_dense_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_v_dense_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:ž
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_m_dense_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:ž
AssignVariableOp_69AssignVariableOp%assignvariableop_69_adam_v_dense_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_m_dense_1_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_v_dense_1_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_m_dense_1_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_v_dense_1_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:á
AssignVariableOp_74AssignVariableOpHassignvariableop_74_adam_m_transformer_encoder_layer_normalization_gammaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:á
AssignVariableOp_75AssignVariableOpHassignvariableop_75_adam_v_transformer_encoder_layer_normalization_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:ŕ
AssignVariableOp_76AssignVariableOpGassignvariableop_76_adam_m_transformer_encoder_layer_normalization_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:ŕ
AssignVariableOp_77AssignVariableOpGassignvariableop_77_adam_v_transformer_encoder_layer_normalization_betaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:ă
AssignVariableOp_78AssignVariableOpJassignvariableop_78_adam_m_transformer_encoder_layer_normalization_1_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:ă
AssignVariableOp_79AssignVariableOpJassignvariableop_79_adam_v_transformer_encoder_layer_normalization_1_gammaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:â
AssignVariableOp_80AssignVariableOpIassignvariableop_80_adam_m_transformer_encoder_layer_normalization_1_betaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:â
AssignVariableOp_81AssignVariableOpIassignvariableop_81_adam_v_transformer_encoder_layer_normalization_1_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_m_dense_2_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_v_dense_2_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_m_dense_2_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_v_dense_2_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_86AssignVariableOpassignvariableop_86_totalIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_87AssignVariableOpassignvariableop_87_countIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ß
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: Ě
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*Ç
_input_shapesľ
˛: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
Ĺ	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_56718

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
 ´
1
__inference__traced_save_60080
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

identity_1˘MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Â"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*ë!
valueá!BŢ!YB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH˘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*Ç
value˝BşYB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 0
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_dem_time_dist_conv2d_kernel_read_readvariableop4savev2_dem_time_dist_conv2d_bias_read_readvariableop7savev2_temp_time_dist_conv2d_kernel_read_readvariableop5savev2_temp_time_dist_conv2d_bias_read_readvariableop9savev2_precip_time_dist_conv2d_kernel_read_readvariableop7savev2_precip_time_dist_conv2d_bias_read_readvariableop6savev2_swe_time_dist_conv2d_kernel_read_readvariableop4savev2_swe_time_dist_conv2d_bias_read_readvariableop5savev2_et_time_dist_conv2d_kernel_read_readvariableop3savev2_et_time_dist_conv2d_bias_read_readvariableopPsavev2_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_query_bias_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopLsavev2_transformer_encoder_multi_head_attention_key_bias_read_readvariableopPsavev2_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopNsavev2_transformer_encoder_multi_head_attention_value_bias_read_readvariableop[savev2_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopYsavev2_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopHsavev2_transformer_encoder_layer_normalization_gamma_read_readvariableopGsavev2_transformer_encoder_layer_normalization_beta_read_readvariableopJsavev2_transformer_encoder_layer_normalization_1_gamma_read_readvariableopIsavev2_transformer_encoder_layer_normalization_1_beta_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_dem_time_dist_conv2d_kernel_read_readvariableop=savev2_adam_v_dem_time_dist_conv2d_kernel_read_readvariableop;savev2_adam_m_dem_time_dist_conv2d_bias_read_readvariableop;savev2_adam_v_dem_time_dist_conv2d_bias_read_readvariableop>savev2_adam_m_temp_time_dist_conv2d_kernel_read_readvariableop>savev2_adam_v_temp_time_dist_conv2d_kernel_read_readvariableop<savev2_adam_m_temp_time_dist_conv2d_bias_read_readvariableop<savev2_adam_v_temp_time_dist_conv2d_bias_read_readvariableop@savev2_adam_m_precip_time_dist_conv2d_kernel_read_readvariableop@savev2_adam_v_precip_time_dist_conv2d_kernel_read_readvariableop>savev2_adam_m_precip_time_dist_conv2d_bias_read_readvariableop>savev2_adam_v_precip_time_dist_conv2d_bias_read_readvariableop=savev2_adam_m_swe_time_dist_conv2d_kernel_read_readvariableop=savev2_adam_v_swe_time_dist_conv2d_kernel_read_readvariableop;savev2_adam_m_swe_time_dist_conv2d_bias_read_readvariableop;savev2_adam_v_swe_time_dist_conv2d_bias_read_readvariableop<savev2_adam_m_et_time_dist_conv2d_kernel_read_readvariableop<savev2_adam_v_et_time_dist_conv2d_kernel_read_readvariableop:savev2_adam_m_et_time_dist_conv2d_bias_read_readvariableop:savev2_adam_v_et_time_dist_conv2d_bias_read_readvariableopWsavev2_adam_m_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopWsavev2_adam_v_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_query_bias_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_query_bias_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopSsavev2_adam_m_transformer_encoder_multi_head_attention_key_bias_read_readvariableopSsavev2_adam_v_transformer_encoder_multi_head_attention_key_bias_read_readvariableopWsavev2_adam_m_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopWsavev2_adam_v_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopUsavev2_adam_m_transformer_encoder_multi_head_attention_value_bias_read_readvariableopUsavev2_adam_v_transformer_encoder_multi_head_attention_value_bias_read_readvariableopbsavev2_adam_m_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableopbsavev2_adam_v_transformer_encoder_multi_head_attention_attention_output_kernel_read_readvariableop`savev2_adam_m_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop`savev2_adam_v_transformer_encoder_multi_head_attention_attention_output_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableopOsavev2_adam_m_transformer_encoder_layer_normalization_gamma_read_readvariableopOsavev2_adam_v_transformer_encoder_layer_normalization_gamma_read_readvariableopNsavev2_adam_m_transformer_encoder_layer_normalization_beta_read_readvariableopNsavev2_adam_v_transformer_encoder_layer_normalization_beta_read_readvariableopQsavev2_adam_m_transformer_encoder_layer_normalization_1_gamma_read_readvariableopQsavev2_adam_v_transformer_encoder_layer_normalization_1_gamma_read_readvariableopPsavev2_adam_m_transformer_encoder_layer_normalization_1_beta_read_readvariableopPsavev2_adam_v_transformer_encoder_layer_normalization_1_beta_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *g
dtypes]
[2Y	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
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

identity_1Identity_1:output:0*
_input_shapesď
ě: :
::úŢ::::::::!'::

:
:

:
:

:
:

:
:
 : : 
:
:
:
:
:
: : :úŢ:úŢ:::::::::::::::!':!':::

:

:
:
:

:

:
:
:

:

:
:
:

:

:
:
:
 :
 : : : 
: 
:
:
:
:
:
:
:
:
:
:
:
:
::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
::.*
(
_output_shapes
:úŢ: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::.	*
(
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:!': 

_output_shapes
::($
"
_output_shapes
:

:$ 

_output_shapes

:
:($
"
_output_shapes
:

:$ 

_output_shapes

:
:($
"
_output_shapes
:

:$ 

_output_shapes

:
:($
"
_output_shapes
:

: 

_output_shapes
:
:$ 

_output_shapes

:
 : 

_output_shapes
: :$ 

_output_shapes

: 
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:úŢ:. *
(
_output_shapes
:úŢ: !

_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::.+*
(
_output_shapes
::.,*
(
_output_shapes
:: -

_output_shapes
:: .

_output_shapes
::,/(
&
_output_shapes
:!':,0(
&
_output_shapes
:!': 1

_output_shapes
:: 2

_output_shapes
::(3$
"
_output_shapes
:

:(4$
"
_output_shapes
:

:$5 

_output_shapes

:
:$6 

_output_shapes

:
:(7$
"
_output_shapes
:

:(8$
"
_output_shapes
:

:$9 

_output_shapes

:
:$: 

_output_shapes

:
:(;$
"
_output_shapes
:

:(<$
"
_output_shapes
:

:$= 

_output_shapes

:
:$> 

_output_shapes

:
:(?$
"
_output_shapes
:

:(@$
"
_output_shapes
:

: A

_output_shapes
:
: B

_output_shapes
:
:$C 

_output_shapes

:
 :$D 

_output_shapes

:
 : E

_output_shapes
: : F

_output_shapes
: :$G 

_output_shapes

: 
:$H 

_output_shapes

: 
: I

_output_shapes
:
: J

_output_shapes
:
: K

_output_shapes
:
: L

_output_shapes
:
: M

_output_shapes
:
: N

_output_shapes
:
: O

_output_shapes
:
: P

_output_shapes
:
: Q

_output_shapes
:
: R

_output_shapes
:
:$S 

_output_shapes

:
:$T 

_output_shapes

:
: U
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
É
ů
B__inference_dense_1_layer_call_and_return_conditional_losses_56282

inputs3
!tensordot_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : ť
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
value	B : ż
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
value	B : 
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
:˙˙˙˙˙˙˙˙˙ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Á
Ő
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55531

inputs(
conv2d_55519:úŢ
conv2d_55521:
identity˘conv2d/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙÷       n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷÷
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_55519conv2d_55521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_55518\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
Â
^
B__inference_flatten_layer_call_and_return_conditional_losses_55943

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î

'__inference_dense_1_layer_call_fn_59759

inputs
unknown: 

	unknown_0:

identity˘StatefulPartitionedCallŰ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_56282s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ă
F
*__inference_et_flatten_layer_call_fn_58876

inputs
identity˝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56178m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â
^
B__inference_flatten_layer_call_and_return_conditional_losses_59526

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58722

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   v
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
é
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56091

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_3/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_56057\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_3/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56007

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_56000\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
Ž
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58605

inputsC
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity˘conv2d_4/BiasAdd/ReadVariableOp˘conv2d_4/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙Ă   Ó     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ś
conv2d_4/Conv2DConv2DReshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AM
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_4/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_56814

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ś
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Č=
Ě
E__inference_sequential_layer_call_and_return_conditional_losses_59710

inputs9
'dense_tensordot_readvariableop_resource:
 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 
5
'dense_1_biasadd_readvariableop_resource:

identity˘dense/BiasAdd/ReadVariableOp˘dense/Tensordot/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : Ó
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
value	B : ×
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
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : Ű
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
value	B : ß
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
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ů
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58827

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_3/ReshapeReshapeReshape:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_3/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
A__inference_conv2d_layer_call_and_return_conditional_losses_55518

inputs:
conv2d_readvariableop_resource:úŢ-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙÷: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs

÷
@__inference_dense_layer_call_and_return_conditional_losses_59750

inputs3
!tensordot_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ť
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
value	B : ż
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
value	B : 
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
:˙˙˙˙˙˙˙˙˙

Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ë
J
.__inference_precip_flatten_layer_call_fn_58793

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56091m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_56057

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58783

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_2/ReshapeReshapeReshape:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_2/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
ł
3__inference_transformer_encoder_layer_call_fn_58971

inputs
unknown:


	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
 

unknown_10: 

unknown_11: 


unknown_12:


unknown_13:


unknown_14:

identity˘StatefulPartitionedCall˘
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
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_56666s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Đ
Ú
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55617

inputs(
conv2d_2_55605:
conv2d_2_55607:
identity˘ conv2d_2/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˙
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_2_55605conv2d_2_55607*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55604\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
Í
*__inference_sequential_layer_call_fn_59596

inputs
unknown:
 
	unknown_0: 
	unknown_1: 

	unknown_2:

identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_56349s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ő
`
B__inference_dropout_layer_call_and_return_conditional_losses_59384

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ť
Ź
7__inference_precip_time_dist_conv2d_layer_call_fn_58515

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55744
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Á
Ő
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55572

inputs(
conv2d_55560:úŢ
conv2d_55562:
identity˘conv2d/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙÷       n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷÷
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_55560conv2d_55562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_55518\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
ŐŢ
­
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59183

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_query_add_readvariableop_resource:
T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:

F
4multi_head_attention_key_add_readvariableop_resource:
V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_value_add_readvariableop_resource:
a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:

O
Amulti_head_attention_attention_output_add_readvariableop_resource:
?
1layer_normalization_mul_3_readvariableop_resource:
=
/layer_normalization_add_readvariableop_resource:
D
2sequential_dense_tensordot_readvariableop_resource:
 >
0sequential_dense_biasadd_readvariableop_resource: F
4sequential_dense_1_tensordot_readvariableop_resource: 
@
2sequential_dense_1_biasadd_readvariableop_resource:
A
3layer_normalization_1_mul_3_readvariableop_resource:
?
1layer_normalization_1_add_readvariableop_resource:

identity˘&layer_normalization/add/ReadVariableOp˘(layer_normalization/mul_3/ReadVariableOp˘(layer_normalization_1/add/ReadVariableOp˘*layer_normalization_1/mul_3/ReadVariableOp˘8multi_head_attention/attention_output/add/ReadVariableOp˘Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘+multi_head_attention/key/add/ReadVariableOp˘5multi_head_attention/key/einsum/Einsum/ReadVariableOp˘-multi_head_attention/query/add/ReadVariableOp˘7multi_head_attention/query/einsum/Einsum/ReadVariableOp˘-multi_head_attention/value/add/ReadVariableOp˘7multi_head_attention/value/einsum/Einsum/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOpź
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0×
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ĺ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>˘
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdŇ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeś
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0č
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
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
valueB:ľ
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
value	B :
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
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:˝
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
value	B :
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
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
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
 *  ?Ą
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙u
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
 *    ¤
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:Ź
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0°
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ľ
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ˙
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
value	B : 
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
valueB: Ą
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ŕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ź
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˝
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ś
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ż
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙  
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : 
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
value	B : 
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
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:˛
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ź
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ă
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ź
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ľ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
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
valueB:ż
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
value	B :
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
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ç
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
value	B :
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
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
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
 *  ?§
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙y
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
 *    Ş
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB §
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:˛
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ś
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ť
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
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
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ĺ
G
+__inference_dem_flatten_layer_call_fn_58705

inputs
identityž
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55977m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
äf
Đ
@__inference_model_layer_call_and_return_conditional_losses_57470

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs3
et_time_dist_conv2d_57378:!''
et_time_dist_conv2d_57380:6
swe_time_dist_conv2d_57385:(
swe_time_dist_conv2d_57387:7
precip_time_dist_conv2d_57392:+
precip_time_dist_conv2d_57394:5
temp_time_dist_conv2d_57399:)
temp_time_dist_conv2d_57401:6
dem_time_dist_conv2d_57406:úŢ(
dem_time_dist_conv2d_57408:/
transformer_encoder_57429:

+
transformer_encoder_57431:
/
transformer_encoder_57433:

+
transformer_encoder_57435:
/
transformer_encoder_57437:

+
transformer_encoder_57439:
/
transformer_encoder_57441:

'
transformer_encoder_57443:
'
transformer_encoder_57445:
'
transformer_encoder_57447:
+
transformer_encoder_57449:
 '
transformer_encoder_57451: +
transformer_encoder_57453: 
'
transformer_encoder_57455:
'
transformer_encoder_57457:
'
transformer_encoder_57459:

dense_2_57464:

dense_2_57466:
identity˘,dem_time_dist_conv2d/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘+et_time_dist_conv2d/StatefulPartitionedCall˘/precip_time_dist_conv2d/StatefulPartitionedCall˘,swe_time_dist_conv2d/StatefulPartitionedCall˘-temp_time_dist_conv2d/StatefulPartitionedCall˘+transformer_encoder/StatefulPartitionedCall¨
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall	et_inputset_time_dist_conv2d_57378et_time_dist_conv2d_57380*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55875z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshape	et_inputs*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w­
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
swe_inputsswe_time_dist_conv2d_57385swe_time_dist_conv2d_57387*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55789{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshape
swe_inputs+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓź
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallprecip_inputsprecip_time_dist_conv2d_57392precip_time_dist_conv2d_57394*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55703~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ł
precip_time_dist_conv2d/ReshapeReshapeprecip_inputs.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCalltemp_inputstemp_time_dist_conv2d_57399temp_time_dist_conv2d_57401*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55617|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapetemp_inputs,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙­
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
dem_inputsdem_time_dist_conv2d_57406dem_time_dist_conv2d_57408*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55531{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshape
dem_inputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷đ
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55950r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ó
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56007s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ś
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ů
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56064u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ź
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56121r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙í
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56178q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         °
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ţ
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_56489Ń
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_57429transformer_encoder_57431transformer_encoder_57433transformer_encoder_57435transformer_encoder_57437transformer_encoder_57439transformer_encoder_57441transformer_encoder_57443transformer_encoder_57445transformer_encoder_57447transformer_encoder_57449transformer_encoder_57451transformer_encoder_57453transformer_encoder_57455transformer_encoder_57457transformer_encoder_57459*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_56666ý
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411Ü
dropout/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56706
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_57464dense_2_57466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_56718w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:a ]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs
őĆ
í!
@__inference_model_layer_call_and_return_conditional_losses_58365
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4U
;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:!'J
<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:X
<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:K
=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:Y
?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:N
@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:W
=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:L
>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:V
:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:úŢI
;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:j
Ttransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:

\
Jtransformer_encoder_multi_head_attention_query_add_readvariableop_resource:
h
Rtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:

Z
Htransformer_encoder_multi_head_attention_key_add_readvariableop_resource:
j
Ttransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:

\
Jtransformer_encoder_multi_head_attention_value_add_readvariableop_resource:
u
_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:

c
Utransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:
S
Etransformer_encoder_layer_normalization_mul_3_readvariableop_resource:
Q
Ctransformer_encoder_layer_normalization_add_readvariableop_resource:
X
Ftransformer_encoder_sequential_dense_tensordot_readvariableop_resource:
 R
Dtransformer_encoder_sequential_dense_biasadd_readvariableop_resource: Z
Htransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource: 
T
Ftransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:
U
Gtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:
S
Etransformer_encoder_layer_normalization_1_add_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identity˘2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp˘1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp˘3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp˘2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp˘7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp˘6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp˘4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp˘3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp˘5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp˘4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp˘:transformer_encoder/layer_normalization/add/ReadVariableOp˘<transformer_encoder/layer_normalization/mul_3/ReadVariableOp˘<transformer_encoder/layer_normalization_1/add/ReadVariableOp˘>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp˘Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp˘Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘?transformer_encoder/multi_head_attention/key/add/ReadVariableOp˘Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp˘Atransformer_encoder/multi_head_attention/query/add/ReadVariableOp˘Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp˘Atransformer_encoder/multi_head_attention/value/add/ReadVariableOp˘Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp˘;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp˘=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp˘=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp˘?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpz
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2wś
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0ň
#et_time_dist_conv2d/conv2d_1/Conv2DConv2D$et_time_dist_conv2d/Reshape:output:0:et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
Ź
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ô
$et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd,et_time_dist_conv2d/conv2d_1/Conv2D:output:0;et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!et_time_dist_conv2d/conv2d_1/ReluRelu-et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
#et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ĺ
et_time_dist_conv2d/Reshape_1Reshape/et_time_dist_conv2d/conv2d_1/Relu:activations:0,et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙|
#et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/Reshape_2Reshapeinputs_4,et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓş
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ő
$swe_time_dist_conv2d/conv2d_4/Conv2DConv2D%swe_time_dist_conv2d/Reshape:output:0;swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AMŽ
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
%swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd-swe_time_dist_conv2d/conv2d_4/Conv2D:output:0<swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"swe_time_dist_conv2d/conv2d_4/ReluRelu.swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
$swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Č
swe_time_dist_conv2d/Reshape_1Reshape0swe_time_dist_conv2d/conv2d_4/Relu:activations:0-swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙}
$swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/Reshape_2Reshapeinputs_3-swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ţ
'precip_time_dist_conv2d/conv2d_3/Conv2DConv2D(precip_time_dist_conv2d/Reshape:output:0>precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
´
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ŕ
(precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd0precip_time_dist_conv2d/conv2d_3/Conv2D:output:0?precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%precip_time_dist_conv2d/conv2d_3/ReluRelu1precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
'precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ń
!precip_time_dist_conv2d/Reshape_1Reshape3precip_time_dist_conv2d/conv2d_3/Relu:activations:00precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˘
!precip_time_dist_conv2d/Reshape_2Reshapeinputs_20precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ř
%temp_time_dist_conv2d/conv2d_2/Conv2DConv2D&temp_time_dist_conv2d/Reshape:output:0<temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
°
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ú
&temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd.temp_time_dist_conv2d/conv2d_2/Conv2D:output:0=temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
#temp_time_dist_conv2d/conv2d_2/ReluRelu/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ë
temp_time_dist_conv2d/Reshape_1Reshape1temp_time_dist_conv2d/conv2d_2/Relu:activations:0.temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙~
%temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/Reshape_2Reshapeinputs_1.temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshapeinputs_0+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷ś
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0ó
"dem_time_dist_conv2d/conv2d/Conv2DConv2D%dem_time_dist_conv2d/Reshape:output:09dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻŞ
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOp;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ń
#dem_time_dist_conv2d/conv2d/BiasAddBiasAdd+dem_time_dist_conv2d/conv2d/Conv2D:output:0:dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 dem_time_dist_conv2d/conv2d/ReluRelu,dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ć
dem_time_dist_conv2d/Reshape_1Reshape.dem_time_dist_conv2d/conv2d/Relu:activations:0-dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙}
$dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/Reshape_2Reshapeinputs_0-dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ľ
dem_flatten/ReshapeReshape'dem_time_dist_conv2d/Reshape_1:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
dem_flatten/flatten/ReshapeReshapedem_flatten/Reshape:output:0"dem_flatten/flatten/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ˘
dem_flatten/Reshape_1Reshape$dem_flatten/flatten/Reshape:output:0$dem_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Š
dem_flatten/Reshape_2Reshape'dem_time_dist_conv2d/Reshape_1:output:0$dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ¨
temp_flatten/ReshapeReshape(temp_time_dist_conv2d/Reshape_1:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙m
temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   Ą
temp_flatten/flatten_2/ReshapeReshapetemp_flatten/Reshape:output:0%temp_flatten/flatten_2/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   §
temp_flatten/Reshape_1Reshape'temp_flatten/flatten_2/Reshape:output:0%temp_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ź
temp_flatten/Reshape_2Reshape(temp_time_dist_conv2d/Reshape_1:output:0%temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ž
precip_flatten/ReshapeReshape*precip_time_dist_conv2d/Reshape_1:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙o
precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   §
 precip_flatten/flatten_3/ReshapeReshapeprecip_flatten/Reshape:output:0'precip_flatten/flatten_3/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ­
precip_flatten/Reshape_1Reshape)precip_flatten/flatten_3/Reshape:output:0'precip_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˛
precip_flatten/Reshape_2Reshape*precip_time_dist_conv2d/Reshape_1:output:0'precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ľ
swe_flatten/ReshapeReshape'swe_time_dist_conv2d/Reshape_1:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙l
swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
swe_flatten/flatten_4/ReshapeReshapeswe_flatten/Reshape:output:0$swe_flatten/flatten_4/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ¤
swe_flatten/Reshape_1Reshape&swe_flatten/flatten_4/Reshape:output:0$swe_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Š
swe_flatten/Reshape_2Reshape'swe_time_dist_conv2d/Reshape_1:output:0$swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˘
et_flatten/ReshapeReshape&et_time_dist_conv2d/Reshape_1:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙k
et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
et_flatten/flatten_1/ReshapeReshapeet_flatten/Reshape:output:0#et_flatten/flatten_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   Ą
et_flatten/Reshape_1Reshape%et_flatten/flatten_1/Reshape:output:0#et_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ś
et_flatten/Reshape_2Reshape&et_time_dist_conv2d/Reshape_1:output:0#et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :˘
concatenate/concatConcatV2dem_flatten/Reshape_1:output:0temp_flatten/Reshape_1:output:0!precip_flatten/Reshape_1:output:0swe_flatten/Reshape_1:output:0et_flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
<transformer_encoder/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeĚ
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0
2transformer_encoder/multi_head_attention/query/addAddV2Etransformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
:transformer_encoder/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeČ
?transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpHtransformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0
0transformer_encoder/multi_head_attention/key/addAddV2Ctransformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Gtransformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
<transformer_encoder/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeĚ
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0
2transformer_encoder/multi_head_attention/value/addAddV2Etransformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
.transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>Ţ
,transformer_encoder/multi_head_attention/MulMul6transformer_encoder/multi_head_attention/query/add:z:07transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

6transformer_encoder/multi_head_attention/einsum/EinsumEinsum4transformer_encoder/multi_head_attention/key/add:z:00transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbež
8transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmax?transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
9transformer_encoder/multi_head_attention/dropout/IdentityIdentityBtransformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumBtransformer_encoder/multi_head_attention/dropout/Identity:output:06transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdú
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Đ
Gtransformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeŢ
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpUtransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0¤
=transformer_encoder/multi_head_attention/attention_output/addAddV2Ptransformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Ttransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
transformer_encoder/addAddV2concatenate/concat:output:0Atransformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
-transformer_encoder/layer_normalization/ShapeShapetransformer_encoder/add:z:0*
T0*
_output_shapes
:
;transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
value	B :Ë
+transformer_encoder/layer_normalization/mulMul6transformer_encoder/layer_normalization/mul/x:output:0>transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 
=transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
7transformer_encoder/layer_normalization/strided_slice_1StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_1/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskČ
-transformer_encoder/layer_normalization/mul_1Mul/transformer_encoder/layer_normalization/mul:z:0@transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 
=transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
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
value	B :Ń
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
value	B :Ő
5transformer_encoder/layer_normalization/Reshape/shapePack@transformer_encoder/layer_normalization/Reshape/shape/0:output:01transformer_encoder/layer_normalization/mul_1:z:01transformer_encoder/layer_normalization/mul_2:z:0@transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ń
/transformer_encoder/layer_normalization/ReshapeReshapetransformer_encoder/add:z:0>transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
 *  ?Ý
,transformer_encoder/layer_normalization/onesFill<transformer_encoder/layer_normalization/ones/packed:output:0;transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
 *    ŕ
-transformer_encoder/layer_normalization/zerosFill=transformer_encoder/layer_normalization/zeros/packed:output:0<transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙p
-transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB r
/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
8transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV38transformer_encoder/layer_normalization/Reshape:output:05transformer_encoder/layer_normalization/ones:output:06transformer_encoder/layer_normalization/zeros:output:06transformer_encoder/layer_normalization/Const:output:08transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:č
1transformer_encoder/layer_normalization/Reshape_1Reshape<transformer_encoder/layer_normalization/FusedBatchNormV3:y:06transformer_encoder/layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
<transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ě
-transformer_encoder/layer_normalization/mul_3Mul:transformer_encoder/layer_normalization/Reshape_1:output:0Dtransformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
:transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpCtransformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0á
+transformer_encoder/layer_normalization/addAddV21transformer_encoder/layer_normalization/mul_3:z:0Btransformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
dtype0}
3transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4transformer_encoder/sequential/dense/Tensordot/ShapeShape/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:~
<transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ď
7transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/free:output:0Etransformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
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
valueB: Ý
3transformer_encoder/sequential/dense/Tensordot/ProdProd@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0=transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ă
5transformer_encoder/sequential/dense/Tensordot/Prod_1ProdBtransformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0?transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : °
5transformer_encoder/sequential/dense/Tensordot/concatConcatV2<transformer_encoder/sequential/dense/Tensordot/free:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Ctransformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:č
4transformer_encoder/sequential/dense/Tensordot/stackPack<transformer_encoder/sequential/dense/Tensordot/Prod:output:0>transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ě
8transformer_encoder/sequential/dense/Tensordot/transpose	Transpose/transformer_encoder/layer_normalization/add:z:0>transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
6transformer_encoder/sequential/dense/Tensordot/ReshapeReshape<transformer_encoder/sequential/dense/Tensordot/transpose:y:0=transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ů
5transformer_encoder/sequential/dense/Tensordot/MatMulMatMul?transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Etransformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
6transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ~
<transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
7transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Etransformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ň
.transformer_encoder/sequential/dense/TensordotReshape?transformer_encoder/sequential/dense/Tensordot/MatMul:product:0@transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ź
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpDtransformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ë
,transformer_encoder/sequential/dense/BiasAddBiasAdd7transformer_encoder/sequential/dense/Tensordot:output:0Ctransformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)transformer_encoder/sequential/dense/ReluRelu5transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Č
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpHtransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
dtype0
5transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
5transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
6transformer_encoder/sequential/dense_1/Tensordot/ShapeShape7transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:
>transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
9transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/free:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
@transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
;transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Itransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
6transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ă
5transformer_encoder/sequential/dense_1/Tensordot/ProdProdBtransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
8transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: é
7transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdDtransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
7transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2>transformer_encoder/sequential/dense_1/Tensordot/free:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Etransformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:î
6transformer_encoder/sequential/dense_1/Tensordot/stackPack>transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0@transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ř
:transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose7transformer_encoder/sequential/dense/Relu:activations:0@transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ˙
8transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshape>transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0?transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
7transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulAtransformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:

>transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ă
9transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Btransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ř
0transformer_encoder/sequential/dense_1/TensordotReshapeAtransformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Btransformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ń
.transformer_encoder/sequential/dense_1/BiasAddBiasAdd9transformer_encoder/sequential/dense_1/Tensordot:output:0Etransformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
transformer_encoder/add_1AddV2/transformer_encoder/layer_normalization/add:z:07transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
/transformer_encoder/layer_normalization_1/ShapeShapetransformer_encoder/add_1:z:0*
T0*
_output_shapes
:
=transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
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
value	B :Ń
-transformer_encoder/layer_normalization_1/mulMul8transformer_encoder/layer_normalization_1/mul/x:output:0@transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 
?transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
9transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÎ
/transformer_encoder/layer_normalization_1/mul_1Mul1transformer_encoder/layer_normalization_1/mul:z:0Btransformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 
?transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
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
value	B :×
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
value	B :ß
7transformer_encoder/layer_normalization_1/Reshape/shapePackBtransformer_encoder/layer_normalization_1/Reshape/shape/0:output:03transformer_encoder/layer_normalization_1/mul_1:z:03transformer_encoder/layer_normalization_1/mul_2:z:0Btransformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:×
1transformer_encoder/layer_normalization_1/ReshapeReshapetransformer_encoder/add_1:z:0@transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
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
 *  ?ă
.transformer_encoder/layer_normalization_1/onesFill>transformer_encoder/layer_normalization_1/ones/packed:output:0=transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
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
 *    ć
/transformer_encoder/layer_normalization_1/zerosFill?transformer_encoder/layer_normalization_1/zeros/packed:output:0>transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙r
/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB t
1transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
:transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3:transformer_encoder/layer_normalization_1/Reshape:output:07transformer_encoder/layer_normalization_1/ones:output:08transformer_encoder/layer_normalization_1/zeros:output:08transformer_encoder/layer_normalization_1/Const:output:0:transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:î
3transformer_encoder/layer_normalization_1/Reshape_1Reshape>transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:08transformer_encoder/layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpGtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ň
/transformer_encoder/layer_normalization_1/mul_3Mul<transformer_encoder/layer_normalization_1/Reshape_1:output:0Ftransformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
<transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0ç
-transformer_encoder/layer_normalization_1/addAddV23transformer_encoder/layer_normalization_1/mul_3:z:0Dtransformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :š
global_max_pooling1d/MaxMax1transformer_encoder/layer_normalization_1/add:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMul!global_max_pooling1d/Max:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
dropout/dropout/ShapeShape!global_max_pooling1d/Max:output:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ž
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ł
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_2/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp3^dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2^dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp4^et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3^et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp8^precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7^precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp5^swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4^swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp6^temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5^temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp;^transformer_encoder/layer_normalization/add/ReadVariableOp=^transformer_encoder/layer_normalization/mul_3/ReadVariableOp=^transformer_encoder/layer_normalization_1/add/ReadVariableOp?^transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpM^transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpW^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp@^transformer_encoder/multi_head_attention/key/add/ReadVariableOpJ^transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/query/add/ReadVariableOpL^transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/value/add/ReadVariableOpL^transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp<^transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp>^transformer_encoder/sequential/dense/Tensordot/ReadVariableOp>^transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp@^transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
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
<transformer_encoder/layer_normalization_1/add/ReadVariableOp<transformer_encoder/layer_normalization_1/add/ReadVariableOp2
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpLtransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2°
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpVtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2
?transformer_encoder/multi_head_attention/key/add/ReadVariableOp?transformer_encoder/multi_head_attention/key/add/ReadVariableOp2
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpItransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpAtransformer_encoder/multi_head_attention/query/add/ReadVariableOp2
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpAtransformer_encoder/multi_head_attention/value/add/ReadVariableOp2
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2z
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2~
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
"
_user_specified_name
inputs_4
 f
Á
@__inference_model_layer_call_and_return_conditional_losses_56725

inputs
inputs_1
inputs_2
inputs_3
inputs_43
et_time_dist_conv2d_56429:!''
et_time_dist_conv2d_56431:6
swe_time_dist_conv2d_56436:(
swe_time_dist_conv2d_56438:7
precip_time_dist_conv2d_56443:+
precip_time_dist_conv2d_56445:5
temp_time_dist_conv2d_56450:)
temp_time_dist_conv2d_56452:6
dem_time_dist_conv2d_56457:úŢ(
dem_time_dist_conv2d_56459:/
transformer_encoder_56667:

+
transformer_encoder_56669:
/
transformer_encoder_56671:

+
transformer_encoder_56673:
/
transformer_encoder_56675:

+
transformer_encoder_56677:
/
transformer_encoder_56679:

'
transformer_encoder_56681:
'
transformer_encoder_56683:
'
transformer_encoder_56685:
+
transformer_encoder_56687:
 '
transformer_encoder_56689: +
transformer_encoder_56691: 
'
transformer_encoder_56693:
'
transformer_encoder_56695:
'
transformer_encoder_56697:

dense_2_56719:

dense_2_56721:
identity˘,dem_time_dist_conv2d/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘+et_time_dist_conv2d/StatefulPartitionedCall˘/precip_time_dist_conv2d/StatefulPartitionedCall˘,swe_time_dist_conv2d/StatefulPartitionedCall˘-temp_time_dist_conv2d/StatefulPartitionedCall˘+transformer_encoder/StatefulPartitionedCall§
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_4et_time_dist_conv2d_56429et_time_dist_conv2d_56431*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55875z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2wŤ
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_3swe_time_dist_conv2d_56436swe_time_dist_conv2d_56438*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55789{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓˇ
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_2precip_time_dist_conv2d_56443precip_time_dist_conv2d_56445*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55703~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1temp_time_dist_conv2d_56450temp_time_dist_conv2d_56452*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55617|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsdem_time_dist_conv2d_56457dem_time_dist_conv2d_56459*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55531{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshapeinputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷đ
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55950r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ó
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56007s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ś
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ů
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56064u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ź
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56121r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙í
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56178q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         °
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ţ
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_56489Ń
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_56667transformer_encoder_56669transformer_encoder_56671transformer_encoder_56673transformer_encoder_56675transformer_encoder_56677transformer_encoder_56679transformer_encoder_56681transformer_encoder_56683transformer_encoder_56685transformer_encoder_56687transformer_encoder_56689transformer_encoder_56691transformer_encoder_56693transformer_encoder_56695transformer_encoder_56697*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_56666ý
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411Ü
dropout/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56706
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_56719dense_2_56721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_56718w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,dem_time_dist_conv2d/StatefulPartitionedCall,dem_time_dist_conv2d/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+et_time_dist_conv2d/StatefulPartitionedCall+et_time_dist_conv2d/StatefulPartitionedCall2b
/precip_time_dist_conv2d/StatefulPartitionedCall/precip_time_dist_conv2d/StatefulPartitionedCall2\
,swe_time_dist_conv2d/StatefulPartitionedCall,swe_time_dist_conv2d/StatefulPartitionedCall2^
-temp_time_dist_conv2d/StatefulPartitionedCall-temp_time_dist_conv2d/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs:[W
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Š
C
'__inference_flatten_layer_call_fn_59520

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_55943`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
×
Ű
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55789

inputs*
conv2d_4_55777:
conv2d_4_55779:
identity˘ conv2d_4/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙Ă   Ó     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ˙
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_4_55777conv2d_4_55779*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55776\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_4/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
Ę
Ť
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58671

inputsA
'conv2d_1_conv2d_readvariableop_resource:!'6
(conv2d_1_biasadd_readvariableop_resource:
identity˘conv2d_1/BiasAdd/ReadVariableOp˘conv2d_1/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙2   w      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0ś
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Đ
ł
3__inference_transformer_encoder_layer_call_fn_59008

inputs
unknown:


	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:


	unknown_6:

	unknown_7:

	unknown_8:

	unknown_9:
 

unknown_10: 

unknown_11: 


unknown_12:


unknown_13:


unknown_14:

identity˘StatefulPartitionedCall˘
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
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_57032s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ŐŢ
­
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_57032

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_query_add_readvariableop_resource:
T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:

F
4multi_head_attention_key_add_readvariableop_resource:
V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:

H
6multi_head_attention_value_add_readvariableop_resource:
a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:

O
Amulti_head_attention_attention_output_add_readvariableop_resource:
?
1layer_normalization_mul_3_readvariableop_resource:
=
/layer_normalization_add_readvariableop_resource:
D
2sequential_dense_tensordot_readvariableop_resource:
 >
0sequential_dense_biasadd_readvariableop_resource: F
4sequential_dense_1_tensordot_readvariableop_resource: 
@
2sequential_dense_1_biasadd_readvariableop_resource:
A
3layer_normalization_1_mul_3_readvariableop_resource:
?
1layer_normalization_1_add_readvariableop_resource:

identity˘&layer_normalization/add/ReadVariableOp˘(layer_normalization/mul_3/ReadVariableOp˘(layer_normalization_1/add/ReadVariableOp˘*layer_normalization_1/mul_3/ReadVariableOp˘8multi_head_attention/attention_output/add/ReadVariableOp˘Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘+multi_head_attention/key/add/ReadVariableOp˘5multi_head_attention/key/einsum/Einsum/ReadVariableOp˘-multi_head_attention/query/add/ReadVariableOp˘7multi_head_attention/query/einsum/Einsum/ReadVariableOp˘-multi_head_attention/value/add/ReadVariableOp˘7multi_head_attention/value/einsum/Einsum/ReadVariableOp˘'sequential/dense/BiasAdd/ReadVariableOp˘)sequential/dense/Tensordot/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘+sequential/dense_1/Tensordot/ReadVariableOpź
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0×
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ĺ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ű
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0Ë
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>˘
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdŇ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeś
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0č
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
addAddV2inputs-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
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
valueB:ľ
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
value	B :
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
valueB:˝
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:˝
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
value	B :
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
value	B :ń
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
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
 *  ?Ą
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙u
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
 *    ¤
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:Ź
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0°
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ľ
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ˙
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
value	B : 
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
valueB: Ą
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ŕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ź
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$sequential/dense/Tensordot/transpose	Transposelayer_normalization/add:z:0*sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˝
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ś
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ż
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ v
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙  
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : 
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
value	B : 
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
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:˛
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ź
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ă
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ź
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ľ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

add_1AddV2layer_normalization/add:z:0#sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
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
valueB:ż
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
value	B :
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
valueB:Ç
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ç
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
value	B :
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
value	B :ű
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
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
 *  ?§
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙y
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
 *    Ş
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB §
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:˛
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ś
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0Ť
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
IdentityIdentitylayer_normalization_1/add:z:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙
: : : : : : : : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
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
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ń
P
4__inference_global_max_pooling1d_layer_call_fn_59363

inputs
identityĂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:e a
=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î

(__inference_conv2d_4_layer_call_fn_59484

inputs#
unknown:
	unknown_0:
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55776w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs

ü
A__inference_conv2d_layer_call_and_return_conditional_losses_59435

inputs:
conv2d_readvariableop_resource:úŢ-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙÷: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs

č
%__inference_model_layer_call_fn_57371

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs!
unknown:!'
	unknown_0:%
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:úŢ
	unknown_8:
	unknown_9:



unknown_10:
 

unknown_11:



unknown_12:
 

unknown_13:



unknown_14:
 

unknown_15:



unknown_16:


unknown_17:


unknown_18:


unknown_19:
 

unknown_20: 

unknown_21: 


unknown_22:


unknown_23:


unknown_24:


unknown_25:


unknown_26:
identity˘StatefulPartitionedCalló
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
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_57247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs
ť
Ť
4__inference_dem_time_dist_conv2d_layer_call_fn_58383

inputs#
unknown:úŢ
	unknown_0:
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55572
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
č

(__inference_conv2d_1_layer_call_fn_59504

inputs!
unknown:!'
	unknown_0:
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55862w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙2w: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_59570

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ö
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58854

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_4/ReshapeReshapeReshape:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_4/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58766

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_2/ReshapeReshapeReshape:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_2/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
H
,__inference_temp_flatten_layer_call_fn_58744

inputs
identityż
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56007m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Ş
5__inference_temp_time_dist_conv2d_layer_call_fn_58440

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55617
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
Ť
4__inference_dem_time_dist_conv2d_layer_call_fn_58374

inputs#
unknown:úŢ
	unknown_0:
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55531
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_56114

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ć
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56148

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_4/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_56114\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_4/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
h
ň
@__inference_model_layer_call_and_return_conditional_losses_57569

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs3
et_time_dist_conv2d_57477:!''
et_time_dist_conv2d_57479:6
swe_time_dist_conv2d_57484:(
swe_time_dist_conv2d_57486:7
precip_time_dist_conv2d_57491:+
precip_time_dist_conv2d_57493:5
temp_time_dist_conv2d_57498:)
temp_time_dist_conv2d_57500:6
dem_time_dist_conv2d_57505:úŢ(
dem_time_dist_conv2d_57507:/
transformer_encoder_57528:

+
transformer_encoder_57530:
/
transformer_encoder_57532:

+
transformer_encoder_57534:
/
transformer_encoder_57536:

+
transformer_encoder_57538:
/
transformer_encoder_57540:

'
transformer_encoder_57542:
'
transformer_encoder_57544:
'
transformer_encoder_57546:
+
transformer_encoder_57548:
 '
transformer_encoder_57550: +
transformer_encoder_57552: 
'
transformer_encoder_57554:
'
transformer_encoder_57556:
'
transformer_encoder_57558:

dense_2_57563:

dense_2_57565:
identity˘,dem_time_dist_conv2d/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘+et_time_dist_conv2d/StatefulPartitionedCall˘/precip_time_dist_conv2d/StatefulPartitionedCall˘,swe_time_dist_conv2d/StatefulPartitionedCall˘-temp_time_dist_conv2d/StatefulPartitionedCall˘+transformer_encoder/StatefulPartitionedCall¨
+et_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall	et_inputset_time_dist_conv2d_57477et_time_dist_conv2d_57479*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55916z
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshape	et_inputs*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w­
,swe_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
swe_inputsswe_time_dist_conv2d_57484swe_time_dist_conv2d_57486*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55830{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshape
swe_inputs+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓź
/precip_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCallprecip_inputsprecip_time_dist_conv2d_57491precip_time_dist_conv2d_57493*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55744~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ł
precip_time_dist_conv2d/ReshapeReshapeprecip_inputs.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
-temp_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCalltemp_inputstemp_time_dist_conv2d_57498temp_time_dist_conv2d_57500*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55658|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapetemp_inputs,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙­
,dem_time_dist_conv2d/StatefulPartitionedCallStatefulPartitionedCall
dem_inputsdem_time_dist_conv2d_57505dem_time_dist_conv2d_57507*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_55572{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshape
dem_inputs+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷đ
dem_flatten/PartitionedCallPartitionedCall5dem_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55977r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
dem_flatten/ReshapeReshape5dem_time_dist_conv2d/StatefulPartitionedCall:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ó
temp_flatten/PartitionedCallPartitionedCall6temp_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56034s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ś
temp_flatten/ReshapeReshape6temp_time_dist_conv2d/StatefulPartitionedCall:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ů
precip_flatten/PartitionedCallPartitionedCall8precip_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_precip_flatten_layer_call_and_return_conditional_losses_56091u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ź
precip_flatten/ReshapeReshape8precip_time_dist_conv2d/StatefulPartitionedCall:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
swe_flatten/PartitionedCallPartitionedCall5swe_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56148r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
swe_flatten/ReshapeReshape5swe_time_dist_conv2d/StatefulPartitionedCall:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙í
et_flatten/PartitionedCallPartitionedCall4et_time_dist_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56205q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         °
et_flatten/ReshapeReshape4et_time_dist_conv2d/StatefulPartitionedCall:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ţ
concatenate/PartitionedCallPartitionedCall$dem_flatten/PartitionedCall:output:0%temp_flatten/PartitionedCall:output:0'precip_flatten/PartitionedCall:output:0$swe_flatten/PartitionedCall:output:0#et_flatten/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_56489Ń
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0transformer_encoder_57528transformer_encoder_57530transformer_encoder_57532transformer_encoder_57534transformer_encoder_57536transformer_encoder_57538transformer_encoder_57540transformer_encoder_57542transformer_encoder_57544transformer_encoder_57546transformer_encoder_57548transformer_encoder_57550transformer_encoder_57552transformer_encoder_57554transformer_encoder_57556transformer_encoder_57558*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_57032ý
$global_max_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_56411ě
dropout/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_56814
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_57563dense_2_57565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_56718w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
NoOpNoOp-^dem_time_dist_conv2d/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^et_time_dist_conv2d/StatefulPartitionedCall0^precip_time_dist_conv2d/StatefulPartitionedCall-^swe_time_dist_conv2d/StatefulPartitionedCall.^temp_time_dist_conv2d/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs
úß
Ş$
 __inference__wrapped_model_55493

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs[
Amodel_et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:!'P
Bmodel_et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:^
Bmodel_swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:Q
Cmodel_swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:_
Emodel_precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:T
Fmodel_precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:]
Cmodel_temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:R
Dmodel_temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:\
@model_dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:úŢO
Amodel_dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:p
Zmodel_transformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:

b
Pmodel_transformer_encoder_multi_head_attention_query_add_readvariableop_resource:
n
Xmodel_transformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:

`
Nmodel_transformer_encoder_multi_head_attention_key_add_readvariableop_resource:
p
Zmodel_transformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:

b
Pmodel_transformer_encoder_multi_head_attention_value_add_readvariableop_resource:
{
emodel_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:

i
[model_transformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:
Y
Kmodel_transformer_encoder_layer_normalization_mul_3_readvariableop_resource:
W
Imodel_transformer_encoder_layer_normalization_add_readvariableop_resource:
^
Lmodel_transformer_encoder_sequential_dense_tensordot_readvariableop_resource:
 X
Jmodel_transformer_encoder_sequential_dense_biasadd_readvariableop_resource: `
Nmodel_transformer_encoder_sequential_dense_1_tensordot_readvariableop_resource: 
Z
Lmodel_transformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:
[
Mmodel_transformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:
Y
Kmodel_transformer_encoder_layer_normalization_1_add_readvariableop_resource:
>
,model_dense_2_matmul_readvariableop_resource:
;
-model_dense_2_biasadd_readvariableop_resource:
identity˘8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp˘7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp˘$model/dense_2/BiasAdd/ReadVariableOp˘#model/dense_2/MatMul/ReadVariableOp˘9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp˘8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp˘=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp˘<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp˘:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp˘9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp˘;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp˘:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp˘@model/transformer_encoder/layer_normalization/add/ReadVariableOp˘Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp˘Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp˘Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp˘Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp˘\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp˘Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp˘Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp˘Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp˘Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp˘Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp˘Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp˘Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp˘Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp˘Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp
'model/et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      Ł
!model/et_time_dist_conv2d/ReshapeReshape	et_inputs0model/et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2wÂ
8model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAmodel_et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0
)model/et_time_dist_conv2d/conv2d_1/Conv2DConv2D*model/et_time_dist_conv2d/Reshape:output:0@model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
¸
9model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBmodel_et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ć
*model/et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd2model/et_time_dist_conv2d/conv2d_1/Conv2D:output:0Amodel/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
'model/et_time_dist_conv2d/conv2d_1/ReluRelu3model/et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)model/et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            ×
#model/et_time_dist_conv2d/Reshape_1Reshape5model/et_time_dist_conv2d/conv2d_1/Relu:activations:02model/et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
)model/et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      §
#model/et_time_dist_conv2d/Reshape_2Reshape	et_inputs2model/et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w
(model/swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     ¨
"model/swe_time_dist_conv2d/ReshapeReshape
swe_inputs1model/swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓĆ
9model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOpBmodel_swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/swe_time_dist_conv2d/conv2d_4/Conv2DConv2D+model/swe_time_dist_conv2d/Reshape:output:0Amodel/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AMş
:model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpCmodel_swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+model/swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd3model/swe_time_dist_conv2d/conv2d_4/Conv2D:output:0Bmodel/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
(model/swe_time_dist_conv2d/conv2d_4/ReluRelu4model/swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*model/swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ú
$model/swe_time_dist_conv2d/Reshape_1Reshape6model/swe_time_dist_conv2d/conv2d_4/Relu:activations:03model/swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
*model/swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     Ź
$model/swe_time_dist_conv2d/Reshape_2Reshape
swe_inputs3model/swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
+model/precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ż
%model/precip_time_dist_conv2d/ReshapeReshapeprecip_inputs4model/precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ę
<model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOpEmodel_precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
-model/precip_time_dist_conv2d/conv2d_3/Conv2DConv2D.model/precip_time_dist_conv2d/Reshape:output:0Dmodel/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
Ŕ
=model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpFmodel_precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ň
.model/precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd6model/precip_time_dist_conv2d/conv2d_3/Conv2D:output:0Emodel/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
+model/precip_time_dist_conv2d/conv2d_3/ReluRelu7model/precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
-model/precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            ă
'model/precip_time_dist_conv2d/Reshape_1Reshape9model/precip_time_dist_conv2d/conv2d_3/Relu:activations:06model/precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
-model/precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ł
'model/precip_time_dist_conv2d/Reshape_2Reshapeprecip_inputs6model/precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)model/temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Š
#model/temp_time_dist_conv2d/ReshapeReshapetemp_inputs2model/temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOpCmodel_temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
+model/temp_time_dist_conv2d/conv2d_2/Conv2DConv2D,model/temp_time_dist_conv2d/Reshape:output:0Bmodel/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
ź
;model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpDmodel_temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ě
,model/temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd4model/temp_time_dist_conv2d/conv2d_2/Conv2D:output:0Cmodel/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
)model/temp_time_dist_conv2d/conv2d_2/ReluRelu5model/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
+model/temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ý
%model/temp_time_dist_conv2d/Reshape_1Reshape7model/temp_time_dist_conv2d/conv2d_2/Relu:activations:04model/temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
+model/temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ­
%model/temp_time_dist_conv2d/Reshape_2Reshapetemp_inputs4model/temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
(model/dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       ¨
"model/dem_time_dist_conv2d/ReshapeReshape
dem_inputs1model/dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷Â
7model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp@model_dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0
(model/dem_time_dist_conv2d/conv2d/Conv2DConv2D+model/dem_time_dist_conv2d/Reshape:output:0?model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻś
8model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOpAmodel_dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ă
)model/dem_time_dist_conv2d/conv2d/BiasAddBiasAdd1model/dem_time_dist_conv2d/conv2d/Conv2D:output:0@model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
&model/dem_time_dist_conv2d/conv2d/ReluRelu2model/dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*model/dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ř
$model/dem_time_dist_conv2d/Reshape_1Reshape4model/dem_time_dist_conv2d/conv2d/Relu:activations:03model/dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
*model/dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       Ź
$model/dem_time_dist_conv2d/Reshape_2Reshape
dem_inputs3model/dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷x
model/dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˇ
model/dem_flatten/ReshapeReshape-model/dem_time_dist_conv2d/Reshape_1:output:0(model/dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙p
model/dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   Ź
!model/dem_flatten/flatten/ReshapeReshape"model/dem_flatten/Reshape:output:0(model/dem_flatten/flatten/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
!model/dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ´
model/dem_flatten/Reshape_1Reshape*model/dem_flatten/flatten/Reshape:output:0*model/dem_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
!model/dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ť
model/dem_flatten/Reshape_2Reshape-model/dem_time_dist_conv2d/Reshape_1:output:0*model/dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙y
 model/temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ş
model/temp_flatten/ReshapeReshape.model/temp_time_dist_conv2d/Reshape_1:output:0)model/temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙s
"model/temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   ł
$model/temp_flatten/flatten_2/ReshapeReshape#model/temp_flatten/Reshape:output:0+model/temp_flatten/flatten_2/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
"model/temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   š
model/temp_flatten/Reshape_1Reshape-model/temp_flatten/flatten_2/Reshape:output:0+model/temp_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
"model/temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ž
model/temp_flatten/Reshape_2Reshape.model/temp_time_dist_conv2d/Reshape_1:output:0+model/temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙{
"model/precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ŕ
model/precip_flatten/ReshapeReshape0model/precip_time_dist_conv2d/Reshape_1:output:0+model/precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙u
$model/precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   š
&model/precip_flatten/flatten_3/ReshapeReshape%model/precip_flatten/Reshape:output:0-model/precip_flatten/flatten_3/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
$model/precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ż
model/precip_flatten/Reshape_1Reshape/model/precip_flatten/flatten_3/Reshape:output:0-model/precip_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
$model/precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ä
model/precip_flatten/Reshape_2Reshape0model/precip_time_dist_conv2d/Reshape_1:output:0-model/precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x
model/swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˇ
model/swe_flatten/ReshapeReshape-model/swe_time_dist_conv2d/Reshape_1:output:0(model/swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r
!model/swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   °
#model/swe_flatten/flatten_4/ReshapeReshape"model/swe_flatten/Reshape:output:0*model/swe_flatten/flatten_4/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
!model/swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ś
model/swe_flatten/Reshape_1Reshape,model/swe_flatten/flatten_4/Reshape:output:0*model/swe_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
!model/swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ť
model/swe_flatten/Reshape_2Reshape-model/swe_time_dist_conv2d/Reshape_1:output:0*model/swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
model/et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ´
model/et_flatten/ReshapeReshape,model/et_time_dist_conv2d/Reshape_1:output:0'model/et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q
 model/et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   ­
"model/et_flatten/flatten_1/ReshapeReshape!model/et_flatten/Reshape:output:0)model/et_flatten/flatten_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
 model/et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ł
model/et_flatten/Reshape_1Reshape+model/et_flatten/flatten_1/Reshape:output:0)model/et_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
 model/et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ¸
model/et_flatten/Reshape_2Reshape,model/et_time_dist_conv2d/Reshape_1:output:0)model/et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ě
model/concatenate/concatConcatV2$model/dem_flatten/Reshape_1:output:0%model/temp_flatten/Reshape_1:output:0'model/precip_flatten/Reshape_1:output:0$model/swe_flatten/Reshape_1:output:0#model/et_flatten/Reshape_1:output:0&model/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ş
Bmodel/transformer_encoder/multi_head_attention/query/einsum/EinsumEinsum!model/concatenate/concat:output:0Ymodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeŘ
Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpPmodel_transformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0
8model/transformer_encoder/multi_head_attention/query/addAddV2Kmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Omodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ś
@model/transformer_encoder/multi_head_attention/key/einsum/EinsumEinsum!model/concatenate/concat:output:0Wmodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeÔ
Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpNmodel_transformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0
6model/transformer_encoder/multi_head_attention/key/addAddV2Imodel/transformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Mmodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Ş
Bmodel/transformer_encoder/multi_head_attention/value/einsum/EinsumEinsum!model/concatenate/concat:output:0Ymodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeŘ
Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpPmodel_transformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0
8model/transformer_encoder/multi_head_attention/value/addAddV2Kmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Omodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
4model/transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>đ
2model/transformer_encoder/multi_head_attention/MulMul<model/transformer_encoder/multi_head_attention/query/add:z:0=model/transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

<model/transformer_encoder/multi_head_attention/einsum/EinsumEinsum:model/transformer_encoder/multi_head_attention/key/add:z:06model/transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbeĘ
>model/transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmaxEmodel/transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ď
?model/transformer_encoder/multi_head_attention/dropout/IdentityIdentityHmodel/transformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
>model/transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumHmodel/transformer_encoder/multi_head_attention/dropout/Identity:output:0<model/transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcd
\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpemodel_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0â
Mmodel/transformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumGmodel/transformer_encoder/multi_head_attention/einsum_1/Einsum:output:0dmodel/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeę
Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp[model_transformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0ś
Cmodel/transformer_encoder/multi_head_attention/attention_output/addAddV2Vmodel/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Zmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
model/transformer_encoder/addAddV2!model/concatenate/concat:output:0Gmodel/transformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

3model/transformer_encoder/layer_normalization/ShapeShape!model/transformer_encoder/add:z:0*
T0*
_output_shapes
:
Amodel/transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cmodel/transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cmodel/transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ˇ
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
value	B :Ý
1model/transformer_encoder/layer_normalization/mulMul<model/transformer_encoder/layer_normalization/mul/x:output:0Dmodel/transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 
Cmodel/transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Emodel/transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Emodel/transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
=model/transformer_encoder/layer_normalization/strided_slice_1StridedSlice<model/transformer_encoder/layer_normalization/Shape:output:0Lmodel/transformer_encoder/layer_normalization/strided_slice_1/stack:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Nmodel/transformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÚ
3model/transformer_encoder/layer_normalization/mul_1Mul5model/transformer_encoder/layer_normalization/mul:z:0Fmodel/transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 
Cmodel/transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Emodel/transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Emodel/transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
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
value	B :ă
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
value	B :ó
;model/transformer_encoder/layer_normalization/Reshape/shapePackFmodel/transformer_encoder/layer_normalization/Reshape/shape/0:output:07model/transformer_encoder/layer_normalization/mul_1:z:07model/transformer_encoder/layer_normalization/mul_2:z:0Fmodel/transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ă
5model/transformer_encoder/layer_normalization/ReshapeReshape!model/transformer_encoder/add:z:0Dmodel/transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
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
 *  ?ď
2model/transformer_encoder/layer_normalization/onesFillBmodel/transformer_encoder/layer_normalization/ones/packed:output:0Amodel/transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
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
 *    ň
3model/transformer_encoder/layer_normalization/zerosFillCmodel/transformer_encoder/layer_normalization/zeros/packed:output:0Bmodel/transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙v
3model/transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB x
5model/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ˇ
>model/transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV3>model/transformer_encoder/layer_normalization/Reshape:output:0;model/transformer_encoder/layer_normalization/ones:output:0<model/transformer_encoder/layer_normalization/zeros:output:0<model/transformer_encoder/layer_normalization/Const:output:0>model/transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:ú
7model/transformer_encoder/layer_normalization/Reshape_1ReshapeBmodel/transformer_encoder/layer_normalization/FusedBatchNormV3:y:0<model/transformer_encoder/layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpKmodel_transformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ţ
3model/transformer_encoder/layer_normalization/mul_3Mul@model/transformer_encoder/layer_normalization/Reshape_1:output:0Jmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
@model/transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpImodel_transformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0ó
1model/transformer_encoder/layer_normalization/addAddV27model/transformer_encoder/layer_normalization/mul_3:z:0Hmodel/transformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
dtype0
9model/transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9model/transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
:model/transformer_encoder/sequential/dense/Tensordot/ShapeShape5model/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:
Bmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=model/transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2Cmodel/transformer_encoder/sequential/dense/Tensordot/Shape:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/free:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?model/transformer_encoder/sequential/dense/Tensordot/GatherV2_1GatherV2Cmodel/transformer_encoder/sequential/dense/Tensordot/Shape:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/axes:output:0Mmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:model/transformer_encoder/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ď
9model/transformer_encoder/sequential/dense/Tensordot/ProdProdFmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0Cmodel/transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<model/transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ő
;model/transformer_encoder/sequential/dense/Tensordot/Prod_1ProdHmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0Emodel/transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@model/transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Č
;model/transformer_encoder/sequential/dense/Tensordot/concatConcatV2Bmodel/transformer_encoder/sequential/dense/Tensordot/free:output:0Bmodel/transformer_encoder/sequential/dense/Tensordot/axes:output:0Imodel/transformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:model/transformer_encoder/sequential/dense/Tensordot/stackPackBmodel/transformer_encoder/sequential/dense/Tensordot/Prod:output:0Dmodel/transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ţ
>model/transformer_encoder/sequential/dense/Tensordot/transpose	Transpose5model/transformer_encoder/layer_normalization/add:z:0Dmodel/transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

<model/transformer_encoder/sequential/dense/Tensordot/ReshapeReshapeBmodel/transformer_encoder/sequential/dense/Tensordot/transpose:y:0Cmodel/transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
;model/transformer_encoder/sequential/dense/Tensordot/MatMulMatMulEmodel/transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
<model/transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
Bmodel/transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=model/transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2Fmodel/transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0Emodel/transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Kmodel/transformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4model/transformer_encoder/sequential/dense/TensordotReshapeEmodel/transformer_encoder/sequential/dense/Tensordot/MatMul:product:0Fmodel/transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Č
Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ý
2model/transformer_encoder/sequential/dense/BiasAddBiasAdd=model/transformer_encoder/sequential/dense/Tensordot:output:0Imodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ş
/model/transformer_encoder/sequential/dense/ReluRelu;model/transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ô
Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpNmodel_transformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
dtype0
;model/transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
;model/transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Š
<model/transformer_encoder/sequential/dense_1/Tensordot/ShapeShape=model/transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:
Dmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ď
?model/transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2Emodel/transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/free:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Fmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
Amodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2Emodel/transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Omodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
<model/transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ő
;model/transformer_encoder/sequential/dense_1/Tensordot/ProdProdHmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Emodel/transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
>model/transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ű
=model/transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdJmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Gmodel/transformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Bmodel/transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Đ
=model/transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2Dmodel/transformer_encoder/sequential/dense_1/Tensordot/free:output:0Dmodel/transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Kmodel/transformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
<model/transformer_encoder/sequential/dense_1/Tensordot/stackPackDmodel/transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0Fmodel/transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
@model/transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose=model/transformer_encoder/sequential/dense/Relu:activations:0Fmodel/transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
>model/transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshapeDmodel/transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0Emodel/transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
=model/transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulGmodel/transformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>model/transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:

Dmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
?model/transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Hmodel/transformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Gmodel/transformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Mmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
6model/transformer_encoder/sequential/dense_1/TensordotReshapeGmodel/transformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Hmodel/transformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpLmodel_transformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
4model/transformer_encoder/sequential/dense_1/BiasAddBiasAdd?model/transformer_encoder/sequential/dense_1/Tensordot:output:0Kmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
model/transformer_encoder/add_1AddV25model/transformer_encoder/layer_normalization/add:z:0=model/transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

5model/transformer_encoder/layer_normalization_1/ShapeShape#model/transformer_encoder/add_1:z:0*
T0*
_output_shapes
:
Cmodel/transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Emodel/transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Emodel/transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
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
value	B :ă
3model/transformer_encoder/layer_normalization_1/mulMul>model/transformer_encoder/layer_normalization_1/mul/x:output:0Fmodel/transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 
Emodel/transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
?model/transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice>model/transformer_encoder/layer_normalization_1/Shape:output:0Nmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Pmodel/transformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŕ
5model/transformer_encoder/layer_normalization_1/mul_1Mul7model/transformer_encoder/layer_normalization_1/mul:z:0Hmodel/transformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 
Emodel/transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/transformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
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
value	B :é
5model/transformer_encoder/layer_normalization_1/mul_2Mul@model/transformer_encoder/layer_normalization_1/mul_2/x:output:0Hmodel/transformer_encoder/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 
?model/transformer_encoder/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :
?model/transformer_encoder/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ý
=model/transformer_encoder/layer_normalization_1/Reshape/shapePackHmodel/transformer_encoder/layer_normalization_1/Reshape/shape/0:output:09model/transformer_encoder/layer_normalization_1/mul_1:z:09model/transformer_encoder/layer_normalization_1/mul_2:z:0Hmodel/transformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:é
7model/transformer_encoder/layer_normalization_1/ReshapeReshape#model/transformer_encoder/add_1:z:0Fmodel/transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
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
 *  ?ő
4model/transformer_encoder/layer_normalization_1/onesFillDmodel/transformer_encoder/layer_normalization_1/ones/packed:output:0Cmodel/transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙­
<model/transformer_encoder/layer_normalization_1/zeros/packedPack9model/transformer_encoder/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:
;model/transformer_encoder/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ř
5model/transformer_encoder/layer_normalization_1/zerosFillEmodel/transformer_encoder/layer_normalization_1/zeros/packed:output:0Dmodel/transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙x
5model/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB z
7model/transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ă
@model/transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3@model/transformer_encoder/layer_normalization_1/Reshape:output:0=model/transformer_encoder/layer_normalization_1/ones:output:0>model/transformer_encoder/layer_normalization_1/zeros:output:0>model/transformer_encoder/layer_normalization_1/Const:output:0@model/transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:
9model/transformer_encoder/layer_normalization_1/Reshape_1ReshapeDmodel/transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:0>model/transformer_encoder/layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpMmodel_transformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0
5model/transformer_encoder/layer_normalization_1/mul_3MulBmodel/transformer_encoder/layer_normalization_1/Reshape_1:output:0Lmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpKmodel_transformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0ů
3model/transformer_encoder/layer_normalization_1/addAddV29model/transformer_encoder/layer_normalization_1/mul_3:z:0Jmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
0model/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ë
model/global_max_pooling1d/MaxMax7model/transformer_encoder/layer_normalization_1/add:z:09model/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
model/dropout/IdentityIdentity'model/global_max_pooling1d/Max:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
model/dense_2/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
NoOpNoOp9^model/dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp8^model/dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp:^model/et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp9^model/et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp>^model/precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp=^model/precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp;^model/swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:^model/swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp<^model/temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp;^model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpA^model/transformer_encoder/layer_normalization/add/ReadVariableOpC^model/transformer_encoder/layer_normalization/mul_3/ReadVariableOpC^model/transformer_encoder/layer_normalization_1/add/ReadVariableOpE^model/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpS^model/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp]^model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpF^model/transformer_encoder/multi_head_attention/key/add/ReadVariableOpP^model/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpH^model/transformer_encoder/multi_head_attention/query/add/ReadVariableOpR^model/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpH^model/transformer_encoder/multi_head_attention/value/add/ReadVariableOpR^model/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpB^model/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpD^model/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpD^model/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpF^model/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
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
:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:model/temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp2
@model/transformer_encoder/layer_normalization/add/ReadVariableOp@model/transformer_encoder/layer_normalization/add/ReadVariableOp2
Bmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOpBmodel/transformer_encoder/layer_normalization/mul_3/ReadVariableOp2
Bmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOpBmodel/transformer_encoder/layer_normalization_1/add/ReadVariableOp2
Dmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpDmodel/transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2¨
Rmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpRmodel/transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2ź
\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp\model/transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2
Emodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOpEmodel/transformer_encoder/multi_head_attention/key/add/ReadVariableOp2˘
Omodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpOmodel/transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
Gmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOpGmodel/transformer_encoder/multi_head_attention/query/add/ReadVariableOp2Ś
Qmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpQmodel/transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
Gmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOpGmodel/transformer_encoder/multi_head_attention/value/add/ReadVariableOp2Ś
Qmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpQmodel/transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2
Amodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpAmodel/transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2
Cmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOpCmodel/transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2
Cmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpCmodel/transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2
Emodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpEmodel/transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:a ]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs
ąż
í!
@__inference_model_layer_call_and_return_conditional_losses_58063
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4U
;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource:!'J
<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource:X
<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource:K
=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource:Y
?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource:N
@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource:W
=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource:L
>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource:V
:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource:úŢI
;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource:j
Ttransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource:

\
Jtransformer_encoder_multi_head_attention_query_add_readvariableop_resource:
h
Rtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource:

Z
Htransformer_encoder_multi_head_attention_key_add_readvariableop_resource:
j
Ttransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource:

\
Jtransformer_encoder_multi_head_attention_value_add_readvariableop_resource:
u
_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:

c
Utransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource:
S
Etransformer_encoder_layer_normalization_mul_3_readvariableop_resource:
Q
Ctransformer_encoder_layer_normalization_add_readvariableop_resource:
X
Ftransformer_encoder_sequential_dense_tensordot_readvariableop_resource:
 R
Dtransformer_encoder_sequential_dense_biasadd_readvariableop_resource: Z
Htransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource: 
T
Ftransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource:
U
Gtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource:
S
Etransformer_encoder_layer_normalization_1_add_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identity˘2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp˘1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp˘3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp˘2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp˘7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp˘6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp˘4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp˘3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp˘5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp˘4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp˘:transformer_encoder/layer_normalization/add/ReadVariableOp˘<transformer_encoder/layer_normalization/mul_3/ReadVariableOp˘<transformer_encoder/layer_normalization_1/add/ReadVariableOp˘>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp˘Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp˘Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp˘?transformer_encoder/multi_head_attention/key/add/ReadVariableOp˘Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp˘Atransformer_encoder/multi_head_attention/query/add/ReadVariableOp˘Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp˘Atransformer_encoder/multi_head_attention/value/add/ReadVariableOp˘Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp˘;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp˘=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp˘=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp˘?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpz
!et_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/ReshapeReshapeinputs_4*et_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2wś
2et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;et_time_dist_conv2d_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:!'*
dtype0ň
#et_time_dist_conv2d/conv2d_1/Conv2DConv2D$et_time_dist_conv2d/Reshape:output:0:et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
Ź
3et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<et_time_dist_conv2d_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ô
$et_time_dist_conv2d/conv2d_1/BiasAddBiasAdd,et_time_dist_conv2d/conv2d_1/Conv2D:output:0;et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!et_time_dist_conv2d/conv2d_1/ReluRelu-et_time_dist_conv2d/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
#et_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ĺ
et_time_dist_conv2d/Reshape_1Reshape/et_time_dist_conv2d/conv2d_1/Relu:activations:0,et_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙|
#et_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙2   w      
et_time_dist_conv2d/Reshape_2Reshapeinputs_4,et_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w{
"swe_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/ReshapeReshapeinputs_3+swe_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓş
3swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<swe_time_dist_conv2d_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ő
$swe_time_dist_conv2d/conv2d_4/Conv2DConv2D%swe_time_dist_conv2d/Reshape:output:0;swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AMŽ
4swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=swe_time_dist_conv2d_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
%swe_time_dist_conv2d/conv2d_4/BiasAddBiasAdd-swe_time_dist_conv2d/conv2d_4/Conv2D:output:0<swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"swe_time_dist_conv2d/conv2d_4/ReluRelu.swe_time_dist_conv2d/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
$swe_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Č
swe_time_dist_conv2d/Reshape_1Reshape0swe_time_dist_conv2d/conv2d_4/Relu:activations:0-swe_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙}
$swe_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙Ă   Ó     
swe_time_dist_conv2d/Reshape_2Reshapeinputs_3-swe_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ~
%precip_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
precip_time_dist_conv2d/ReshapeReshapeinputs_2.precip_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
6precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?precip_time_dist_conv2d_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ţ
'precip_time_dist_conv2d/conv2d_3/Conv2DConv2D(precip_time_dist_conv2d/Reshape:output:0>precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
´
7precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@precip_time_dist_conv2d_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ŕ
(precip_time_dist_conv2d/conv2d_3/BiasAddBiasAdd0precip_time_dist_conv2d/conv2d_3/Conv2D:output:0?precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%precip_time_dist_conv2d/conv2d_3/ReluRelu1precip_time_dist_conv2d/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
'precip_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ń
!precip_time_dist_conv2d/Reshape_1Reshape3precip_time_dist_conv2d/conv2d_3/Relu:activations:00precip_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'precip_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˘
!precip_time_dist_conv2d/Reshape_2Reshapeinputs_20precip_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙|
#temp_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/ReshapeReshapeinputs_1,temp_time_dist_conv2d/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
4temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=temp_time_dist_conv2d_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ř
%temp_time_dist_conv2d/conv2d_2/Conv2DConv2D&temp_time_dist_conv2d/Reshape:output:0<temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
°
5temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>temp_time_dist_conv2d_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ú
&temp_time_dist_conv2d/conv2d_2/BiasAddBiasAdd.temp_time_dist_conv2d/conv2d_2/Conv2D:output:0=temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
#temp_time_dist_conv2d/conv2d_2/ReluRelu/temp_time_dist_conv2d/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
%temp_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ë
temp_time_dist_conv2d/Reshape_1Reshape1temp_time_dist_conv2d/conv2d_2/Relu:activations:0.temp_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙~
%temp_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         
temp_time_dist_conv2d/Reshape_2Reshapeinputs_1.temp_time_dist_conv2d/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙{
"dem_time_dist_conv2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/ReshapeReshapeinputs_0+dem_time_dist_conv2d/Reshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷ś
1dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOpReadVariableOp:dem_time_dist_conv2d_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0ó
"dem_time_dist_conv2d/conv2d/Conv2DConv2D%dem_time_dist_conv2d/Reshape:output:09dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻŞ
2dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOpReadVariableOp;dem_time_dist_conv2d_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ń
#dem_time_dist_conv2d/conv2d/BiasAddBiasAdd+dem_time_dist_conv2d/conv2d/Conv2D:output:0:dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 dem_time_dist_conv2d/conv2d/ReluRelu,dem_time_dist_conv2d/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dem_time_dist_conv2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"˙˙˙˙            Ć
dem_time_dist_conv2d/Reshape_1Reshape.dem_time_dist_conv2d/conv2d/Relu:activations:0-dem_time_dist_conv2d/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙}
$dem_time_dist_conv2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙÷       
dem_time_dist_conv2d/Reshape_2Reshapeinputs_0-dem_time_dist_conv2d/Reshape_2/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷r
dem_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ľ
dem_flatten/ReshapeReshape'dem_time_dist_conv2d/Reshape_1:output:0"dem_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dem_flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
dem_flatten/flatten/ReshapeReshapedem_flatten/Reshape:output:0"dem_flatten/flatten/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
dem_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ˘
dem_flatten/Reshape_1Reshape$dem_flatten/flatten/Reshape:output:0$dem_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
dem_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Š
dem_flatten/Reshape_2Reshape'dem_time_dist_conv2d/Reshape_1:output:0$dem_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙s
temp_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ¨
temp_flatten/ReshapeReshape(temp_time_dist_conv2d/Reshape_1:output:0#temp_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙m
temp_flatten/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   Ą
temp_flatten/flatten_2/ReshapeReshapetemp_flatten/Reshape:output:0%temp_flatten/flatten_2/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
temp_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   §
temp_flatten/Reshape_1Reshape'temp_flatten/flatten_2/Reshape:output:0%temp_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
temp_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ź
temp_flatten/Reshape_2Reshape(temp_time_dist_conv2d/Reshape_1:output:0%temp_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙u
precip_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ž
precip_flatten/ReshapeReshape*precip_time_dist_conv2d/Reshape_1:output:0%precip_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙o
precip_flatten/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   §
 precip_flatten/flatten_3/ReshapeReshapeprecip_flatten/Reshape:output:0'precip_flatten/flatten_3/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
precip_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ­
precip_flatten/Reshape_1Reshape)precip_flatten/flatten_3/Reshape:output:0'precip_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
precip_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˛
precip_flatten/Reshape_2Reshape*precip_time_dist_conv2d/Reshape_1:output:0'precip_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r
swe_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ľ
swe_flatten/ReshapeReshape'swe_time_dist_conv2d/Reshape_1:output:0"swe_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙l
swe_flatten/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
swe_flatten/flatten_4/ReshapeReshapeswe_flatten/Reshape:output:0$swe_flatten/flatten_4/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
swe_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   ¤
swe_flatten/Reshape_1Reshape&swe_flatten/flatten_4/Reshape:output:0$swe_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
swe_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Š
swe_flatten/Reshape_2Reshape'swe_time_dist_conv2d/Reshape_1:output:0$swe_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q
et_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         ˘
et_flatten/ReshapeReshape&et_time_dist_conv2d/Reshape_1:output:0!et_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙k
et_flatten/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   
et_flatten/flatten_1/ReshapeReshapeet_flatten/Reshape:output:0#et_flatten/flatten_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
et_flatten/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"˙˙˙˙   
   Ą
et_flatten/Reshape_1Reshape%et_flatten/flatten_1/Reshape:output:0#et_flatten/Reshape_1/shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
et_flatten/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙         Ś
et_flatten/Reshape_2Reshape&et_time_dist_conv2d/Reshape_1:output:0#et_flatten/Reshape_2/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :˘
concatenate/concatConcatV2dem_flatten/Reshape_1:output:0temp_flatten/Reshape_1:output:0!precip_flatten/Reshape_1:output:0swe_flatten/Reshape_1:output:0et_flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
<transformer_encoder/multi_head_attention/query/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeĚ
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:
*
dtype0
2transformer_encoder/multi_head_attention/query/addAddV2Etransformer_encoder/multi_head_attention/query/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_encoder_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
:transformer_encoder/multi_head_attention/key/einsum/EinsumEinsumconcatenate/concat:output:0Qtransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeČ
?transformer_encoder/multi_head_attention/key/add/ReadVariableOpReadVariableOpHtransformer_encoder_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:
*
dtype0
0transformer_encoder/multi_head_attention/key/addAddV2Ctransformer_encoder/multi_head_attention/key/einsum/Einsum:output:0Gtransformer_encoder/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0
<transformer_encoder/multi_head_attention/value/einsum/EinsumEinsumconcatenate/concat:output:0Stransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabc,cde->abdeĚ
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpReadVariableOpJtransformer_encoder_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:
*
dtype0
2transformer_encoder/multi_head_attention/value/addAddV2Etransformer_encoder/multi_head_attention/value/einsum/Einsum:output:0Itransformer_encoder/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
.transformer_encoder/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *čĄ>Ţ
,transformer_encoder/multi_head_attention/MulMul6transformer_encoder/multi_head_attention/query/add:z:07transformer_encoder/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

6transformer_encoder/multi_head_attention/einsum/EinsumEinsum4transformer_encoder/multi_head_attention/key/add:z:00transformer_encoder/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationaecd,abcd->acbež
8transformer_encoder/multi_head_attention/softmax/SoftmaxSoftmax?transformer_encoder/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
9transformer_encoder/multi_head_attention/dropout/IdentityIdentityBtransformer_encoder/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumBtransformer_encoder/multi_head_attention/dropout/Identity:output:06transformer_encoder/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationacbe,aecd->abcdú
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp_transformer_encoder_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype0Đ
Gtransformer_encoder/multi_head_attention/attention_output/einsum/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
equationabcd,cde->abeŢ
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpUtransformer_encoder_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:
*
dtype0¤
=transformer_encoder/multi_head_attention/attention_output/addAddV2Ptransformer_encoder/multi_head_attention/attention_output/einsum/Einsum:output:0Ttransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
transformer_encoder/addAddV2concatenate/concat:output:0Atransformer_encoder/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
-transformer_encoder/layer_normalization/ShapeShapetransformer_encoder/add:z:0*
T0*
_output_shapes
:
;transformer_encoder/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=transformer_encoder/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=transformer_encoder/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
value	B :Ë
+transformer_encoder/layer_normalization/mulMul6transformer_encoder/layer_normalization/mul/x:output:0>transformer_encoder/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 
=transformer_encoder/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
7transformer_encoder/layer_normalization/strided_slice_1StridedSlice6transformer_encoder/layer_normalization/Shape:output:0Ftransformer_encoder/layer_normalization/strided_slice_1/stack:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_1:output:0Htransformer_encoder/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskČ
-transformer_encoder/layer_normalization/mul_1Mul/transformer_encoder/layer_normalization/mul:z:0@transformer_encoder/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 
=transformer_encoder/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
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
value	B :Ń
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
value	B :Ő
5transformer_encoder/layer_normalization/Reshape/shapePack@transformer_encoder/layer_normalization/Reshape/shape/0:output:01transformer_encoder/layer_normalization/mul_1:z:01transformer_encoder/layer_normalization/mul_2:z:0@transformer_encoder/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ń
/transformer_encoder/layer_normalization/ReshapeReshapetransformer_encoder/add:z:0>transformer_encoder/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
 *  ?Ý
,transformer_encoder/layer_normalization/onesFill<transformer_encoder/layer_normalization/ones/packed:output:0;transformer_encoder/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
 *    ŕ
-transformer_encoder/layer_normalization/zerosFill=transformer_encoder/layer_normalization/zeros/packed:output:0<transformer_encoder/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙p
-transformer_encoder/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB r
/transformer_encoder/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
8transformer_encoder/layer_normalization/FusedBatchNormV3FusedBatchNormV38transformer_encoder/layer_normalization/Reshape:output:05transformer_encoder/layer_normalization/ones:output:06transformer_encoder/layer_normalization/zeros:output:06transformer_encoder/layer_normalization/Const:output:08transformer_encoder/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:č
1transformer_encoder/layer_normalization/Reshape_1Reshape<transformer_encoder/layer_normalization/FusedBatchNormV3:y:06transformer_encoder/layer_normalization/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
<transformer_encoder/layer_normalization/mul_3/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ě
-transformer_encoder/layer_normalization/mul_3Mul:transformer_encoder/layer_normalization/Reshape_1:output:0Dtransformer_encoder/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
:transformer_encoder/layer_normalization/add/ReadVariableOpReadVariableOpCtransformer_encoder_layer_normalization_add_readvariableop_resource*
_output_shapes
:
*
dtype0á
+transformer_encoder/layer_normalization/addAddV21transformer_encoder/layer_normalization/mul_3:z:0Btransformer_encoder/layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
dtype0}
3transformer_encoder/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3transformer_encoder/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4transformer_encoder/sequential/dense/Tensordot/ShapeShape/transformer_encoder/layer_normalization/add:z:0*
T0*
_output_shapes
:~
<transformer_encoder/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ď
7transformer_encoder/sequential/dense/Tensordot/GatherV2GatherV2=transformer_encoder/sequential/dense/Tensordot/Shape:output:0<transformer_encoder/sequential/dense/Tensordot/free:output:0Etransformer_encoder/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>transformer_encoder/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
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
valueB: Ý
3transformer_encoder/sequential/dense/Tensordot/ProdProd@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0=transformer_encoder/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6transformer_encoder/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ă
5transformer_encoder/sequential/dense/Tensordot/Prod_1ProdBtransformer_encoder/sequential/dense/Tensordot/GatherV2_1:output:0?transformer_encoder/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_encoder/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : °
5transformer_encoder/sequential/dense/Tensordot/concatConcatV2<transformer_encoder/sequential/dense/Tensordot/free:output:0<transformer_encoder/sequential/dense/Tensordot/axes:output:0Ctransformer_encoder/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:č
4transformer_encoder/sequential/dense/Tensordot/stackPack<transformer_encoder/sequential/dense/Tensordot/Prod:output:0>transformer_encoder/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ě
8transformer_encoder/sequential/dense/Tensordot/transpose	Transpose/transformer_encoder/layer_normalization/add:z:0>transformer_encoder/sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
6transformer_encoder/sequential/dense/Tensordot/ReshapeReshape<transformer_encoder/sequential/dense/Tensordot/transpose:y:0=transformer_encoder/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ů
5transformer_encoder/sequential/dense/Tensordot/MatMulMatMul?transformer_encoder/sequential/dense/Tensordot/Reshape:output:0Etransformer_encoder/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
6transformer_encoder/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ~
<transformer_encoder/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
7transformer_encoder/sequential/dense/Tensordot/concat_1ConcatV2@transformer_encoder/sequential/dense/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense/Tensordot/Const_2:output:0Etransformer_encoder/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ň
.transformer_encoder/sequential/dense/TensordotReshape?transformer_encoder/sequential/dense/Tensordot/MatMul:product:0@transformer_encoder/sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ź
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpDtransformer_encoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ë
,transformer_encoder/sequential/dense/BiasAddBiasAdd7transformer_encoder/sequential/dense/Tensordot:output:0Ctransformer_encoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)transformer_encoder/sequential/dense/ReluRelu5transformer_encoder/sequential/dense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Č
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpHtransformer_encoder_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
dtype0
5transformer_encoder/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
5transformer_encoder/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
6transformer_encoder/sequential/dense_1/Tensordot/ShapeShape7transformer_encoder/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:
>transformer_encoder/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
9transformer_encoder/sequential/dense_1/Tensordot/GatherV2GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/free:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
@transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ű
;transformer_encoder/sequential/dense_1/Tensordot/GatherV2_1GatherV2?transformer_encoder/sequential/dense_1/Tensordot/Shape:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Itransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
6transformer_encoder/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ă
5transformer_encoder/sequential/dense_1/Tensordot/ProdProdBtransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_encoder/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
8transformer_encoder/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: é
7transformer_encoder/sequential/dense_1/Tensordot/Prod_1ProdDtransformer_encoder/sequential/dense_1/Tensordot/GatherV2_1:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_encoder/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
7transformer_encoder/sequential/dense_1/Tensordot/concatConcatV2>transformer_encoder/sequential/dense_1/Tensordot/free:output:0>transformer_encoder/sequential/dense_1/Tensordot/axes:output:0Etransformer_encoder/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:î
6transformer_encoder/sequential/dense_1/Tensordot/stackPack>transformer_encoder/sequential/dense_1/Tensordot/Prod:output:0@transformer_encoder/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ř
:transformer_encoder/sequential/dense_1/Tensordot/transpose	Transpose7transformer_encoder/sequential/dense/Relu:activations:0@transformer_encoder/sequential/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ˙
8transformer_encoder/sequential/dense_1/Tensordot/ReshapeReshape>transformer_encoder/sequential/dense_1/Tensordot/transpose:y:0?transformer_encoder/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
7transformer_encoder/sequential/dense_1/Tensordot/MatMulMatMulAtransformer_encoder/sequential/dense_1/Tensordot/Reshape:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8transformer_encoder/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:

>transformer_encoder/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ă
9transformer_encoder/sequential/dense_1/Tensordot/concat_1ConcatV2Btransformer_encoder/sequential/dense_1/Tensordot/GatherV2:output:0Atransformer_encoder/sequential/dense_1/Tensordot/Const_2:output:0Gtransformer_encoder/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ř
0transformer_encoder/sequential/dense_1/TensordotReshapeAtransformer_encoder/sequential/dense_1/Tensordot/MatMul:product:0Btransformer_encoder/sequential/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpFtransformer_encoder_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ń
.transformer_encoder/sequential/dense_1/BiasAddBiasAdd9transformer_encoder/sequential/dense_1/Tensordot:output:0Etransformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
transformer_encoder/add_1AddV2/transformer_encoder/layer_normalization/add:z:07transformer_encoder/sequential/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
/transformer_encoder/layer_normalization_1/ShapeShapetransformer_encoder/add_1:z:0*
T0*
_output_shapes
:
=transformer_encoder/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?transformer_encoder/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_encoder/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
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
value	B :Ń
-transformer_encoder/layer_normalization_1/mulMul8transformer_encoder/layer_normalization_1/mul/x:output:0@transformer_encoder/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 
?transformer_encoder/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
9transformer_encoder/layer_normalization_1/strided_slice_1StridedSlice8transformer_encoder/layer_normalization_1/Shape:output:0Htransformer_encoder/layer_normalization_1/strided_slice_1/stack:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_1:output:0Jtransformer_encoder/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÎ
/transformer_encoder/layer_normalization_1/mul_1Mul1transformer_encoder/layer_normalization_1/mul:z:0Btransformer_encoder/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 
?transformer_encoder/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atransformer_encoder/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ť
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
value	B :×
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
value	B :ß
7transformer_encoder/layer_normalization_1/Reshape/shapePackBtransformer_encoder/layer_normalization_1/Reshape/shape/0:output:03transformer_encoder/layer_normalization_1/mul_1:z:03transformer_encoder/layer_normalization_1/mul_2:z:0Btransformer_encoder/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:×
1transformer_encoder/layer_normalization_1/ReshapeReshapetransformer_encoder/add_1:z:0@transformer_encoder/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
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
 *  ?ă
.transformer_encoder/layer_normalization_1/onesFill>transformer_encoder/layer_normalization_1/ones/packed:output:0=transformer_encoder/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
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
 *    ć
/transformer_encoder/layer_normalization_1/zerosFill?transformer_encoder/layer_normalization_1/zeros/packed:output:0>transformer_encoder/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙r
/transformer_encoder/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB t
1transformer_encoder/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
:transformer_encoder/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3:transformer_encoder/layer_normalization_1/Reshape:output:07transformer_encoder/layer_normalization_1/ones:output:08transformer_encoder/layer_normalization_1/zeros:output:08transformer_encoder/layer_normalization_1/Const:output:0:transformer_encoder/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
data_formatNCHW*
epsilon%o:î
3transformer_encoder/layer_normalization_1/Reshape_1Reshape>transformer_encoder/layer_normalization_1/FusedBatchNormV3:y:08transformer_encoder/layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpGtransformer_encoder_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:
*
dtype0ň
/transformer_encoder/layer_normalization_1/mul_3Mul<transformer_encoder/layer_normalization_1/Reshape_1:output:0Ftransformer_encoder/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
<transformer_encoder/layer_normalization_1/add/ReadVariableOpReadVariableOpEtransformer_encoder_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:
*
dtype0ç
-transformer_encoder/layer_normalization_1/addAddV23transformer_encoder/layer_normalization_1/mul_3:z:0Dtransformer_encoder/layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :š
global_max_pooling1d/MaxMax1transformer_encoder/layer_normalization_1/add:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
dropout/IdentityIdentity!global_max_pooling1d/Max:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_2/MatMulMatMuldropout/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp3^dem_time_dist_conv2d/conv2d/BiasAdd/ReadVariableOp2^dem_time_dist_conv2d/conv2d/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp4^et_time_dist_conv2d/conv2d_1/BiasAdd/ReadVariableOp3^et_time_dist_conv2d/conv2d_1/Conv2D/ReadVariableOp8^precip_time_dist_conv2d/conv2d_3/BiasAdd/ReadVariableOp7^precip_time_dist_conv2d/conv2d_3/Conv2D/ReadVariableOp5^swe_time_dist_conv2d/conv2d_4/BiasAdd/ReadVariableOp4^swe_time_dist_conv2d/conv2d_4/Conv2D/ReadVariableOp6^temp_time_dist_conv2d/conv2d_2/BiasAdd/ReadVariableOp5^temp_time_dist_conv2d/conv2d_2/Conv2D/ReadVariableOp;^transformer_encoder/layer_normalization/add/ReadVariableOp=^transformer_encoder/layer_normalization/mul_3/ReadVariableOp=^transformer_encoder/layer_normalization_1/add/ReadVariableOp?^transformer_encoder/layer_normalization_1/mul_3/ReadVariableOpM^transformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpW^transformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp@^transformer_encoder/multi_head_attention/key/add/ReadVariableOpJ^transformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/query/add/ReadVariableOpL^transformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpB^transformer_encoder/multi_head_attention/value/add/ReadVariableOpL^transformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp<^transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp>^transformer_encoder/sequential/dense/Tensordot/ReadVariableOp>^transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp@^transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
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
<transformer_encoder/layer_normalization_1/add/ReadVariableOp<transformer_encoder/layer_normalization_1/add/ReadVariableOp2
>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp>transformer_encoder/layer_normalization_1/mul_3/ReadVariableOp2
Ltransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOpLtransformer_encoder/multi_head_attention/attention_output/add/ReadVariableOp2°
Vtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpVtransformer_encoder/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2
?transformer_encoder/multi_head_attention/key/add/ReadVariableOp?transformer_encoder/multi_head_attention/key/add/ReadVariableOp2
Itransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOpItransformer_encoder/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
Atransformer_encoder/multi_head_attention/query/add/ReadVariableOpAtransformer_encoder/multi_head_attention/query/add/ReadVariableOp2
Ktransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
Atransformer_encoder/multi_head_attention/value/add/ReadVariableOpAtransformer_encoder/multi_head_attention/value/add/ReadVariableOp2
Ktransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOpKtransformer_encoder/multi_head_attention/value/einsum/Einsum/ReadVariableOp2z
;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp;transformer_encoder/sequential/dense/BiasAdd/ReadVariableOp2~
=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp=transformer_encoder/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp=transformer_encoder/sequential/dense_1/BiasAdd/ReadVariableOp2
?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp?transformer_encoder/sequential/dense_1/Tensordot/ReadVariableOp:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
"
_user_specified_name
inputs_0:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
"
_user_specified_name
inputs_3:]Y
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
"
_user_specified_name
inputs_4

Ł
E__inference_sequential_layer_call_and_return_conditional_losses_56387
dense_input
dense_56376:
 
dense_56378: 
dense_1_56381: 

dense_1_56383:

identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallę
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_56376dense_56378*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_56246
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56381dense_1_56383*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_56282{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

%
_user_specified_namedense_input

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_59455

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
G
+__inference_swe_flatten_layer_call_fn_58837

inputs
identityž
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56148m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ő
Ž
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58629

inputsC
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity˘conv2d_4/BiasAdd/ReadVariableOp˘conv2d_4/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙Ă   Ó     n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ś
conv2d_4/Conv2DConv2DReshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AM
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_4/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
ť
Ź
7__inference_precip_time_dist_conv2d_layer_call_fn_58506

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55703
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ę
ć
#__inference_signature_wrapper_57638

dem_inputs
	et_inputs
precip_inputs

swe_inputs
temp_inputs!
unknown:!'
	unknown_0:%
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:úŢ
	unknown_8:
	unknown_9:



unknown_10:
 

unknown_11:



unknown_12:
 

unknown_13:



unknown_14:
 

unknown_15:



unknown_16:


unknown_17:


unknown_18:


unknown_19:
 

unknown_20: 

unknown_21: 


unknown_22:


unknown_23:


unknown_24:


unknown_25:


unknown_26:
identity˘StatefulPartitionedCallÓ
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
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_55493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙2w:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs
ˇ
Ň
*__inference_sequential_layer_call_fn_56300
dense_input
unknown:
 
	unknown_0: 
	unknown_1: 

	unknown_2:

identity˘StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_56289s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

%
_user_specified_namedense_input
ł
¨
3__inference_et_time_dist_conv2d_layer_call_fn_58638

inputs!
unknown:!'
	unknown_0:
identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55875
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
Ň
Ü
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_55703

inputs(
conv2d_3_55691:
conv2d_3_55693:
identity˘ conv2d_3/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˙
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_3_55691conv2d_3_55693*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55690\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ł
E__inference_sequential_layer_call_and_return_conditional_losses_56401
dense_input
dense_56390:
 
dense_56392: 
dense_1_56395: 

dense_1_56397:

identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCallę
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_56390dense_56392*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_56246
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_56395dense_1_56397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_56282{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

%
_user_specified_namedense_input
Ő
`
B__inference_dropout_layer_call_and_return_conditional_losses_56706

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙
:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Î
Ř
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_55916

inputs(
conv2d_1_55904:!'
conv2d_1_55906:
identity˘ conv2d_1/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙2   w      l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2w˙
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_55904conv2d_1_55906*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55862\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
 
_user_specified_nameinputs
ů
e
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58810

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_3/ReshapeReshapeReshape:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_3/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ě
­
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58497

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity˘conv2d_2/BiasAdd/ReadVariableOp˘conv2d_2/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ś
conv2d_2/Conv2DConv2DReshape:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ú
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_55658

inputs(
conv2d_2_55646:
conv2d_2_55648:
identity˘ conv2d_2/StatefulPartitionedCall;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˙
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_2_55646conv2d_2_55648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55604\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
Ż
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58539

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:
identity˘conv2d_3/BiasAdd/ReadVariableOp˘conv2d_3/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ś
conv2d_3/Conv2DConv2DReshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Č=
Ě
E__inference_sequential_layer_call_and_return_conditional_losses_59653

inputs9
'dense_tensordot_readvariableop_resource:
 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 
5
'dense_1_biasadd_readvariableop_resource:

identity˘dense/BiasAdd/ReadVariableOp˘dense/Tensordot/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : Ó
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
value	B : ×
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
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: 
*
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
value	B : Ű
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
value	B : ß
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
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ź
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ˘
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˘
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ç
H
,__inference_temp_flatten_layer_call_fn_58749

inputs
identityż
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56034m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ă
F
*__inference_et_flatten_layer_call_fn_58881

inputs
identity˝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_et_flatten_layer_call_and_return_conditional_losses_56205m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
c
G__inference_temp_flatten_layer_call_and_return_conditional_losses_56034

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
flatten_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_56000\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape"flatten_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
G
+__inference_swe_flatten_layer_call_fn_58832

inputs
identityž
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_swe_flatten_layer_call_and_return_conditional_losses_56121m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

(__inference_conv2d_2_layer_call_fn_59444

inputs!
unknown:
	unknown_0:
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55604w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
Ť
4__inference_swe_time_dist_conv2d_layer_call_fn_58572

inputs#
unknown:
	unknown_0:
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55789
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
ě

F__inference_concatenate_layer_call_and_return_conditional_losses_58934
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
value	B :
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesu
s:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:U Q
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_4
Ő	

+__inference_concatenate_layer_call_fn_58924
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityă
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_56489d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesu
s:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙
:U Q
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

"
_user_specified_name
inputs_4
ö
b
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58871

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   z
flatten_4/ReshapeReshapeReshape:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten_4/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

č
%__inference_model_layer_call_fn_56784

dem_inputs
temp_inputs
precip_inputs

swe_inputs
	et_inputs!
unknown:!'
	unknown_0:%
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:%
	unknown_7:úŢ
	unknown_8:
	unknown_9:



unknown_10:
 

unknown_11:



unknown_12:
 

unknown_13:



unknown_14:
 

unknown_15:



unknown_16:


unknown_17:


unknown_18:


unknown_19:
 

unknown_20: 

unknown_21: 


unknown_22:


unknown_23:


unknown_24:


unknown_25:


unknown_26:
identity˘StatefulPartitionedCalló
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
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_56725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ě
_input_shapesÚ
×:˙˙˙˙˙˙˙˙˙÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙ĂÓ:˙˙˙˙˙˙˙˙˙2w: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙÷
$
_user_specified_name
dem_inputs:`\
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nametemp_inputs:b^
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprecip_inputs:a]
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ
$
_user_specified_name
swe_inputs:^Z
3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2w
#
_user_specified_name	et_inputs
î
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58739

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   v
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

÷
@__inference_dense_layer_call_and_return_conditional_losses_56246

inputs3
!tensordot_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
 *
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
value	B : ť
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
value	B : ż
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
value	B : 
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
:˙˙˙˙˙˙˙˙˙

Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ŕ
b
F__inference_dem_flatten_layer_call_and_return_conditional_losses_55950

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
valueB:Ń
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
valueB"˙˙˙˙         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_55943\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :

Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:d `
<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
Í
*__inference_sequential_layer_call_fn_59583

inputs
unknown:
 
	unknown_0: 
	unknown_1: 

	unknown_2:

identity˘StatefulPartitionedCallř
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_56289s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ä
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_56000

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
­
E
)__inference_flatten_4_layer_call_fn_59553

inputs
identityŻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_56114`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ţ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_59495

inputs:
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
AMr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙ĂÓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
ť
Ť
4__inference_swe_time_dist_conv2d_layer_call_fn_58581

inputs#
unknown:
	unknown_0:
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_55830
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
 
_user_specified_nameinputs
Ť
Ś
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58431

inputsA
%conv2d_conv2d_readvariableop_resource:úŢ4
&conv2d_biasadd_readvariableop_resource:
identity˘conv2d/BiasAdd/ReadVariableOp˘conv2d/Conv2D/ReadVariableOp;
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
valueB:Ń
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
valueB"˙˙˙˙÷       n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙÷
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:úŢ*
dtype0´
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides

ýŻ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:f b
>
_output_shapes,
*:(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
 
_user_specified_nameinputs

ü
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55690

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_59537

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultî
O

dem_inputsA
serving_default_dem_inputs:0˙˙˙˙˙˙˙˙˙÷
K
	et_inputs>
serving_default_et_inputs:0˙˙˙˙˙˙˙˙˙2w
S
precip_inputsB
serving_default_precip_inputs:0˙˙˙˙˙˙˙˙˙
O

swe_inputsA
serving_default_swe_inputs:0˙˙˙˙˙˙˙˙˙ĂÓ
O
temp_inputs@
serving_default_temp_inputs:0˙˙˙˙˙˙˙˙˙;
dense_20
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Ö
§
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
°
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer"
_tf_keras_layer
°
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer"
_tf_keras_layer
°
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
	2layer"
_tf_keras_layer
°
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
	9layer"
_tf_keras_layer
°
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
	@layer"
_tf_keras_layer
°
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Glayer"
_tf_keras_layer
°
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
	Nlayer"
_tf_keras_layer
°
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
	Ulayer"
_tf_keras_layer
°
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
	\layer"
_tf_keras_layer
°
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
	clayer"
_tf_keras_layer
Ľ
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
ć
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
Ľ
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
˝
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Ă
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
Ą24
˘25
26
27"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
Ą24
˘25
26
27"
trackable_list_wrapper
 "
trackable_list_wrapper
Ď
Łnon_trainable_variables
¤layers
Ľmetrics
 Ślayer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ń
¨trace_0
Štrace_1
Ştrace_2
Ťtrace_32Ţ
%__inference_model_layer_call_fn_56784
%__inference_model_layer_call_fn_57703
%__inference_model_layer_call_fn_57768
%__inference_model_layer_call_fn_57371ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z¨trace_0zŠtrace_1zŞtrace_2zŤtrace_3
˝
Źtrace_0
­trace_1
Žtrace_2
Żtrace_32Ę
@__inference_model_layer_call_and_return_conditional_losses_58063
@__inference_model_layer_call_and_return_conditional_losses_58365
@__inference_model_layer_call_and_return_conditional_losses_57470
@__inference_model_layer_call_and_return_conditional_losses_57569ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŹtrace_0z­trace_1zŽtrace_2zŻtrace_3
Bţ
 __inference__wrapped_model_55493
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ł
°
_variables
ą_iterations
˛_learning_rate
ł_index_dict
´
_momentums
ľ_velocities
ś_update_step_xla"
experimentalOptimizer
-
ˇserving_default"
signature_map
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
¸non_trainable_variables
šlayers
şmetrics
 ťlayer_regularization_losses
źlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
é
˝trace_0
žtrace_12Ž
4__inference_dem_time_dist_conv2d_layer_call_fn_58374
4__inference_dem_time_dist_conv2d_layer_call_fn_58383ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z˝trace_0zžtrace_1

żtrace_0
Ŕtrace_12ä
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58407
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58431ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zżtrace_0zŔtrace_1
ć
Á	variables
Âtrainable_variables
Ăregularization_losses
Ä	keras_api
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
kernel
	bias
!Ç_jit_compiled_convolution_op"
_tf_keras_layer
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Čnon_trainable_variables
Élayers
Ęmetrics
 Ëlayer_regularization_losses
Ělayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ë
Ítrace_0
Îtrace_12°
5__inference_temp_time_dist_conv2d_layer_call_fn_58440
5__inference_temp_time_dist_conv2d_layer_call_fn_58449ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÍtrace_0zÎtrace_1
Ą
Ďtrace_0
Đtrace_12ć
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58473
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58497ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zĎtrace_0zĐtrace_1
ć
Ń	variables
Ňtrainable_variables
Óregularization_losses
Ô	keras_api
Ő__call__
+Ö&call_and_return_all_conditional_losses
kernel
	bias
!×_jit_compiled_convolution_op"
_tf_keras_layer
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Řnon_trainable_variables
Ůlayers
Úmetrics
 Űlayer_regularization_losses
Ülayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ď
Ýtrace_0
Ţtrace_12´
7__inference_precip_time_dist_conv2d_layer_call_fn_58506
7__inference_precip_time_dist_conv2d_layer_call_fn_58515ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÝtrace_0zŢtrace_1
Ľ
ßtrace_0
ŕtrace_12ę
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58539
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58563ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zßtrace_0zŕtrace_1
ć
á	variables
âtrainable_variables
ăregularization_losses
ä	keras_api
ĺ__call__
+ć&call_and_return_all_conditional_losses
kernel
	bias
!ç_jit_compiled_convolution_op"
_tf_keras_layer
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
čnon_trainable_variables
élayers
ęmetrics
 ëlayer_regularization_losses
ělayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
é
ítrace_0
îtrace_12Ž
4__inference_swe_time_dist_conv2d_layer_call_fn_58572
4__inference_swe_time_dist_conv2d_layer_call_fn_58581ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zítrace_0zîtrace_1

ďtrace_0
đtrace_12ä
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58605
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58629ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zďtrace_0zđtrace_1
ć
ń	variables
ňtrainable_variables
óregularization_losses
ô	keras_api
ő__call__
+ö&call_and_return_all_conditional_losses
kernel
	bias
!÷_jit_compiled_convolution_op"
_tf_keras_layer
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
řnon_trainable_variables
ůlayers
úmetrics
 űlayer_regularization_losses
ülayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ç
ýtrace_0
ţtrace_12Ź
3__inference_et_time_dist_conv2d_layer_call_fn_58638
3__inference_et_time_dist_conv2d_layer_call_fn_58647ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zýtrace_0zţtrace_1

˙trace_0
trace_12â
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58671
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58695ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z˙trace_0ztrace_1
ć
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
×
trace_0
trace_12
+__inference_dem_flatten_layer_call_fn_58700
+__inference_dem_flatten_layer_call_fn_58705ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1

trace_0
trace_12Ň
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58722
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58739ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
Ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Ů
trace_0
trace_12
,__inference_temp_flatten_layer_call_fn_58744
,__inference_temp_flatten_layer_call_fn_58749ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1

trace_0
trace_12Ô
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58766
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58783ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
Ť
 	variables
Ątrainable_variables
˘regularization_losses
Ł	keras_api
¤__call__
+Ľ&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Śnon_trainable_variables
§layers
¨metrics
 Šlayer_regularization_losses
Şlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
Ý
Ťtrace_0
Źtrace_12˘
.__inference_precip_flatten_layer_call_fn_58788
.__inference_precip_flatten_layer_call_fn_58793ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŤtrace_0zŹtrace_1

­trace_0
Žtrace_12Ř
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58810
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58827ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z­trace_0zŽtrace_1
Ť
Ż	variables
°trainable_variables
ąregularization_losses
˛	keras_api
ł__call__
+´&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
ľnon_trainable_variables
ślayers
ˇmetrics
 ¸layer_regularization_losses
šlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
×
ştrace_0
ťtrace_12
+__inference_swe_flatten_layer_call_fn_58832
+__inference_swe_flatten_layer_call_fn_58837ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zştrace_0zťtrace_1

źtrace_0
˝trace_12Ň
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58854
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58871ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zźtrace_0z˝trace_1
Ť
ž	variables
żtrainable_variables
Ŕregularization_losses
Á	keras_api
Â__call__
+Ă&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Änon_trainable_variables
Ĺlayers
Ćmetrics
 Çlayer_regularization_losses
Člayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Ő
Étrace_0
Ętrace_12
*__inference_et_flatten_layer_call_fn_58876
*__inference_et_flatten_layer_call_fn_58881ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÉtrace_0zĘtrace_1

Ëtrace_0
Ětrace_12Đ
E__inference_et_flatten_layer_call_and_return_conditional_losses_58898
E__inference_et_flatten_layer_call_and_return_conditional_losses_58915ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zËtrace_0zĚtrace_1
Ť
Í	variables
Îtrainable_variables
Ďregularization_losses
Đ	keras_api
Ń__call__
+Ň&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Ónon_trainable_variables
Ôlayers
Őmetrics
 Ölayer_regularization_losses
×layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ń
Řtrace_02Ň
+__inference_concatenate_layer_call_fn_58924˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŘtrace_0

Ůtrace_02í
F__inference_concatenate_layer_call_and_return_conditional_losses_58934˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŮtrace_0
Ś
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
Ą14
˘15"
trackable_list_wrapper
Ś
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
Ą14
˘15"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
Únon_trainable_variables
Űlayers
Ümetrics
 Ýlayer_regularization_losses
Ţlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ô
ßtrace_0
ŕtrace_12š
3__inference_transformer_encoder_layer_call_fn_58971
3__inference_transformer_encoder_layer_call_fn_59008Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 zßtrace_0zŕtrace_1
Ş
átrace_0
âtrace_12ď
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59183
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59358Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 zátrace_0zâtrace_1

ă	variables
ätrainable_variables
ĺregularization_losses
ć	keras_api
ç__call__
+č&call_and_return_all_conditional_losses
é_query_dense
ę
_key_dense
ë_value_dense
ě_softmax
í_dropout_layer
î_output_dense"
_tf_keras_layer

ďlayer_with_weights-0
ďlayer-0
đlayer_with_weights-1
đlayer-1
ń	variables
ňtrainable_variables
óregularization_losses
ô	keras_api
ő__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_sequential
Í
÷	variables
řtrainable_variables
ůregularization_losses
ú	keras_api
ű__call__
+ü&call_and_return_all_conditional_losses
	ýaxis

gamma
	 beta"
_tf_keras_layer
Í
ţ	variables
˙trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

Ągamma
	˘beta"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
ú
trace_02Ű
4__inference_global_max_pooling1d_layer_call_fn_59363˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ö
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_59369˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ă
trace_0
trace_12
'__inference_dropout_layer_call_fn_59374
'__inference_dropout_layer_call_fn_59379ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
ů
trace_0
trace_12ž
B__inference_dropout_layer_call_and_return_conditional_losses_59384
B__inference_dropout_layer_call_and_return_conditional_losses_59396ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_2_layer_call_fn_59405˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02é
B__inference_dense_2_layer_call_and_return_conditional_losses_59415˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
 :
2dense_2/kernel
:2dense_2/bias
7:5úŢ2dem_time_dist_conv2d/kernel
':%2dem_time_dist_conv2d/bias
6:42temp_time_dist_conv2d/kernel
(:&2temp_time_dist_conv2d/bias
8:62precip_time_dist_conv2d/kernel
*:(2precip_time_dist_conv2d/bias
7:52swe_time_dist_conv2d/kernel
':%2swe_time_dist_conv2d/bias
4:2!'2et_time_dist_conv2d/kernel
&:$2et_time_dist_conv2d/bias
K:I

25transformer_encoder/multi_head_attention/query/kernel
E:C
23transformer_encoder/multi_head_attention/query/bias
I:G

23transformer_encoder/multi_head_attention/key/kernel
C:A
21transformer_encoder/multi_head_attention/key/bias
K:I

25transformer_encoder/multi_head_attention/value/kernel
E:C
23transformer_encoder/multi_head_attention/value/bias
V:T

2@transformer_encoder/multi_head_attention/attention_output/kernel
L:J
2>transformer_encoder/multi_head_attention/attention_output/bias
:
 2dense/kernel
: 2
dense/bias
 : 
2dense_1/kernel
:
2dense_1/bias
;:9
2-transformer_encoder/layer_normalization/gamma
::8
2,transformer_encoder/layer_normalization/beta
=:;
2/transformer_encoder/layer_normalization_1/gamma
<::
2.transformer_encoder/layer_normalization_1/beta
 "
trackable_list_wrapper
ś
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
­BŞ
%__inference_model_layer_call_fn_56784
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
%__inference_model_layer_call_fn_57703inputs_0inputs_1inputs_2inputs_3inputs_4"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
%__inference_model_layer_call_fn_57768inputs_0inputs_1inputs_2inputs_3inputs_4"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
­BŞ
%__inference_model_layer_call_fn_57371
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ťB¸
@__inference_model_layer_call_and_return_conditional_losses_58063inputs_0inputs_1inputs_2inputs_3inputs_4"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ťB¸
@__inference_model_layer_call_and_return_conditional_losses_58365inputs_0inputs_1inputs_2inputs_3inputs_4"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ČBĹ
@__inference_model_layer_call_and_return_conditional_losses_57470
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ČBĹ
@__inference_model_layer_call_and_return_conditional_losses_57569
dem_inputstemp_inputsprecip_inputs
swe_inputs	et_inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 

ą0
1
2
3
 4
Ą5
˘6
Ł7
¤8
Ľ9
Ś10
§11
¨12
Š13
Ş14
Ť15
Ź16
­17
Ž18
Ż19
°20
ą21
˛22
ł23
´24
ľ25
ś26
ˇ27
¸28
š29
ş30
ť31
ź32
˝33
ž34
ż35
Ŕ36
Á37
Â38
Ă39
Ä40
Ĺ41
Ć42
Ç43
Č44
É45
Ę46
Ë47
Ě48
Í49
Î50
Ď51
Đ52
Ń53
Ň54
Ó55
Ô56"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

0
1
Ą2
Ł3
Ľ4
§5
Š6
Ť7
­8
Ż9
ą10
ł11
ľ12
ˇ13
š14
ť15
˝16
ż17
Á18
Ă19
Ĺ20
Ç21
É22
Ë23
Í24
Ď25
Ń26
Ó27"
trackable_list_wrapper

0
 1
˘2
¤3
Ś4
¨5
Ş6
Ź7
Ž8
°9
˛10
´11
ś12
¸13
ş14
ź15
ž16
Ŕ17
Â18
Ä19
Ć20
Č21
Ę22
Ě23
Î24
Đ25
Ň26
Ô27"
trackable_list_wrapper
ż2źš
Ž˛Ş
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 0
ţBű
#__inference_signature_wrapper_57638
dem_inputs	et_inputsprecip_inputs
swe_inputstemp_inputs"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
4__inference_dem_time_dist_conv2d_layer_call_fn_58374inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
4__inference_dem_time_dist_conv2d_layer_call_fn_58383inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58407inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58431inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Őnon_trainable_variables
Ölayers
×metrics
 Řlayer_regularization_losses
Ůlayer_metrics
Á	variables
Âtrainable_variables
Ăregularization_losses
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
ě
Útrace_02Í
&__inference_conv2d_layer_call_fn_59424˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zÚtrace_0

Űtrace_02č
A__inference_conv2d_layer_call_and_return_conditional_losses_59435˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŰtrace_0
´2ąŽ
Ł˛
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
5__inference_temp_time_dist_conv2d_layer_call_fn_58440inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
5__inference_temp_time_dist_conv2d_layer_call_fn_58449inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58473inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĄB
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58497inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ünon_trainable_variables
Ýlayers
Ţmetrics
 ßlayer_regularization_losses
ŕlayer_metrics
Ń	variables
Ňtrainable_variables
Óregularization_losses
Ő__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
î
átrace_02Ď
(__inference_conv2d_2_layer_call_fn_59444˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zátrace_0

âtrace_02ę
C__inference_conv2d_2_layer_call_and_return_conditional_losses_59455˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zâtrace_0
´2ąŽ
Ł˛
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
7__inference_precip_time_dist_conv2d_layer_call_fn_58506inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
7__inference_precip_time_dist_conv2d_layer_call_fn_58515inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŁB 
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58539inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŁB 
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58563inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ănon_trainable_variables
älayers
ĺmetrics
 ćlayer_regularization_losses
çlayer_metrics
á	variables
âtrainable_variables
ăregularization_losses
ĺ__call__
+ć&call_and_return_all_conditional_losses
'ć"call_and_return_conditional_losses"
_generic_user_object
î
čtrace_02Ď
(__inference_conv2d_3_layer_call_fn_59464˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zčtrace_0

étrace_02ę
C__inference_conv2d_3_layer_call_and_return_conditional_losses_59475˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zétrace_0
´2ąŽ
Ł˛
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
4__inference_swe_time_dist_conv2d_layer_call_fn_58572inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
4__inference_swe_time_dist_conv2d_layer_call_fn_58581inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58605inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58629inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ęnon_trainable_variables
ëlayers
ěmetrics
 ílayer_regularization_losses
îlayer_metrics
ń	variables
ňtrainable_variables
óregularization_losses
ő__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
î
ďtrace_02Ď
(__inference_conv2d_4_layer_call_fn_59484˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zďtrace_0

đtrace_02ę
C__inference_conv2d_4_layer_call_and_return_conditional_losses_59495˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zđtrace_0
´2ąŽ
Ł˛
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
3__inference_et_time_dist_conv2d_layer_call_fn_58638inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
3__inference_et_time_dist_conv2d_layer_call_fn_58647inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58671inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58695inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ńnon_trainable_variables
ňlayers
ómetrics
 ôlayer_regularization_losses
őlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
î
ötrace_02Ď
(__inference_conv2d_1_layer_call_fn_59504˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zötrace_0

÷trace_02ę
C__inference_conv2d_1_layer_call_and_return_conditional_losses_59515˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z÷trace_0
´2ąŽ
Ł˛
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
üBů
+__inference_dem_flatten_layer_call_fn_58700inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
üBů
+__inference_dem_flatten_layer_call_fn_58705inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58722inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58739inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
řnon_trainable_variables
ůlayers
úmetrics
 űlayer_regularization_losses
ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
ýtrace_02Î
'__inference_flatten_layer_call_fn_59520˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zýtrace_0

ţtrace_02é
B__inference_flatten_layer_call_and_return_conditional_losses_59526˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zţtrace_0
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
ýBú
,__inference_temp_flatten_layer_call_fn_58744inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
,__inference_temp_flatten_layer_call_fn_58749inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58766inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58783inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
˙non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
Ątrainable_variables
˘regularization_losses
¤__call__
+Ľ&call_and_return_all_conditional_losses
'Ľ"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_flatten_2_layer_call_fn_59531˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_flatten_2_layer_call_and_return_conditional_losses_59537˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
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
˙Bü
.__inference_precip_flatten_layer_call_fn_58788inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
.__inference_precip_flatten_layer_call_fn_58793inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58810inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58827inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ż	variables
°trainable_variables
ąregularization_losses
ł__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_flatten_3_layer_call_fn_59542˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_flatten_3_layer_call_and_return_conditional_losses_59548˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
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
üBů
+__inference_swe_flatten_layer_call_fn_58832inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
üBů
+__inference_swe_flatten_layer_call_fn_58837inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58854inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58871inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ž	variables
żtrainable_variables
Ŕregularization_losses
Â__call__
+Ă&call_and_return_all_conditional_losses
'Ă"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_flatten_4_layer_call_fn_59553˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_flatten_4_layer_call_and_return_conditional_losses_59559˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
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
űBř
*__inference_et_flatten_layer_call_fn_58876inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
*__inference_et_flatten_layer_call_fn_58881inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_et_flatten_layer_call_and_return_conditional_losses_58898inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_et_flatten_layer_call_and_return_conditional_losses_58915inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Í	variables
Îtrainable_variables
Ďregularization_losses
Ń__call__
+Ň&call_and_return_all_conditional_losses
'Ň"call_and_return_conditional_losses"
_generic_user_object
ď
trace_02Đ
)__inference_flatten_1_layer_call_fn_59564˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ë
D__inference_flatten_1_layer_call_and_return_conditional_losses_59570˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
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
B
+__inference_concatenate_layer_call_fn_58924inputs_0inputs_1inputs_2inputs_3inputs_4"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¤BĄ
F__inference_concatenate_layer_call_and_return_conditional_losses_58934inputs_0inputs_1inputs_2inputs_3inputs_4"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
B
3__inference_transformer_encoder_layer_call_fn_58971inputs"Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
B
3__inference_transformer_encoder_layer_call_fn_59008inputs"Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ŹBŠ
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59183inputs"Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
ŹBŠ
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59358inputs"Ě
Ă˛ż
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs

jtraining%
kwonlydefaultsŞ

trainingp 
annotationsŞ *
 
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ă	variables
ätrainable_variables
ĺregularization_losses
ç__call__
+č&call_and_return_all_conditional_losses
'č"call_and_return_conditional_losses"
_generic_user_object
2
˛
FullArgSpecx
argspm
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
defaults

 

 
p 
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
˛
FullArgSpecx
argspm
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
defaults

 

 
p 
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ö
 	variables
Ątrainable_variables
˘regularization_losses
Ł	keras_api
¤__call__
+Ľ&call_and_return_all_conditional_losses
Śpartial_output_shape
§full_output_shape
kernel
	bias"
_tf_keras_layer
ö
¨	variables
Štrainable_variables
Şregularization_losses
Ť	keras_api
Ź__call__
+­&call_and_return_all_conditional_losses
Žpartial_output_shape
Żfull_output_shape
kernel
	bias"
_tf_keras_layer
ö
°	variables
ątrainable_variables
˛regularization_losses
ł	keras_api
´__call__
+ľ&call_and_return_all_conditional_losses
śpartial_output_shape
ˇfull_output_shape
kernel
	bias"
_tf_keras_layer
Ť
¸	variables
štrainable_variables
şregularization_losses
ť	keras_api
ź__call__
+˝&call_and_return_all_conditional_losses"
_tf_keras_layer
Ă
ž	variables
żtrainable_variables
Ŕregularization_losses
Á	keras_api
Â__call__
+Ă&call_and_return_all_conditional_losses
Ä_random_generator"
_tf_keras_layer
ö
Ĺ	variables
Ćtrainable_variables
Çregularization_losses
Č	keras_api
É__call__
+Ę&call_and_return_all_conditional_losses
Ëpartial_output_shape
Ěfull_output_shape
kernel
	bias"
_tf_keras_layer
Ă
Í	variables
Îtrainable_variables
Ďregularization_losses
Đ	keras_api
Ń__call__
+Ň&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Ă
Ó	variables
Ôtrainable_variables
Őregularization_losses
Ö	keras_api
×__call__
+Ř&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
@
0
1
2
3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ůnon_trainable_variables
Úlayers
Űmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
ń	variables
ňtrainable_variables
óregularization_losses
ő__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
ĺ
Ţtrace_0
ßtrace_1
ŕtrace_2
átrace_32ň
*__inference_sequential_layer_call_fn_56300
*__inference_sequential_layer_call_fn_59583
*__inference_sequential_layer_call_fn_59596
*__inference_sequential_layer_call_fn_56373ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zŢtrace_0zßtrace_1zŕtrace_2zátrace_3
Ń
âtrace_0
ătrace_1
ätrace_2
ĺtrace_32Ţ
E__inference_sequential_layer_call_and_return_conditional_losses_59653
E__inference_sequential_layer_call_and_return_conditional_losses_59710
E__inference_sequential_layer_call_and_return_conditional_losses_56387
E__inference_sequential_layer_call_and_return_conditional_losses_56401ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zâtrace_0zătrace_1zätrace_2zĺtrace_3
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ćnon_trainable_variables
çlayers
čmetrics
 élayer_regularization_losses
ęlayer_metrics
÷	variables
řtrainable_variables
ůregularization_losses
ű__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
0
Ą0
˘1"
trackable_list_wrapper
0
Ą0
˘1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ënon_trainable_variables
ělayers
ímetrics
 îlayer_regularization_losses
ďlayer_metrics
ţ	variables
˙trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
čBĺ
4__inference_global_max_pooling1d_layer_call_fn_59363inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_59369inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ěBé
'__inference_dropout_layer_call_fn_59374inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ěBé
'__inference_dropout_layer_call_fn_59379inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_59384inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_59396inputs"ł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ŰBŘ
'__inference_dense_2_layer_call_fn_59405inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_dense_2_layer_call_and_return_conditional_losses_59415inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
R
đ	variables
ń	keras_api

ňtotal

ócount"
_tf_keras_metric
<::úŢ2"Adam/m/dem_time_dist_conv2d/kernel
<::úŢ2"Adam/v/dem_time_dist_conv2d/kernel
,:*2 Adam/m/dem_time_dist_conv2d/bias
,:*2 Adam/v/dem_time_dist_conv2d/bias
;:92#Adam/m/temp_time_dist_conv2d/kernel
;:92#Adam/v/temp_time_dist_conv2d/kernel
-:+2!Adam/m/temp_time_dist_conv2d/bias
-:+2!Adam/v/temp_time_dist_conv2d/bias
=:;2%Adam/m/precip_time_dist_conv2d/kernel
=:;2%Adam/v/precip_time_dist_conv2d/kernel
/:-2#Adam/m/precip_time_dist_conv2d/bias
/:-2#Adam/v/precip_time_dist_conv2d/bias
<::2"Adam/m/swe_time_dist_conv2d/kernel
<::2"Adam/v/swe_time_dist_conv2d/kernel
,:*2 Adam/m/swe_time_dist_conv2d/bias
,:*2 Adam/v/swe_time_dist_conv2d/bias
9:7!'2!Adam/m/et_time_dist_conv2d/kernel
9:7!'2!Adam/v/et_time_dist_conv2d/kernel
+:)2Adam/m/et_time_dist_conv2d/bias
+:)2Adam/v/et_time_dist_conv2d/bias
P:N

2<Adam/m/transformer_encoder/multi_head_attention/query/kernel
P:N

2<Adam/v/transformer_encoder/multi_head_attention/query/kernel
J:H
2:Adam/m/transformer_encoder/multi_head_attention/query/bias
J:H
2:Adam/v/transformer_encoder/multi_head_attention/query/bias
N:L

2:Adam/m/transformer_encoder/multi_head_attention/key/kernel
N:L

2:Adam/v/transformer_encoder/multi_head_attention/key/kernel
H:F
28Adam/m/transformer_encoder/multi_head_attention/key/bias
H:F
28Adam/v/transformer_encoder/multi_head_attention/key/bias
P:N

2<Adam/m/transformer_encoder/multi_head_attention/value/kernel
P:N

2<Adam/v/transformer_encoder/multi_head_attention/value/kernel
J:H
2:Adam/m/transformer_encoder/multi_head_attention/value/bias
J:H
2:Adam/v/transformer_encoder/multi_head_attention/value/bias
[:Y

2GAdam/m/transformer_encoder/multi_head_attention/attention_output/kernel
[:Y

2GAdam/v/transformer_encoder/multi_head_attention/attention_output/kernel
Q:O
2EAdam/m/transformer_encoder/multi_head_attention/attention_output/bias
Q:O
2EAdam/v/transformer_encoder/multi_head_attention/attention_output/bias
#:!
 2Adam/m/dense/kernel
#:!
 2Adam/v/dense/kernel
: 2Adam/m/dense/bias
: 2Adam/v/dense/bias
%:# 
2Adam/m/dense_1/kernel
%:# 
2Adam/v/dense_1/kernel
:
2Adam/m/dense_1/bias
:
2Adam/v/dense_1/bias
@:>
24Adam/m/transformer_encoder/layer_normalization/gamma
@:>
24Adam/v/transformer_encoder/layer_normalization/gamma
?:=
23Adam/m/transformer_encoder/layer_normalization/beta
?:=
23Adam/v/transformer_encoder/layer_normalization/beta
B:@
26Adam/m/transformer_encoder/layer_normalization_1/gamma
B:@
26Adam/v/transformer_encoder/layer_normalization_1/gamma
A:?
25Adam/m/transformer_encoder/layer_normalization_1/beta
A:?
25Adam/v/transformer_encoder/layer_normalization_1/beta
%:#
2Adam/m/dense_2/kernel
%:#
2Adam/v/dense_2/kernel
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
ÚB×
&__inference_conv2d_layer_call_fn_59424inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
őBň
A__inference_conv2d_layer_call_and_return_conditional_losses_59435inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÜBŮ
(__inference_conv2d_2_layer_call_fn_59444inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
C__inference_conv2d_2_layer_call_and_return_conditional_losses_59455inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÜBŮ
(__inference_conv2d_3_layer_call_fn_59464inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
C__inference_conv2d_3_layer_call_and_return_conditional_losses_59475inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÜBŮ
(__inference_conv2d_4_layer_call_fn_59484inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
C__inference_conv2d_4_layer_call_and_return_conditional_losses_59495inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÜBŮ
(__inference_conv2d_1_layer_call_fn_59504inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
C__inference_conv2d_1_layer_call_and_return_conditional_losses_59515inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ŰBŘ
'__inference_flatten_layer_call_fn_59520inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_flatten_layer_call_and_return_conditional_losses_59526inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÝBÚ
)__inference_flatten_2_layer_call_fn_59531inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_flatten_2_layer_call_and_return_conditional_losses_59537inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÝBÚ
)__inference_flatten_3_layer_call_fn_59542inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_flatten_3_layer_call_and_return_conditional_losses_59548inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÝBÚ
)__inference_flatten_4_layer_call_fn_59553inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_flatten_4_layer_call_and_return_conditional_losses_59559inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ÝBÚ
)__inference_flatten_1_layer_call_fn_59564inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_flatten_1_layer_call_and_return_conditional_losses_59570inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
P
é0
ę1
ë2
ě3
í4
î5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
őlayers
ömetrics
 ÷layer_regularization_losses
řlayer_metrics
 	variables
Ątrainable_variables
˘regularization_losses
¤__call__
+Ľ&call_and_return_all_conditional_losses
'Ľ"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ůnon_trainable_variables
úlayers
űmetrics
 ülayer_regularization_losses
ýlayer_metrics
¨	variables
Štrainable_variables
Şregularization_losses
Ź__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ţnon_trainable_variables
˙layers
metrics
 layer_regularization_losses
layer_metrics
°	variables
ątrainable_variables
˛regularization_losses
´__call__
+ľ&call_and_return_all_conditional_losses
'ľ"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
štrainable_variables
şregularization_losses
ź__call__
+˝&call_and_return_all_conditional_losses
'˝"call_and_return_conditional_losses"
_generic_user_object
ľ2˛Ż
Ś˛˘
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ľ2˛Ż
Ś˛˘
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults˘

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ž	variables
żtrainable_variables
Ŕregularization_losses
Â__call__
+Ă&call_and_return_all_conditional_losses
'Ă"call_and_return_conditional_losses"
_generic_user_object
š2śł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
š2śł
Ş˛Ś
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ĺ	variables
Ćtrainable_variables
Çregularization_losses
É__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses"
_generic_user_object
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
¨2Ľ˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Í	variables
Îtrainable_variables
Ďregularization_losses
Ń__call__
+Ň&call_and_return_all_conditional_losses
'Ň"call_and_return_conditional_losses"
_generic_user_object
ë
trace_02Ě
%__inference_dense_layer_call_fn_59719˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_59750˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Őregularization_losses
×__call__
+Ř&call_and_return_all_conditional_losses
'Ř"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_1_layer_call_fn_59759˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0

trace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_59789˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0
 "
trackable_list_wrapper
0
ď0
đ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bý
*__inference_sequential_layer_call_fn_56300dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
*__inference_sequential_layer_call_fn_59583inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
*__inference_sequential_layer_call_fn_59596inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Bý
*__inference_sequential_layer_call_fn_56373dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_59653inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_59710inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_56387dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_56401dense_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ň0
ó1"
trackable_list_wrapper
.
đ	variables"
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
ŮBÖ
%__inference_dense_layer_call_fn_59719inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ôBń
@__inference_dense_layer_call_and_return_conditional_losses_59750inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ŰBŘ
'__inference_dense_1_layer_call_fn_59759inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_59789inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ­
 __inference__wrapped_model_554938 Ą˘˘
˘

2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
1.
temp_inputs˙˙˙˙˙˙˙˙˙
30
precip_inputs˙˙˙˙˙˙˙˙˙
2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
Ş "1Ş.
,
dense_2!
dense_2˙˙˙˙˙˙˙˙˙ß
F__inference_concatenate_layer_call_and_return_conditional_losses_58934ß˘Ű
Ó˘Ď
ĚČ
&#
inputs_0˙˙˙˙˙˙˙˙˙

&#
inputs_1˙˙˙˙˙˙˙˙˙

&#
inputs_2˙˙˙˙˙˙˙˙˙

&#
inputs_3˙˙˙˙˙˙˙˙˙

&#
inputs_4˙˙˙˙˙˙˙˙˙

Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 š
+__inference_concatenate_layer_call_fn_58924ß˘Ű
Ó˘Ď
ĚČ
&#
inputs_0˙˙˙˙˙˙˙˙˙

&#
inputs_1˙˙˙˙˙˙˙˙˙

&#
inputs_2˙˙˙˙˙˙˙˙˙

&#
inputs_3˙˙˙˙˙˙˙˙˙

&#
inputs_4˙˙˙˙˙˙˙˙˙

Ş "%"
unknown˙˙˙˙˙˙˙˙˙
ź
C__inference_conv2d_1_layer_call_and_return_conditional_losses_59515u7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙2w
Ş "4˘1
*'
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_1_layer_call_fn_59504j7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙2w
Ş ")&
unknown˙˙˙˙˙˙˙˙˙ź
C__inference_conv2d_2_layer_call_and_return_conditional_losses_59455u7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "4˘1
*'
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_2_layer_call_fn_59444j7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ")&
unknown˙˙˙˙˙˙˙˙˙ź
C__inference_conv2d_3_layer_call_and_return_conditional_losses_59475u7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "4˘1
*'
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_3_layer_call_fn_59464j7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ")&
unknown˙˙˙˙˙˙˙˙˙ž
C__inference_conv2d_4_layer_call_and_return_conditional_losses_59495w9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙ĂÓ
Ş "4˘1
*'
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_4_layer_call_fn_59484l9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙ĂÓ
Ş ")&
unknown˙˙˙˙˙˙˙˙˙ź
A__inference_conv2d_layer_call_and_return_conditional_losses_59435w9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙÷
Ş "4˘1
*'
tensor_0˙˙˙˙˙˙˙˙˙
 
&__inference_conv2d_layer_call_fn_59424l9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙÷
Ş ")&
unknown˙˙˙˙˙˙˙˙˙Ô
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58722L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ô
F__inference_dem_flatten_layer_call_and_return_conditional_losses_58739L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 ­
+__inference_dem_flatten_layer_call_fn_58700~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
+__inference_dem_flatten_layer_call_fn_58705~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
í
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58407N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
p 

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
O__inference_dem_time_dist_conv2d_layer_call_and_return_conditional_losses_58431N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
p

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
4__inference_dem_time_dist_conv2d_layer_call_fn_58374N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
p 

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ç
4__inference_dem_time_dist_conv2d_layer_call_fn_58383N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙÷
p

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ł
B__inference_dense_1_layer_call_and_return_conditional_losses_59789m3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙ 
Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 
'__inference_dense_1_layer_call_fn_59759b3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙ 
Ş "%"
unknown˙˙˙˙˙˙˙˙˙
Ť
B__inference_dense_2_layer_call_and_return_conditional_losses_59415e/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
'__inference_dense_2_layer_call_fn_59405Z/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙

Ş "!
unknown˙˙˙˙˙˙˙˙˙ą
@__inference_dense_layer_call_and_return_conditional_losses_59750m3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙

Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙ 
 
%__inference_dense_layer_call_fn_59719b3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙

Ş "%"
unknown˙˙˙˙˙˙˙˙˙ Š
B__inference_dropout_layer_call_and_return_conditional_losses_59384c3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙

p 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 Š
B__inference_dropout_layer_call_and_return_conditional_losses_59396c3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙

p
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
'__inference_dropout_layer_call_fn_59374X3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙

p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙

'__inference_dropout_layer_call_fn_59379X3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙

p
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ó
E__inference_et_flatten_layer_call_and_return_conditional_losses_58898L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ó
E__inference_et_flatten_layer_call_and_return_conditional_losses_58915L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ź
*__inference_et_flatten_layer_call_fn_58876~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ź
*__inference_et_flatten_layer_call_fn_58881~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ę
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58671L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
p 

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ę
N__inference_et_time_dist_conv2d_layer_call_and_return_conditional_losses_58695L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
p

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ä
3__inference_et_time_dist_conv2d_layer_call_fn_58638L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
p 

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ä
3__inference_et_time_dist_conv2d_layer_call_fn_58647L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2w
p

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ż
D__inference_flatten_1_layer_call_and_return_conditional_losses_59570g7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
)__inference_flatten_1_layer_call_fn_59564\7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ż
D__inference_flatten_2_layer_call_and_return_conditional_losses_59537g7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
)__inference_flatten_2_layer_call_fn_59531\7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ż
D__inference_flatten_3_layer_call_and_return_conditional_losses_59548g7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
)__inference_flatten_3_layer_call_fn_59542\7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ż
D__inference_flatten_4_layer_call_and_return_conditional_losses_59559g7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
)__inference_flatten_4_layer_call_fn_59553\7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙
­
B__inference_flatten_layer_call_and_return_conditional_losses_59526g7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
'__inference_flatten_layer_call_fn_59520\7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ń
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_59369~E˘B
;˘8
63
inputs'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "5˘2
+(
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ť
4__inference_global_max_pooling1d_layer_call_fn_59363sE˘B
;˘8
63
inputs'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "*'
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Đ
@__inference_model_layer_call_and_return_conditional_losses_574708 Ą˘ ˘
˘

2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
1.
temp_inputs˙˙˙˙˙˙˙˙˙
30
precip_inputs˙˙˙˙˙˙˙˙˙
2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Đ
@__inference_model_layer_call_and_return_conditional_losses_575698 Ą˘ ˘
˘

2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
1.
temp_inputs˙˙˙˙˙˙˙˙˙
30
precip_inputs˙˙˙˙˙˙˙˙˙
2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ă
@__inference_model_layer_call_and_return_conditional_losses_58063ţ8 Ą˘˘
˘
řô
0-
inputs_0˙˙˙˙˙˙˙˙˙÷
.+
inputs_1˙˙˙˙˙˙˙˙˙
.+
inputs_2˙˙˙˙˙˙˙˙˙
0-
inputs_3˙˙˙˙˙˙˙˙˙ĂÓ
.+
inputs_4˙˙˙˙˙˙˙˙˙2w
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ă
@__inference_model_layer_call_and_return_conditional_losses_58365ţ8 Ą˘˘
˘
řô
0-
inputs_0˙˙˙˙˙˙˙˙˙÷
.+
inputs_1˙˙˙˙˙˙˙˙˙
.+
inputs_2˙˙˙˙˙˙˙˙˙
0-
inputs_3˙˙˙˙˙˙˙˙˙ĂÓ
.+
inputs_4˙˙˙˙˙˙˙˙˙2w
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ş
%__inference_model_layer_call_fn_567848 Ą˘ ˘
˘

2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
1.
temp_inputs˙˙˙˙˙˙˙˙˙
30
precip_inputs˙˙˙˙˙˙˙˙˙
2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ş
%__inference_model_layer_call_fn_573718 Ą˘ ˘
˘

2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
1.
temp_inputs˙˙˙˙˙˙˙˙˙
30
precip_inputs˙˙˙˙˙˙˙˙˙
2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
%__inference_model_layer_call_fn_57703ó8 Ą˘˘
˘
řô
0-
inputs_0˙˙˙˙˙˙˙˙˙÷
.+
inputs_1˙˙˙˙˙˙˙˙˙
.+
inputs_2˙˙˙˙˙˙˙˙˙
0-
inputs_3˙˙˙˙˙˙˙˙˙ĂÓ
.+
inputs_4˙˙˙˙˙˙˙˙˙2w
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
%__inference_model_layer_call_fn_57768ó8 Ą˘˘
˘
řô
0-
inputs_0˙˙˙˙˙˙˙˙˙÷
.+
inputs_1˙˙˙˙˙˙˙˙˙
.+
inputs_2˙˙˙˙˙˙˙˙˙
0-
inputs_3˙˙˙˙˙˙˙˙˙ĂÓ
.+
inputs_4˙˙˙˙˙˙˙˙˙2w
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙×
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58810L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 ×
I__inference_precip_flatten_layer_call_and_return_conditional_losses_58827L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 °
.__inference_precip_flatten_layer_call_fn_58788~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
°
.__inference_precip_flatten_layer_call_fn_58793~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
î
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58539L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 î
R__inference_precip_time_dist_conv2d_layer_call_and_return_conditional_losses_58563L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Č
7__inference_precip_time_dist_conv2d_layer_call_fn_58506L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Č
7__inference_precip_time_dist_conv2d_layer_call_fn_58515L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ç
E__inference_sequential_layer_call_and_return_conditional_losses_56387~@˘=
6˘3
)&
dense_input˙˙˙˙˙˙˙˙˙

p 

 
Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 Ç
E__inference_sequential_layer_call_and_return_conditional_losses_56401~@˘=
6˘3
)&
dense_input˙˙˙˙˙˙˙˙˙

p

 
Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 Â
E__inference_sequential_layer_call_and_return_conditional_losses_59653y;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙

p 

 
Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 Â
E__inference_sequential_layer_call_and_return_conditional_losses_59710y;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙

p

 
Ş "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 Ą
*__inference_sequential_layer_call_fn_56300s@˘=
6˘3
)&
dense_input˙˙˙˙˙˙˙˙˙

p 

 
Ş "%"
unknown˙˙˙˙˙˙˙˙˙
Ą
*__inference_sequential_layer_call_fn_56373s@˘=
6˘3
)&
dense_input˙˙˙˙˙˙˙˙˙

p

 
Ş "%"
unknown˙˙˙˙˙˙˙˙˙

*__inference_sequential_layer_call_fn_59583n;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙

p 

 
Ş "%"
unknown˙˙˙˙˙˙˙˙˙

*__inference_sequential_layer_call_fn_59596n;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙

p

 
Ş "%"
unknown˙˙˙˙˙˙˙˙˙
ň
#__inference_signature_wrapper_57638Ę8 Ą˘Ú˘Ö
˘ 
ÎŞĘ
@

dem_inputs2/

dem_inputs˙˙˙˙˙˙˙˙˙÷
<
	et_inputs/,
	et_inputs˙˙˙˙˙˙˙˙˙2w
D
precip_inputs30
precip_inputs˙˙˙˙˙˙˙˙˙
@

swe_inputs2/

swe_inputs˙˙˙˙˙˙˙˙˙ĂÓ
@
temp_inputs1.
temp_inputs˙˙˙˙˙˙˙˙˙"1Ş.
,
dense_2!
dense_2˙˙˙˙˙˙˙˙˙Ô
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58854L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ô
F__inference_swe_flatten_layer_call_and_return_conditional_losses_58871L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 ­
+__inference_swe_flatten_layer_call_fn_58832~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
+__inference_swe_flatten_layer_call_fn_58837~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
í
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58605N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
p 

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 í
O__inference_swe_time_dist_conv2d_layer_call_and_return_conditional_losses_58629N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
p

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
4__inference_swe_time_dist_conv2d_layer_call_fn_58572N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
p 

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ç
4__inference_swe_time_dist_conv2d_layer_call_fn_58581N˘K
D˘A
74
inputs(˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĂÓ
p

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ő
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58766L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ő
G__inference_temp_flatten_layer_call_and_return_conditional_losses_58783L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "9˘6
/,
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 Ž
,__inference_temp_flatten_layer_call_fn_58744~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ž
,__inference_temp_flatten_layer_call_fn_58749~L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş ".+
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ě
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58473L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ě
P__inference_temp_time_dist_conv2d_layer_call_and_return_conditional_losses_58497L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "A˘>
74
tensor_0&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ć
5__inference_temp_time_dist_conv2d_layer_call_fn_58440L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ć
5__inference_temp_time_dist_conv2d_layer_call_fn_58449L˘I
B˘?
52
inputs&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "63
unknown&˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙đ
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59183  Ą˘G˘D
-˘*
$!
inputs˙˙˙˙˙˙˙˙˙


 
Ş

trainingp "0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 đ
N__inference_transformer_encoder_layer_call_and_return_conditional_losses_59358  Ą˘G˘D
-˘*
$!
inputs˙˙˙˙˙˙˙˙˙


 
Ş

trainingp"0˘-
&#
tensor_0˙˙˙˙˙˙˙˙˙

 Ę
3__inference_transformer_encoder_layer_call_fn_58971  Ą˘G˘D
-˘*
$!
inputs˙˙˙˙˙˙˙˙˙


 
Ş

trainingp "%"
unknown˙˙˙˙˙˙˙˙˙
Ę
3__inference_transformer_encoder_layer_call_fn_59008  Ą˘G˘D
-˘*
$!
inputs˙˙˙˙˙˙˙˙˙


 
Ş

trainingp"%"
unknown˙˙˙˙˙˙˙˙˙
