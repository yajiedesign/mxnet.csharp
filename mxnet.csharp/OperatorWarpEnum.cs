using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
/// <summary>
/// Activation function to be applied.
/// </summary>
public enum LeakyreluActType
{Elu,
Leaky,
Prelu,
Rrelu
};
/// <summary>
/// Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.
/// </summary>
public enum PadMode
{Constant,
Edge,
Reflect
};
/// <summary>
/// Output storage type.
/// </summary>
public enum CastStorageStype
{Csr,
Default,
RowSparse
};
/// <summary>
/// Output data type.
/// </summary>
public enum CastDtype
{Float16,
Float32,
Float64,
Int32,
Uint8
};
/// <summary>
/// Data type of weight.
/// </summary>
public enum EmbeddingDtype
{Float16,
Float32,
Float64,
Int32,
Uint8
};
/// <summary>
/// Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error. 
/// </summary>
public enum TakeMode
{Clip,
Raise,
Wrap
};
/// <summary>
/// DType of the output
/// </summary>
public enum OneHotDtype
{Float16,
Float32,
Float64,
Int32,
Uint8
};
/// <summary>
/// The return type. "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.
/// </summary>
public enum TopkRetTyp
{Both,
Indices,
Mask,
Value
};
/// <summary>
/// upsampling method
/// </summary>
public enum UpsamplingSampleType
{Bilinear,
Nearest
};
/// <summary>
/// How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
/// </summary>
public enum UpsamplingMultiInputMode
{Concat,
Sum
};
/// <summary>
/// Activation function to be applied.
/// </summary>
public enum ActivationActType
{Relu,
Sigmoid,
Softrelu,
Tanh
};
/// <summary>
/// Whether to pick convolution algo by running performance test.
/// </summary>
public enum ConvolutionCudnnTune
{Fastest,
LimitedWorkspace,
Off
};
/// <summary>
/// Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
/// </summary>
public enum ConvolutionLayout
{NCDHW,
NCHW,
NCW,
NDHWC,
NHWC
};
/// <summary>
/// Whether to pick convolution algo by running performance test.    Leads to higher startup time but may give faster speed. Options are:    'off': no tuning    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.    'fastest': pick the fastest algorithm and ignore workspace limit.    If set to None (default), behavior is determined by environment    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,    1 for limited workspace (default), 2 for fastest.
/// </summary>
public enum ConvolutionV1CudnnTune
{Fastest,
LimitedWorkspace,
Off
};
/// <summary>
/// Set layout for input, output and weight. Empty for    default layout: NCHW for 2d and NCDHW for 3d.
/// </summary>
public enum ConvolutionV1Layout
{NCDHW,
NCHW,
NDHWC,
NHWC
};
/// <summary>
/// Whether to pick convolution algorithm by running performance test.
/// </summary>
public enum DeconvolutionCudnnTune
{Fastest,
LimitedWorkspace,
Off
};
/// <summary>
/// Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.
/// </summary>
public enum DeconvolutionLayout
{NCDHW,
NCHW,
NCW,
NDHWC,
NHWC
};
/// <summary>
/// Whether to only turn on dropout during training or to also turn on for inference.
/// </summary>
public enum DropoutMode
{Always,
Training
};
/// <summary>
/// The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
/// </summary>
public enum GridgeneratorTransformType
{Affine,
Warp
};
/// <summary>
/// Specify the dimension along which to compute L2 norm.
/// </summary>
public enum L2normalizationMode
{Channel,
Instance,
Spatial
};
/// <summary>
/// If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.
/// </summary>
public enum MakelossNormalization
{Batch,
Null,
Valid
};
/// <summary>
/// Pooling type to be applied.
/// </summary>
public enum PoolingPoolType
{Avg,
Max,
Sum
};
/// <summary>
/// Pooling convention to be applied.
/// </summary>
public enum PoolingPoolingConvention
{Full,
Valid
};
/// <summary>
/// Pooling type to be applied.
/// </summary>
public enum PoolingV1PoolType
{Avg,
Max,
Sum
};
/// <summary>
/// Pooling convention to be applied.
/// </summary>
public enum PoolingV1PoolingConvention
{Full,
Valid
};
/// <summary>
/// the type of RNN to compute
/// </summary>
public enum RNNMode
{Gru,
Lstm,
RnnRelu,
RnnTanh
};
/// <summary>
/// Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.
/// </summary>
public enum SoftmaxactivationMode
{Channel,
Instance
};
/// <summary>
/// Normalizes the gradient.
/// </summary>
public enum SoftmaxoutputNormalization
{Batch,
Null,
Valid
};
/// <summary>
/// Normalizes the gradient.
/// </summary>
public enum SoftmaxNormalization
{Batch,
Null,
Valid
};
/// <summary>
/// transformation type
/// </summary>
public enum SpatialtransformerTransformType
{Affine
};
/// <summary>
/// sampling type
/// </summary>
public enum SpatialtransformerSamplerType
{Bilinear
};

                              
}
