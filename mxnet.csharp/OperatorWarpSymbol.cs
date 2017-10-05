using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
    public partial class Symbol
    {
/// <summary>
/// Batch normalization.Normalizes a data batch by mean and variance, and applies a scale ``gamma`` aswell as offset ``beta``.Assume the input has more than one dimension and we normalize along axis 1.We first compute the mean and variance along this axis:.. math::  data\_mean[i] = mean(data[:,i,:,...]) \\  data\_var[i] = var(data[:,i,:,...])Then compute the normalized output, which has the same shape as input, as following:.. math::  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]Both *mean* and *var* returns a scalar by treating the input as a vector.Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and``data_var`` as well, which are needed for the backward pass.Besides the inputs and the outputs, this operator accepts two auxiliarystates, ``moving_mean`` and ``moving_var``, which are *k*-lengthvectors. They are global statistics for the whole dataset, which are updatedby::  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)  moving_var = moving_var * momentum + data_var * (1 - momentum)If ``use_global_stats`` is set to be true, then ``moving_mean`` and``moving_var`` are used instead of ``data_mean`` and ``data_var`` to computethe output. It is often used during inference.The parameter ``axis`` specifies which axis of the input shape denotesthe 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channelaxis to be the last item in the input shape.Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,then set ``gamma`` to 1 and its gradient to 0.Defined in G:\deeplearn\mxnet\src\operator\batch_norm.cc:L399
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to batch normalization</param>
/// <param name="gamma">gamma array</param>
/// <param name="beta">beta array</param>
/// <param name="moving_mean">running mean of input</param>
/// <param name="moving_var">running variance of input</param>
/// <param name="eps">Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
/// <param name="axis">Specify which shape axis the channel is specified</param>
/// <param name="cudnn_off">Do not select CUDNN operator, if available</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchNorm(string symbol_name,
Symbol data,
Symbol gamma,
Symbol beta,
Symbol moving_mean,
Symbol moving_var,
double eps=0.001,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false,
int axis=1,
bool cudnn_off=false)
{
return new Operator("BatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetParam("axis", axis)
.SetParam("cudnn_off", cudnn_off)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.SetInput("moving_mean", moving_mean)
.SetInput("moving_var", moving_var)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Batch normalization.Normalizes a data batch by mean and variance, and applies a scale ``gamma`` aswell as offset ``beta``.Assume the input has more than one dimension and we normalize along axis 1.We first compute the mean and variance along this axis:.. math::  data\_mean[i] = mean(data[:,i,:,...]) \\  data\_var[i] = var(data[:,i,:,...])Then compute the normalized output, which has the same shape as input, as following:.. math::  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]Both *mean* and *var* returns a scalar by treating the input as a vector.Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and``data_var`` as well, which are needed for the backward pass.Besides the inputs and the outputs, this operator accepts two auxiliarystates, ``moving_mean`` and ``moving_var``, which are *k*-lengthvectors. They are global statistics for the whole dataset, which are updatedby::  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)  moving_var = moving_var * momentum + data_var * (1 - momentum)If ``use_global_stats`` is set to be true, then ``moving_mean`` and``moving_var`` are used instead of ``data_mean`` and ``data_var`` to computethe output. It is often used during inference.The parameter ``axis`` specifies which axis of the input shape denotesthe 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channelaxis to be the last item in the input shape.Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,then set ``gamma`` to 1 and its gradient to 0.Defined in G:\deeplearn\mxnet\src\operator\batch_norm.cc:L399
/// </summary>
/// <param name="data">Input data to batch normalization</param>
/// <param name="gamma">gamma array</param>
/// <param name="beta">beta array</param>
/// <param name="moving_mean">running mean of input</param>
/// <param name="moving_var">running variance of input</param>
/// <param name="eps">Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
/// <param name="axis">Specify which shape axis the channel is specified</param>
/// <param name="cudnn_off">Do not select CUDNN operator, if available</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchNorm(Symbol data,
Symbol gamma,
Symbol beta,
Symbol moving_mean,
Symbol moving_var,
double eps=0.001,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false,
int axis=1,
bool cudnn_off=false)
{
return new Operator("BatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetParam("axis", axis)
.SetParam("cudnn_off", cudnn_off)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.SetInput("moving_mean", moving_mean)
.SetInput("moving_var", moving_var)
.CreateSymbol();
}
/// <summary>
/// Batch normalization.Normalizes a data batch by mean and variance, and applies a scale ``gamma`` aswell as offset ``beta``.Assume the input has more than one dimension and we normalize along axis 1.We first compute the mean and variance along this axis:.. math::  data\_mean[i] = mean(data[:,i,:,...]) \\  data\_var[i] = var(data[:,i,:,...])Then compute the normalized output, which has the same shape as input, as following:.. math::  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]Both *mean* and *var* returns a scalar by treating the input as a vector.Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and``data_var`` as well, which are needed for the backward pass.Besides the inputs and the outputs, this operator accepts two auxiliarystates, ``moving_mean`` and ``moving_var``, which are *k*-lengthvectors. They are global statistics for the whole dataset, which are updatedby::  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)  moving_var = moving_var * momentum + data_var * (1 - momentum)If ``use_global_stats`` is set to be true, then ``moving_mean`` and``moving_var`` are used instead of ``data_mean`` and ``data_var`` to computethe output. It is often used during inference.Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,then set ``gamma`` to 1 and its gradient to 0.Defined in G:\deeplearn\mxnet\src\operator\batch_norm_v1.cc:L89
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to batch normalization</param>
/// <param name="gamma">gamma array</param>
/// <param name="beta">beta array</param>
/// <param name="eps">Epsilon to prevent div 0</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchNormV1(string symbol_name,
Symbol data,
Symbol gamma,
Symbol beta,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false)
{
return new Operator("BatchNorm_v1")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Batch normalization.Normalizes a data batch by mean and variance, and applies a scale ``gamma`` aswell as offset ``beta``.Assume the input has more than one dimension and we normalize along axis 1.We first compute the mean and variance along this axis:.. math::  data\_mean[i] = mean(data[:,i,:,...]) \\  data\_var[i] = var(data[:,i,:,...])Then compute the normalized output, which has the same shape as input, as following:.. math::  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]Both *mean* and *var* returns a scalar by treating the input as a vector.Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and``data_var`` as well, which are needed for the backward pass.Besides the inputs and the outputs, this operator accepts two auxiliarystates, ``moving_mean`` and ``moving_var``, which are *k*-lengthvectors. They are global statistics for the whole dataset, which are updatedby::  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)  moving_var = moving_var * momentum + data_var * (1 - momentum)If ``use_global_stats`` is set to be true, then ``moving_mean`` and``moving_var`` are used instead of ``data_mean`` and ``data_var`` to computethe output. It is often used during inference.Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,then set ``gamma`` to 1 and its gradient to 0.Defined in G:\deeplearn\mxnet\src\operator\batch_norm_v1.cc:L89
/// </summary>
/// <param name="data">Input data to batch normalization</param>
/// <param name="gamma">gamma array</param>
/// <param name="beta">beta array</param>
/// <param name="eps">Epsilon to prevent div 0</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchNormV1(Symbol data,
Symbol gamma,
Symbol beta,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false)
{
return new Operator("BatchNorm_v1")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.CreateSymbol();
}
/// <summary>
/// Joins input arrays along a given axis... note:: `Concat` is deprecated. Use `concat` instead.The dimensions of the input arrays should be the same except the axis alongwhich they will be concatenated.The dimension of the output array along the concatenated axis will be equalto the sum of the corresponding dimensions of the input arrays.Example::   x = [[1,1],[2,2]]   y = [[3,3],[4,4],[5,5]]   z = [[6,6], [7,7],[8,8]]   concat(x,y,z,dim=0) = [[ 1.,  1.],                          [ 2.,  2.],                          [ 3.,  3.],                          [ 4.,  4.],                          [ 5.,  5.],                          [ 6.,  6.],                          [ 7.,  7.],                          [ 8.,  8.]]   Note that you cannot concat x,y,z along dimension 1 since dimension   0 is not the same for all the input arrays.   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],                         [ 4.,  4.,  7.,  7.],                         [ 5.,  5.,  8.,  8.]]Defined in G:\deeplearn\mxnet\src\operator\concat.cc:L98
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">List of arrays to concatenate</param>
/// <param name="num_args">Number of inputs to be concated.</param>
/// <param name="dim">the dimension to be concated.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Concat(string symbol_name,
Symbol[] data,
int num_args,
int dim=1)
{
return new Operator("Concat")
.SetParam("num_args", num_args)
.SetParam("dim", dim)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Joins input arrays along a given axis... note:: `Concat` is deprecated. Use `concat` instead.The dimensions of the input arrays should be the same except the axis alongwhich they will be concatenated.The dimension of the output array along the concatenated axis will be equalto the sum of the corresponding dimensions of the input arrays.Example::   x = [[1,1],[2,2]]   y = [[3,3],[4,4],[5,5]]   z = [[6,6], [7,7],[8,8]]   concat(x,y,z,dim=0) = [[ 1.,  1.],                          [ 2.,  2.],                          [ 3.,  3.],                          [ 4.,  4.],                          [ 5.,  5.],                          [ 6.,  6.],                          [ 7.,  7.],                          [ 8.,  8.]]   Note that you cannot concat x,y,z along dimension 1 since dimension   0 is not the same for all the input arrays.   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],                         [ 4.,  4.,  7.,  7.],                         [ 5.,  5.,  8.,  8.]]Defined in G:\deeplearn\mxnet\src\operator\concat.cc:L98
/// </summary>
/// <param name="data">List of arrays to concatenate</param>
/// <param name="num_args">Number of inputs to be concated.</param>
/// <param name="dim">the dimension to be concated.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Concat(Symbol[] data,
int num_args,
int dim=1)
{
return new Operator("Concat")
.SetParam("num_args", num_args)
.SetParam("dim", dim)
.AddInput(data)
.CreateSymbol();
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to batch normalization</param>
/// <param name="eps">Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
/// <param name="axis">Specify which shape axis the channel is specified</param>
/// <param name="cudnn_off">Do not select CUDNN operator, if available</param>
 /// <returns>returns new symbol</returns>
public static Symbol CuDNNBatchNorm(string symbol_name,
Symbol data,
double eps=0.001,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false,
int axis=1,
bool cudnn_off=false)
{
return new Operator("CuDNNBatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetParam("axis", axis)
.SetParam("cudnn_off", cudnn_off)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="data">Input data to batch normalization</param>
/// <param name="eps">Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)</param>
/// <param name="momentum">Momentum for moving average</param>
/// <param name="fix_gamma">Fix gamma while training</param>
/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param>
/// <param name="output_mean_var">Output All,normal mean and var</param>
/// <param name="axis">Specify which shape axis the channel is specified</param>
/// <param name="cudnn_off">Do not select CUDNN operator, if available</param>
 /// <returns>returns new symbol</returns>
public static Symbol CuDNNBatchNorm(Symbol data,
double eps=0.001,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false,
bool output_mean_var=false,
int axis=1,
bool cudnn_off=false)
{
return new Operator("CuDNNBatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetParam("output_mean_var", output_mean_var)
.SetParam("axis", axis)
.SetParam("cudnn_off", cudnn_off)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Apply a custom operator implemented in a frontend language (like Python).Custom operators should override required methods like `forward` and `backward`.The custom operator must be registered before it can be used.Please check the tutorial here: http://mxnet.io/how_to/new_op.html.Defined in G:\deeplearn\mxnet\src\operator\custom\custom.cc:L369
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data for the custom operator.</param>
/// <param name="op_type">Name of the custom operator. This is the name that is passed to `mx.operator.register` to register the operator.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Custom(string symbol_name,
Symbol[] data,
string op_type)
{
return new Operator("Custom")
.SetParam("op_type", op_type)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply a custom operator implemented in a frontend language (like Python).Custom operators should override required methods like `forward` and `backward`.The custom operator must be registered before it can be used.Please check the tutorial here: http://mxnet.io/how_to/new_op.html.Defined in G:\deeplearn\mxnet\src\operator\custom\custom.cc:L369
/// </summary>
/// <param name="data">Input data for the custom operator.</param>
/// <param name="op_type">Name of the custom operator. This is the name that is passed to `mx.operator.register` to register the operator.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Custom(Symbol[] data,
string op_type)
{
return new Operator("Custom")
.SetParam("op_type", op_type)
.AddInput(data)
.CreateSymbol();
}
/// <summary>
/// Apply a sparse regularization to the output a sigmoid activation function.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data.</param>
/// <param name="sparseness_target">The sparseness target</param>
/// <param name="penalty">The tradeoff parameter for the sparseness penalty</param>
/// <param name="momentum">The momentum for running average</param>
 /// <returns>returns new symbol</returns>
public static Symbol IdentityAttachKLSparseReg(string symbol_name,
Symbol data,
float sparseness_target=0.1f,
float penalty=0.001f,
float momentum=0.9f)
{
return new Operator("IdentityAttachKLSparseReg")
.SetParam("sparseness_target", sparseness_target)
.SetParam("penalty", penalty)
.SetParam("momentum", momentum)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply a sparse regularization to the output a sigmoid activation function.
/// </summary>
/// <param name="data">Input data.</param>
/// <param name="sparseness_target">The sparseness target</param>
/// <param name="penalty">The tradeoff parameter for the sparseness penalty</param>
/// <param name="momentum">The momentum for running average</param>
 /// <returns>returns new symbol</returns>
public static Symbol IdentityAttachKLSparseReg(Symbol data,
float sparseness_target=0.1f,
float penalty=0.001f,
float momentum=0.9f)
{
return new Operator("IdentityAttachKLSparseReg")
.SetParam("sparseness_target", sparseness_target)
.SetParam("penalty", penalty)
.SetParam("momentum", momentum)
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> LeakyreluActTypeConvert = new List<string>(){"elu","leaky","prelu","rrelu"};
/// <summary>
/// Applies Leaky rectified linear unit activation element-wise to the input.Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`when the input is negative and has a slope of one when input is positive.The following modified ReLU Activation functions are supported:- *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`- *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`- *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.- *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from  *[lower_bound, upper_bound)* for training, while fixed to be  *(lower_bound+upper_bound)/2* for inference.Defined in G:\deeplearn\mxnet\src\operator\leaky_relu.cc:L57
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to activation function.</param>
/// <param name="act_type">Activation function to be applied.</param>
/// <param name="slope">Init slope for the activation. (For leaky and elu only)</param>
/// <param name="lower_bound">Lower bound of random slope. (For rrelu only)</param>
/// <param name="upper_bound">Upper bound of random slope. (For rrelu only)</param>
 /// <returns>returns new symbol</returns>
public static Symbol LeakyReLU(string symbol_name,
Symbol data,
LeakyreluActType act_type=LeakyreluActType.Leaky,
float slope=0.25f,
float lower_bound=0.125f,
float upper_bound=0.334f)
{
return new Operator("LeakyReLU")
.SetParam("act_type", Util.EnumToString<LeakyreluActType>(act_type,LeakyreluActTypeConvert))
.SetParam("slope", slope)
.SetParam("lower_bound", lower_bound)
.SetParam("upper_bound", upper_bound)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies Leaky rectified linear unit activation element-wise to the input.Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`when the input is negative and has a slope of one when input is positive.The following modified ReLU Activation functions are supported:- *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`- *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`- *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.- *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from  *[lower_bound, upper_bound)* for training, while fixed to be  *(lower_bound+upper_bound)/2* for inference.Defined in G:\deeplearn\mxnet\src\operator\leaky_relu.cc:L57
/// </summary>
/// <param name="data">Input data to activation function.</param>
/// <param name="act_type">Activation function to be applied.</param>
/// <param name="slope">Init slope for the activation. (For leaky and elu only)</param>
/// <param name="lower_bound">Lower bound of random slope. (For rrelu only)</param>
/// <param name="upper_bound">Upper bound of random slope. (For rrelu only)</param>
 /// <returns>returns new symbol</returns>
public static Symbol LeakyReLU(Symbol data,
LeakyreluActType act_type=LeakyreluActType.Leaky,
float slope=0.25f,
float lower_bound=0.125f,
float upper_bound=0.334f)
{
return new Operator("LeakyReLU")
.SetParam("act_type", Util.EnumToString<LeakyreluActType>(act_type,LeakyreluActTypeConvert))
.SetParam("slope", slope)
.SetParam("lower_bound", lower_bound)
.SetParam("upper_bound", upper_bound)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Calculate cross entropy of softmax output and one-hot label.- This operator computes the cross entropy in two steps:  - Applies softmax function on the input array.  - Computes and returns the cross entropy loss between the softmax output and the labels.- The softmax function and cross entropy loss is given by:  - Softmax Function:  .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}  - Cross Entropy Function:  .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)Example::  x = [[1, 2, 3],       [11, 7, 5]]  label = [2, 0]  softmax(x) = [[0.09003057, 0.24472848, 0.66524094],                [0.97962922, 0.01794253, 0.00242826]]  softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871Defined in G:\deeplearn\mxnet\src\operator\loss_binary_op.cc:L58
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data</param>
/// <param name="label">Input label</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxCrossEntropy(string symbol_name,
Symbol data,
Symbol label)
{
return new Operator("softmax_cross_entropy")
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate cross entropy of softmax output and one-hot label.- This operator computes the cross entropy in two steps:  - Applies softmax function on the input array.  - Computes and returns the cross entropy loss between the softmax output and the labels.- The softmax function and cross entropy loss is given by:  - Softmax Function:  .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}  - Cross Entropy Function:  .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)Example::  x = [[1, 2, 3],       [11, 7, 5]]  label = [2, 0]  softmax(x) = [[0.09003057, 0.24472848, 0.66524094],                [0.97962922, 0.01794253, 0.00242826]]  softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871Defined in G:\deeplearn\mxnet\src\operator\loss_binary_op.cc:L58
/// </summary>
/// <param name="data">Input data</param>
/// <param name="label">Input label</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxCrossEntropy(Symbol data,
Symbol label)
{
return new Operator("softmax_cross_entropy")
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Applies the softmax function.The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1... math::   softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}for :math:`j = 1, ..., K`Example::  x = [[ 1.  1.  1.]       [ 1.  1.  1.]]  softmax(x,axis=0) = [[ 0.5  0.5  0.5]                       [ 0.5  0.5  0.5]]  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],                       [ 0.33333334,  0.33333334,  0.33333334]]Defined in G:\deeplearn\mxnet\src\operator\nn\softmax.cc:L53
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
/// <param name="axis">The axis along which to compute softmax.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Softmax(string symbol_name,
Symbol data,
int axis=-1)
{
return new Operator("softmax")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies the softmax function.The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1... math::   softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}for :math:`j = 1, ..., K`Example::  x = [[ 1.  1.  1.]       [ 1.  1.  1.]]  softmax(x,axis=0) = [[ 0.5  0.5  0.5]                       [ 0.5  0.5  0.5]]  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],                       [ 0.33333334,  0.33333334,  0.33333334]]Defined in G:\deeplearn\mxnet\src\operator\nn\softmax.cc:L53
/// </summary>
/// <param name="data">The input array.</param>
/// <param name="axis">The axis along which to compute softmax.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Softmax(Symbol data,
int axis=-1)
{
return new Operator("softmax")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the log softmax of the input.This is equivalent to computing softmax followed by log.Examples::  >>> x = mx.nd.array([1, 2, .1])  >>> mx.nd.log_softmax(x).asnumpy()  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )  >>> mx.nd.log_softmax(x, axis=0).asnumpy()  array([[-0.34115392, -0.69314718, -1.24115396],         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
/// <param name="axis">The axis along which to compute softmax.</param>
 /// <returns>returns new symbol</returns>
public static Symbol LogSoftmax(string symbol_name,
Symbol data,
int axis=-1)
{
return new Operator("log_softmax")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the log softmax of the input.This is equivalent to computing softmax followed by log.Examples::  >>> x = mx.nd.array([1, 2, .1])  >>> mx.nd.log_softmax(x).asnumpy()  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )  >>> mx.nd.log_softmax(x, axis=0).asnumpy()  array([[-0.34115392, -0.69314718, -1.24115396],         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
/// </summary>
/// <param name="data">The input array.</param>
/// <param name="axis">The axis along which to compute softmax.</param>
 /// <returns>returns new symbol</returns>
public static Symbol LogSoftmax(Symbol data,
int axis=-1)
{
return new Operator("log_softmax")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Update function for Stochastic Gradient Descent (SDG) optimizer.It updates the weights using:: weight = weight - learning_rate * gradientIf weight is of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated:: for row in gradient.indices:     weight[row] = weight[row] - learning_rate * gradient[row]Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L51
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol SgdUpdate(string symbol_name,
Symbol grad,
float lr,
Symbol weight=null,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("sgd_update")
.SetParam("lr", lr)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Update function for Stochastic Gradient Descent (SDG) optimizer.It updates the weights using:: weight = weight - learning_rate * gradientIf weight is of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated:: for row in gradient.indices:     weight[row] = weight[row] - learning_rate * gradient[row]Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L51
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol SgdUpdate(Symbol grad,
float lr,
Symbol weight=null,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("sgd_update")
.SetParam("lr", lr)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Momentum update function for Stochastic Gradient Descent (SDG) optimizer.Momentum update has better convergence rates on neural networks. Mathematically it lookslike below:.. math::  v_1 = \alpha * \nabla J(W_0)\\  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\  W_t = W_{t-1} + v_tIt updates the weights using::  v = momentum * v - learning_rate * gradient  weight += vWhere the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.If weight and momentum are both of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::  for row in gradient.indices:      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]      weight[row] += v[row]Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L90
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="mom">Momentum</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol SgdMomUpdate(string symbol_name,
Symbol grad,
Symbol mom,
float lr,
Symbol weight=null,
float momentum=0f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("sgd_mom_update")
.SetParam("lr", lr)
.SetParam("momentum", momentum)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mom", mom)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Momentum update function for Stochastic Gradient Descent (SDG) optimizer.Momentum update has better convergence rates on neural networks. Mathematically it lookslike below:.. math::  v_1 = \alpha * \nabla J(W_0)\\  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\  W_t = W_{t-1} + v_tIt updates the weights using::  v = momentum * v - learning_rate * gradient  weight += vWhere the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.If weight and momentum are both of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::  for row in gradient.indices:      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]      weight[row] += v[row]Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L90
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="mom">Momentum</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol SgdMomUpdate(Symbol grad,
Symbol mom,
float lr,
Symbol weight=null,
float momentum=0f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("sgd_mom_update")
.SetParam("lr", lr)
.SetParam("momentum", momentum)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mom", mom)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Updater function for multi-precision sgd optimizer
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">gradient</param>
/// <param name="weight32">Weight32</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol MpSgdUpdate(string symbol_name,
Symbol grad,
Symbol weight32,
float lr,
Symbol weight=null,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("mp_sgd_update")
.SetParam("lr", lr)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("weight32", weight32)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Updater function for multi-precision sgd optimizer
/// </summary>
/// <param name="grad">gradient</param>
/// <param name="weight32">Weight32</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol MpSgdUpdate(Symbol grad,
Symbol weight32,
float lr,
Symbol weight=null,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("mp_sgd_update")
.SetParam("lr", lr)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("weight32", weight32)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Updater function for multi-precision sgd optimizer
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="mom">Momentum</param>
/// <param name="weight32">Weight32</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol MpSgdMomUpdate(string symbol_name,
Symbol grad,
Symbol mom,
Symbol weight32,
float lr,
Symbol weight=null,
float momentum=0f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("mp_sgd_mom_update")
.SetParam("lr", lr)
.SetParam("momentum", momentum)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mom", mom)
.SetInput("weight32", weight32)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Updater function for multi-precision sgd optimizer
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="mom">Momentum</param>
/// <param name="weight32">Weight32</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol MpSgdMomUpdate(Symbol grad,
Symbol mom,
Symbol weight32,
float lr,
Symbol weight=null,
float momentum=0f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("mp_sgd_mom_update")
.SetParam("lr", lr)
.SetParam("momentum", momentum)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mom", mom)
.SetInput("weight32", weight32)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Update function for Adam optimizer. Adam is seen as a generalizationof AdaGrad.Adam update consists of the following steps, where g represents gradient and m, vare 1st and 2nd order moment estimates (mean and variance)... math:: g_t = \nabla J(W_{t-1})\\ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\ W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }It updates the weights using:: m = beta1*m + (1-beta1)*grad v = beta2*v + (1-beta2)*(grad**2) w += - learning_rate * m / (sqrt(v) + epsilon)If w, m and v are all of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for w, m and v):: for row in grad.indices:     m[row] = beta1*m[row] + (1-beta1)*grad[row]     v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)     w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L171
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="mean">Moving mean</param>
/// <param name="var">Moving variance</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="beta1">The decay rate for the 1st moment estimates.</param>
/// <param name="beta2">The decay rate for the 2nd moment estimates.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol AdamUpdate(string symbol_name,
Symbol grad,
Symbol mean,
Symbol var,
float lr,
Symbol weight=null,
float beta1=0.9f,
float beta2=0.999f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("adam_update")
.SetParam("lr", lr)
.SetParam("beta1", beta1)
.SetParam("beta2", beta2)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mean", mean)
.SetInput("var", var)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Update function for Adam optimizer. Adam is seen as a generalizationof AdaGrad.Adam update consists of the following steps, where g represents gradient and m, vare 1st and 2nd order moment estimates (mean and variance)... math:: g_t = \nabla J(W_{t-1})\\ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\ W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }It updates the weights using:: m = beta1*m + (1-beta1)*grad v = beta2*v + (1-beta2)*(grad**2) w += - learning_rate * m / (sqrt(v) + epsilon)If w, m and v are all of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for w, m and v):: for row in grad.indices:     m[row] = beta1*m[row] + (1-beta1)*grad[row]     v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)     w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L171
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="mean">Moving mean</param>
/// <param name="var">Moving variance</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="beta1">The decay rate for the 1st moment estimates.</param>
/// <param name="beta2">The decay rate for the 2nd moment estimates.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol AdamUpdate(Symbol grad,
Symbol mean,
Symbol var,
float lr,
Symbol weight=null,
float beta1=0.9f,
float beta2=0.999f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("adam_update")
.SetParam("lr", lr)
.SetParam("beta1", beta1)
.SetParam("beta2", beta2)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("mean", mean)
.SetInput("var", var)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Update function for `RMSProp` optimizer.`RMSprop` is a variant of stochastic gradient descent where the gradients aredivided by a cache which grows with the sum of squares of recent gradients?`RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptivelytunes the learning rate of each parameter. `AdaGrad` lowers the learning rate foreach parameter monotonically over the course of training.While this is analytically motivated for convex optimizations, it may not be idealfor non-convex problems. `RMSProp` deals with this heuristically by allowing thelearning rates to rebound as the denominator decays over time.Define the Root Mean Square (RMS) error criterion of the gradient as:math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` representsgradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.The :math:`E[g^2]_t` is given by:.. math::  E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2The update step is.. math::  \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_tThe RMSProp code follows the version inhttp://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdfTieleman & Hinton, 2012.Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate:math:`\eta` to be 0.001.Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L225
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="n">n</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="gamma1">The decay rate of momentum estimates.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
/// <param name="clip_weights">Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
 /// <returns>returns new symbol</returns>
public static Symbol RmspropUpdate(string symbol_name,
Symbol grad,
Symbol n,
float lr,
Symbol weight=null,
float gamma1=0.95f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f,
float clip_weights=-1f)
{
return new Operator("rmsprop_update")
.SetParam("lr", lr)
.SetParam("gamma1", gamma1)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetParam("clip_weights", clip_weights)
.SetInput("grad", grad)
.SetInput("n", n)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Update function for `RMSProp` optimizer.`RMSprop` is a variant of stochastic gradient descent where the gradients aredivided by a cache which grows with the sum of squares of recent gradients?`RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptivelytunes the learning rate of each parameter. `AdaGrad` lowers the learning rate foreach parameter monotonically over the course of training.While this is analytically motivated for convex optimizations, it may not be idealfor non-convex problems. `RMSProp` deals with this heuristically by allowing thelearning rates to rebound as the denominator decays over time.Define the Root Mean Square (RMS) error criterion of the gradient as:math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` representsgradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.The :math:`E[g^2]_t` is given by:.. math::  E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2The update step is.. math::  \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_tThe RMSProp code follows the version inhttp://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdfTieleman & Hinton, 2012.Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate:math:`\eta` to be 0.001.Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L225
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="n">n</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="gamma1">The decay rate of momentum estimates.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
/// <param name="clip_weights">Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
 /// <returns>returns new symbol</returns>
public static Symbol RmspropUpdate(Symbol grad,
Symbol n,
float lr,
Symbol weight=null,
float gamma1=0.95f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f,
float clip_weights=-1f)
{
return new Operator("rmsprop_update")
.SetParam("lr", lr)
.SetParam("gamma1", gamma1)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetParam("clip_weights", clip_weights)
.SetInput("grad", grad)
.SetInput("n", n)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Update function for RMSPropAlex optimizer.`RMSPropAlex` is non-centered version of `RMSProp`.Define :math:`E[g^2]_t` is the decaying average over past squared gradient and:math:`E[g]_t` is the decaying average over past gradient... math::  E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\  E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\  \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\The update step is.. math::  \theta_{t+1} = \theta_t + \Delta_tThe RMSPropAlex code follows the version inhttp://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`to be 0.9 and the learning rate :math:`\eta` to be 0.0001.Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L264
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="n">n</param>
/// <param name="g">g</param>
/// <param name="delta">delta</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="gamma1">Decay rate.</param>
/// <param name="gamma2">Decay rate.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
/// <param name="clip_weights">Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
 /// <returns>returns new symbol</returns>
public static Symbol RmspropalexUpdate(string symbol_name,
Symbol grad,
Symbol n,
Symbol g,
Symbol delta,
float lr,
Symbol weight=null,
float gamma1=0.95f,
float gamma2=0.9f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f,
float clip_weights=-1f)
{
return new Operator("rmspropalex_update")
.SetParam("lr", lr)
.SetParam("gamma1", gamma1)
.SetParam("gamma2", gamma2)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetParam("clip_weights", clip_weights)
.SetInput("grad", grad)
.SetInput("n", n)
.SetInput("g", g)
.SetInput("delta", delta)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Update function for RMSPropAlex optimizer.`RMSPropAlex` is non-centered version of `RMSProp`.Define :math:`E[g^2]_t` is the decaying average over past squared gradient and:math:`E[g]_t` is the decaying average over past gradient... math::  E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\  E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\  \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\The update step is.. math::  \theta_{t+1} = \theta_t + \Delta_tThe RMSPropAlex code follows the version inhttp://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`to be 0.9 and the learning rate :math:`\eta` to be 0.0001.Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L264
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="n">n</param>
/// <param name="g">g</param>
/// <param name="delta">delta</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="gamma1">Decay rate.</param>
/// <param name="gamma2">Decay rate.</param>
/// <param name="epsilon">A small constant for numerical stability.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
/// <param name="clip_weights">Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
 /// <returns>returns new symbol</returns>
public static Symbol RmspropalexUpdate(Symbol grad,
Symbol n,
Symbol g,
Symbol delta,
float lr,
Symbol weight=null,
float gamma1=0.95f,
float gamma2=0.9f,
float epsilon=1e-08f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f,
float clip_weights=-1f)
{
return new Operator("rmspropalex_update")
.SetParam("lr", lr)
.SetParam("gamma1", gamma1)
.SetParam("gamma2", gamma2)
.SetParam("epsilon", epsilon)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetParam("clip_weights", clip_weights)
.SetInput("grad", grad)
.SetInput("n", n)
.SetInput("g", g)
.SetInput("delta", delta)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Update function for Ftrl optimizer.Referenced from *Ad Click Prediction: a View from the Trenches*, available athttp://dl.acm.org/citation.cfm?id=2488200.It updates the weights using:: rescaled_grad = clip(grad * rescale_grad, clip_gradient) z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate n += rescaled_grad**2 w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)If w, z and n are all of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for w, z and n):: for row in grad.indices:     rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)     z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate     n[row] += rescaled_grad[row]**2     w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L303
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="grad">Gradient</param>
/// <param name="z">z</param>
/// <param name="n">Square of grad</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="lamda1">The L1 regularization coefficient.</param>
/// <param name="beta">Per-Coordinate Learning Rate beta.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol FtrlUpdate(string symbol_name,
Symbol grad,
Symbol z,
Symbol n,
float lr,
Symbol weight=null,
float lamda1=0.01f,
float beta=1f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("ftrl_update")
.SetParam("lr", lr)
.SetParam("lamda1", lamda1)
.SetParam("beta", beta)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("z", z)
.SetInput("n", n)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Update function for Ftrl optimizer.Referenced from *Ad Click Prediction: a View from the Trenches*, available athttp://dl.acm.org/citation.cfm?id=2488200.It updates the weights using:: rescaled_grad = clip(grad * rescale_grad, clip_gradient) z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate n += rescaled_grad**2 w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)If w, z and n are all of ``row_sparse`` storage type,only the row slices whose indices appear in grad.indices are updated (for w, z and n):: for row in grad.indices:     rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)     z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate     n[row] += rescaled_grad[row]**2     w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)Defined in G:\deeplearn\mxnet\src\operator\optimizer_op.cc:L303
/// </summary>
/// <param name="grad">Gradient</param>
/// <param name="z">z</param>
/// <param name="n">Square of grad</param>
/// <param name="lr">Learning rate</param>
/// <param name="weight">Weight</param>
/// <param name="lamda1">The L1 regularization coefficient.</param>
/// <param name="beta">Per-Coordinate Learning Rate beta.</param>
/// <param name="wd">Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.</param>
/// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
/// <param name="clip_gradient">Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
 /// <returns>returns new symbol</returns>
public static Symbol FtrlUpdate(Symbol grad,
Symbol z,
Symbol n,
float lr,
Symbol weight=null,
float lamda1=0.01f,
float beta=1f,
float wd=0f,
float rescale_grad=1f,
float clip_gradient=-1f)
{
return new Operator("ftrl_update")
.SetParam("lr", lr)
.SetParam("lamda1", lamda1)
.SetParam("beta", beta)
.SetParam("wd", wd)
.SetParam("rescale_grad", rescale_grad)
.SetParam("clip_gradient", clip_gradient)
.SetInput("grad", grad)
.SetInput("z", z)
.SetInput("n", n)
.SetInput("weight", weight)
.CreateSymbol();
}
private static readonly List<string> PadModeConvert = new List<string>(){"constant","edge","reflect"};
/// <summary>
/// Pads an input array with a constant or edge values of the array... note:: `Pad` is deprecated. Use `pad` instead... note:: Current implementation only supports 4D and 5D input arrays with padding applied   only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.This operation pads an input array with either a `constant_value` or edge valuesalong each axis of the input array. The amount of padding is specified by `pad_width`.`pad_width` is a tuple of integer padding widths for each axis of the format``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``where ``N`` is the number of dimensions of the array.For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many valuesto add before and after the elements of the array along dimension ``N``.The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,``after_2`` must be 0.Example::   x = [[[[  1.   2.   3.]          [  4.   5.   6.]]         [[  7.   8.   9.]          [ 10.  11.  12.]]]        [[[ 11.  12.  13.]          [ 14.  15.  16.]]         [[ 17.  18.  19.]          [ 20.  21.  22.]]]]   pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =         [[[[  1.   1.   2.   3.   3.]            [  1.   1.   2.   3.   3.]            [  4.   4.   5.   6.   6.]            [  4.   4.   5.   6.   6.]]           [[  7.   7.   8.   9.   9.]            [  7.   7.   8.   9.   9.]            [ 10.  10.  11.  12.  12.]            [ 10.  10.  11.  12.  12.]]]          [[[ 11.  11.  12.  13.  13.]            [ 11.  11.  12.  13.  13.]            [ 14.  14.  15.  16.  16.]            [ 14.  14.  15.  16.  16.]]           [[ 17.  17.  18.  19.  19.]            [ 17.  17.  18.  19.  19.]            [ 20.  20.  21.  22.  22.]            [ 20.  20.  21.  22.  22.]]]]   pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =         [[[[  0.   0.   0.   0.   0.]            [  0.   1.   2.   3.   0.]            [  0.   4.   5.   6.   0.]            [  0.   0.   0.   0.   0.]]           [[  0.   0.   0.   0.   0.]            [  0.   7.   8.   9.   0.]            [  0.  10.  11.  12.   0.]            [  0.   0.   0.   0.   0.]]]          [[[  0.   0.   0.   0.   0.]            [  0.  11.  12.  13.   0.]            [  0.  14.  15.  16.   0.]            [  0.   0.   0.   0.   0.]]           [[  0.   0.   0.   0.   0.]            [  0.  17.  18.  19.   0.]            [  0.  20.  21.  22.   0.]            [  0.   0.   0.   0.   0.]]]]Defined in G:\deeplearn\mxnet\src\operator\pad.cc:L765
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">An n-dimensional input array.</param>
/// <param name="mode">Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.</param>
/// <param name="pad_width">Widths of the padding regions applied to the edges of each axis. It is a tuple of integer padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but flattened.</param>
/// <param name="constant_value">The value used for padding when `mode` is "constant".</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pad(string symbol_name,
Symbol data,
PadMode mode,
Shape pad_width,
double constant_value=0)
{
return new Operator("Pad")
.SetParam("mode", Util.EnumToString<PadMode>(mode,PadModeConvert))
.SetParam("pad_width", pad_width)
.SetParam("constant_value", constant_value)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Pads an input array with a constant or edge values of the array... note:: `Pad` is deprecated. Use `pad` instead... note:: Current implementation only supports 4D and 5D input arrays with padding applied   only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.This operation pads an input array with either a `constant_value` or edge valuesalong each axis of the input array. The amount of padding is specified by `pad_width`.`pad_width` is a tuple of integer padding widths for each axis of the format``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``where ``N`` is the number of dimensions of the array.For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many valuesto add before and after the elements of the array along dimension ``N``.The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,``after_2`` must be 0.Example::   x = [[[[  1.   2.   3.]          [  4.   5.   6.]]         [[  7.   8.   9.]          [ 10.  11.  12.]]]        [[[ 11.  12.  13.]          [ 14.  15.  16.]]         [[ 17.  18.  19.]          [ 20.  21.  22.]]]]   pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =         [[[[  1.   1.   2.   3.   3.]            [  1.   1.   2.   3.   3.]            [  4.   4.   5.   6.   6.]            [  4.   4.   5.   6.   6.]]           [[  7.   7.   8.   9.   9.]            [  7.   7.   8.   9.   9.]            [ 10.  10.  11.  12.  12.]            [ 10.  10.  11.  12.  12.]]]          [[[ 11.  11.  12.  13.  13.]            [ 11.  11.  12.  13.  13.]            [ 14.  14.  15.  16.  16.]            [ 14.  14.  15.  16.  16.]]           [[ 17.  17.  18.  19.  19.]            [ 17.  17.  18.  19.  19.]            [ 20.  20.  21.  22.  22.]            [ 20.  20.  21.  22.  22.]]]]   pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =         [[[[  0.   0.   0.   0.   0.]            [  0.   1.   2.   3.   0.]            [  0.   4.   5.   6.   0.]            [  0.   0.   0.   0.   0.]]           [[  0.   0.   0.   0.   0.]            [  0.   7.   8.   9.   0.]            [  0.  10.  11.  12.   0.]            [  0.   0.   0.   0.   0.]]]          [[[  0.   0.   0.   0.   0.]            [  0.  11.  12.  13.   0.]            [  0.  14.  15.  16.   0.]            [  0.   0.   0.   0.   0.]]           [[  0.   0.   0.   0.   0.]            [  0.  17.  18.  19.   0.]            [  0.  20.  21.  22.   0.]            [  0.   0.   0.   0.   0.]]]]Defined in G:\deeplearn\mxnet\src\operator\pad.cc:L765
/// </summary>
/// <param name="data">An n-dimensional input array.</param>
/// <param name="mode">Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.</param>
/// <param name="pad_width">Widths of the padding regions applied to the edges of each axis. It is a tuple of integer padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but flattened.</param>
/// <param name="constant_value">The value used for padding when `mode` is "constant".</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pad(Symbol data,
PadMode mode,
Shape pad_width,
double constant_value=0)
{
return new Operator("Pad")
.SetParam("mode", Util.EnumToString<PadMode>(mode,PadModeConvert))
.SetParam("pad_width", pad_width)
.SetParam("constant_value", constant_value)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Splits an array along a particular axis into multiple sub-arrays... note:: ``SliceChannel`` is deprecated. Use ``split`` instead.**Note** that `num_outputs` should evenly divide the length of the axisalong which to split the array.Example::   x  = [[[ 1.]          [ 2.]]         [[ 3.]          [ 4.]]         [[ 5.]          [ 6.]]]   x.shape = (3, 2, 1)   y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)   y = [[[ 1.]]        [[ 3.]]        [[ 5.]]]       [[[ 2.]]        [[ 4.]]        [[ 6.]]]   y[0].shape = (3, 1, 1)   z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)   z = [[[ 1.]         [ 2.]]]       [[[ 3.]         [ 4.]]]       [[[ 5.]         [ 6.]]]   z[0].shape = (1, 2, 1)`squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.**Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 onlyalong the `axis` which it is split.Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.Example::   z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)   z = [[ 1.]        [ 2.]]       [[ 3.]        [ 4.]]       [[ 5.]        [ 6.]]   z[0].shape = (2 ,1 )Defined in G:\deeplearn\mxnet\src\operator\slice_channel.cc:L106
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="num_outputs">Number of splits. Note that this should evenly divide the length of the `axis`.</param>
/// <param name="axis">Axis along which to split.</param>
/// <param name="squeeze_axis">If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SliceChannel(string symbol_name,
Symbol data,
int num_outputs,
int axis=1,
bool squeeze_axis=false)
{
return new Operator("SliceChannel")
.SetParam("num_outputs", num_outputs)
.SetParam("axis", axis)
.SetParam("squeeze_axis", squeeze_axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Splits an array along a particular axis into multiple sub-arrays... note:: ``SliceChannel`` is deprecated. Use ``split`` instead.**Note** that `num_outputs` should evenly divide the length of the axisalong which to split the array.Example::   x  = [[[ 1.]          [ 2.]]         [[ 3.]          [ 4.]]         [[ 5.]          [ 6.]]]   x.shape = (3, 2, 1)   y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)   y = [[[ 1.]]        [[ 3.]]        [[ 5.]]]       [[[ 2.]]        [[ 4.]]        [[ 6.]]]   y[0].shape = (3, 1, 1)   z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)   z = [[[ 1.]         [ 2.]]]       [[[ 3.]         [ 4.]]]       [[[ 5.]         [ 6.]]]   z[0].shape = (1, 2, 1)`squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.**Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 onlyalong the `axis` which it is split.Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.Example::   z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)   z = [[ 1.]        [ 2.]]       [[ 3.]        [ 4.]]       [[ 5.]        [ 6.]]   z[0].shape = (2 ,1 )Defined in G:\deeplearn\mxnet\src\operator\slice_channel.cc:L106
/// </summary>
/// <param name="data">The input</param>
/// <param name="num_outputs">Number of splits. Note that this should evenly divide the length of the `axis`.</param>
/// <param name="axis">Axis along which to split.</param>
/// <param name="squeeze_axis">If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SliceChannel(Symbol data,
int num_outputs,
int axis=1,
bool squeeze_axis=false)
{
return new Operator("SliceChannel")
.SetParam("num_outputs", num_outputs)
.SetParam("axis", axis)
.SetParam("squeeze_axis", squeeze_axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Interchanges two axes of an array.Examples::  x = [[1, 2, 3]])  swapaxes(x, 0, 1) = [[ 1],                       [ 2],                       [ 3]]  x = [[[ 0, 1],        [ 2, 3]],       [[ 4, 5],        [ 6, 7]]]  // (2,2,2) array swapaxes(x, 0, 2) = [[[ 0, 4],                       [ 2, 6]],                      [[ 1, 5],                       [ 3, 7]]]Defined in G:\deeplearn\mxnet\src\operator\swapaxis.cc:L69
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
/// <param name="dim1">the first axis to be swapped.</param>
/// <param name="dim2">the second axis to be swapped.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SwapAxis(string symbol_name,
Symbol data,
int dim1=0,
int dim2=0)
{
return new Operator("SwapAxis")
.SetParam("dim1", dim1)
.SetParam("dim2", dim2)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Interchanges two axes of an array.Examples::  x = [[1, 2, 3]])  swapaxes(x, 0, 1) = [[ 1],                       [ 2],                       [ 3]]  x = [[[ 0, 1],        [ 2, 3]],       [[ 4, 5],        [ 6, 7]]]  // (2,2,2) array swapaxes(x, 0, 2) = [[[ 0, 4],                       [ 2, 6]],                      [[ 1, 5],                       [ 3, 7]]]Defined in G:\deeplearn\mxnet\src\operator\swapaxis.cc:L69
/// </summary>
/// <param name="data">Input array.</param>
/// <param name="dim1">the first axis to be swapped.</param>
/// <param name="dim2">the second axis to be swapped.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SwapAxis(Symbol data,
int dim1=0,
int dim2=0)
{
return new Operator("SwapAxis")
.SetParam("dim1", dim1)
.SetParam("dim2", dim2)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns indices of the maximum values along an axis.In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  // argmax along axis 0  argmax(x, axis=0) = [ 1.,  1.,  1.]  // argmax along axis 1  argmax(x, axis=1) = [ 2.,  2.]  // argmax along axis 1 keeping same dims as an input array  argmax(x, axis=1, keepdims=True) = [[ 2.],                                      [ 2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L51
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argmax(string symbol_name,
Symbol data,
int? axis=null,
bool keepdims=false)
{
return new Operator("argmax")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns indices of the maximum values along an axis.In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  // argmax along axis 0  argmax(x, axis=0) = [ 1.,  1.,  1.]  // argmax along axis 1  argmax(x, axis=1) = [ 2.,  2.]  // argmax along axis 1 keeping same dims as an input array  argmax(x, axis=1, keepdims=True) = [[ 2.],                                      [ 2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L51
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argmax(Symbol data,
int? axis=null,
bool keepdims=false)
{
return new Operator("argmax")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns indices of the minimum values along an axis.In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  // argmin along axis 0  argmin(x, axis=0) = [ 0.,  0.,  0.]  // argmin along axis 1  argmin(x, axis=1) = [ 0.,  0.]  // argmin along axis 1 keeping same dims as an input array  argmin(x, axis=1, keepdims=True) = [[ 0.],                                      [ 0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L76
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argmin(string symbol_name,
Symbol data,
int? axis=null,
bool keepdims=false)
{
return new Operator("argmin")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns indices of the minimum values along an axis.In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  // argmin along axis 0  argmin(x, axis=0) = [ 0.,  0.,  0.]  // argmin along axis 1  argmin(x, axis=1) = [ 0.,  0.]  // argmin along axis 1 keeping same dims as an input array  argmin(x, axis=1, keepdims=True) = [[ 0.],                                      [ 0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L76
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argmin(Symbol data,
int? axis=null,
bool keepdims=false)
{
return new Operator("argmin")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns argmax indices of each channel from the input array.The result will be an NDArray of shape (num_channel,).In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  argmax_channel(x) = [ 2.,  2.]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L96
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array</param>
 /// <returns>returns new symbol</returns>
public static Symbol ArgmaxChannel(string symbol_name,
Symbol data)
{
return new Operator("argmax_channel")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns argmax indices of each channel from the input array.The result will be an NDArray of shape (num_channel,).In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrenceare returned.Examples::  x = [[ 0.,  1.,  2.],       [ 3.,  4.,  5.]]  argmax_channel(x) = [ 2.,  2.]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L96
/// </summary>
/// <param name="data">The input array</param>
 /// <returns>returns new symbol</returns>
public static Symbol ArgmaxChannel(Symbol data)
{
return new Operator("argmax_channel")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Picks elements from an input array according to the input indices along the given axis.Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will bean output array of shape ``(i0,)`` with::  output[i] = input[i, indices[i]]By default, if any index mentioned is too large, it is replaced by the index that addressesthe last element along an axis (the `clip` mode).This function supports n-dimensional input and (n-1)-dimensional indices arrays.Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // picks elements with specified indices along axis 0  pick(x, y=[0,1], 0) = [ 1.,  4.]  // picks elements with specified indices along axis 1  pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]  y = [[ 1.],       [ 0.],       [ 2.]]  // picks elements with specified indices along axis 1 and dims are maintained  pick(x,y, 1, keepdims=True) = [[ 2.],                                 [ 3.],                                 [ 6.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L144
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array</param>
/// <param name="index">The index array</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pick(string symbol_name,
Symbol data,
Symbol index,
int? axis=null,
bool keepdims=false)
{
return new Operator("pick")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.SetInput("index", index)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Picks elements from an input array according to the input indices along the given axis.Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will bean output array of shape ``(i0,)`` with::  output[i] = input[i, indices[i]]By default, if any index mentioned is too large, it is replaced by the index that addressesthe last element along an axis (the `clip` mode).This function supports n-dimensional input and (n-1)-dimensional indices arrays.Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // picks elements with specified indices along axis 0  pick(x, y=[0,1], 0) = [ 1.,  4.]  // picks elements with specified indices along axis 1  pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]  y = [[ 1.],       [ 0.],       [ 2.]]  // picks elements with specified indices along axis 1 and dims are maintained  pick(x,y, 1, keepdims=True) = [[ 2.],                                 [ 3.],                                 [ 6.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L144
/// </summary>
/// <param name="data">The input array</param>
/// <param name="index">The index array</param>
/// <param name="axis">The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``</param>
/// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pick(Symbol data,
Symbol index,
int? axis=null,
bool keepdims=false)
{
return new Operator("pick")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("data", data)
.SetInput("index", index)
.CreateSymbol();
}
/// <summary>
/// Computes the sum of array elements over given axes... Note::  `sum` and `sum_axis` are equivalent.Example::  data = [[[1,2],[2,3],[1,3]],          [[1,4],[4,3],[5,2]],          [[7,1],[7,2],[7,3]]]  sum(data, axis=1)  [[  4.   8.]   [ 10.   9.]   [ 21.   6.]]  sum(data, axis=[1,2])  [ 12.  19.  27.]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L69
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sum(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("sum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the sum of array elements over given axes... Note::  `sum` and `sum_axis` are equivalent.Example::  data = [[[1,2],[2,3],[1,3]],          [[1,4],[4,3],[5,2]],          [[7,1],[7,2],[7,3]]]  sum(data, axis=1)  [[  4.   8.]   [ 10.   9.]   [ 21.   6.]]  sum(data, axis=[1,2])  [ 12.  19.  27.]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L69
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sum(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("sum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the mean of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L82
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Mean(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("mean")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the mean of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L82
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Mean(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("mean")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the product of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L95
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Prod(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("prod")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the product of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L95
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Prod(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("prod")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L110
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Nansum(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("nansum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L110
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Nansum(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("nansum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L125
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Nanprod(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("nanprod")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L125
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Nanprod(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("nanprod")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the max of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L139
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Max(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("max")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the max of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L139
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Max(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("max")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the min of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L153
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Min(string symbol_name,
Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("min")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the min of array elements over given axes.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L153
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.</param>
/// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
/// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Min(Symbol data,
Shape axis=null,
bool keepdims=false,
bool exclude=false)
{
return new Operator("min")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetParam("exclude", exclude)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Broadcasts the input array over particular axes.Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.Example::   // given x of shape (1,2,1)   x = [[[ 1.],         [ 2.]]]   // broadcast x on on axis 2   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],                                         [ 2.,  2.,  2.]]]   // broadcast x on on axes 0 and 2   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],                                                 [ 2.,  2.,  2.]],                                                [[ 1.,  1.,  1.],                                                 [ 2.,  2.,  2.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L186
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="axis">The axes to perform the broadcasting.</param>
/// <param name="size">Target sizes of the broadcasting axes.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastAxis(string symbol_name,
Symbol data,
Shape axis=null,
Shape size=null)
{
return new Operator("broadcast_axis")
.SetParam("axis", axis)
.SetParam("size", size)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Broadcasts the input array over particular axes.Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.Example::   // given x of shape (1,2,1)   x = [[[ 1.],         [ 2.]]]   // broadcast x on on axis 2   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],                                         [ 2.,  2.,  2.]]]   // broadcast x on on axes 0 and 2   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],                                                 [ 2.,  2.,  2.]],                                                [[ 1.,  1.,  1.],                                                 [ 2.,  2.,  2.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L186
/// </summary>
/// <param name="data">The input</param>
/// <param name="axis">The axes to perform the broadcasting.</param>
/// <param name="size">Target sizes of the broadcasting axes.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastAxis(Symbol data,
Shape axis=null,
Shape size=null)
{
return new Operator("broadcast_axis")
.SetParam("axis", axis)
.SetParam("size", size)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Broadcasts the input array to a new shape.Broadcasting is a mechanism that allows NDArrays to perform arithmetic operationswith arrays of different shapes efficiently without creating multiple copies of arrays.Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.For example::   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],                                           [ 1.,  2.,  3.]])The dimension which you do not want to change can also be kept as `0` which means copy the original value.So with `shape=(2,0)`, we will obtain the same result as in the above example.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L210
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
/// <param name="shape">The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastTo(string symbol_name,
Symbol data,
Shape shape=null)
{
return new Operator("broadcast_to")
.SetParam("shape", shape)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Broadcasts the input array to a new shape.Broadcasting is a mechanism that allows NDArrays to perform arithmetic operationswith arrays of different shapes efficiently without creating multiple copies of arrays.Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.For example::   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],                                           [ 1.,  2.,  3.]])The dimension which you do not want to change can also be kept as `0` which means copy the original value.So with `shape=(2,0)`, we will obtain the same result as in the above example.Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L210
/// </summary>
/// <param name="data">The input</param>
/// <param name="shape">The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastTo(Symbol data,
Shape shape=null)
{
return new Operator("broadcast_to")
.SetParam("shape", shape)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Flattens the input array and then computes the l2 norm.Examples::  x = [[1, 2],       [3, 4]]  norm(x) = [5.47722578]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L236
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Source input</param>
 /// <returns>returns new symbol</returns>
public static Symbol Norm(string symbol_name,
Symbol data)
{
return new Operator("norm")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Flattens the input array and then computes the l2 norm.Examples::  x = [[1, 2],       [3, 4]]  norm(x) = [5.47722578]Defined in G:\deeplearn\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L236
/// </summary>
/// <param name="data">Source input</param>
 /// <returns>returns new symbol</returns>
public static Symbol Norm(Symbol data)
{
return new Operator("norm")
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> CastStorageStypeConvert = new List<string>(){"csr","default","row_sparse"};
/// <summary>
/// Casts tensor storage type to the new type.When an NDArray with default storage type is cast to csr or row_sparse storage,the result is compact, which means:- for csr, zero values will not be retained- for row_sparse, row slices of all zeros will not be retainedThe storage type of ``cast_storage`` output depends on stype parameter:- cast_storage(csr, 'default') = default- cast_storage(row_sparse, 'default') = default- cast_storage(default, 'csr') = csr- cast_storage(default, 'row_sparse') = row_sparseExample::    dense = [[ 0.,  1.,  0.],             [ 2.,  0.,  3.],             [ 0.,  0.,  0.],             [ 0.,  0.,  0.]]    # cast to row_sparse storage type    rsp = cast_storage(default, 'row_sparse')    rsp.indices = [0, 1]    rsp.values = [[ 0.,  1.,  0.],                  [ 2.,  0.,  3.]]    # cast to csr storage type    csr = cast_storage(default, 'csr')    csr.indices = [1, 0, 2]    csr.values = [ 1.,  2.,  3.]    csr.indptr = [0, 1, 3, 3, 3]Defined in G:\deeplearn\mxnet\src\operator\tensor\cast_storage.cc:L69
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input.</param>
/// <param name="stype">Output storage type.</param>
 /// <returns>returns new symbol</returns>
public static Symbol CastStorage(string symbol_name,
Symbol data,
CastStorageStype stype)
{
return new Operator("cast_storage")
.SetParam("stype", Util.EnumToString<CastStorageStype>(stype,CastStorageStypeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Casts tensor storage type to the new type.When an NDArray with default storage type is cast to csr or row_sparse storage,the result is compact, which means:- for csr, zero values will not be retained- for row_sparse, row slices of all zeros will not be retainedThe storage type of ``cast_storage`` output depends on stype parameter:- cast_storage(csr, 'default') = default- cast_storage(row_sparse, 'default') = default- cast_storage(default, 'csr') = csr- cast_storage(default, 'row_sparse') = row_sparseExample::    dense = [[ 0.,  1.,  0.],             [ 2.,  0.,  3.],             [ 0.,  0.,  0.],             [ 0.,  0.,  0.]]    # cast to row_sparse storage type    rsp = cast_storage(default, 'row_sparse')    rsp.indices = [0, 1]    rsp.values = [[ 0.,  1.,  0.],                  [ 2.,  0.,  3.]]    # cast to csr storage type    csr = cast_storage(default, 'csr')    csr.indices = [1, 0, 2]    csr.values = [ 1.,  2.,  3.]    csr.indptr = [0, 1, 3, 3, 3]Defined in G:\deeplearn\mxnet\src\operator\tensor\cast_storage.cc:L69
/// </summary>
/// <param name="data">The input.</param>
/// <param name="stype">Output storage type.</param>
 /// <returns>returns new symbol</returns>
public static Symbol CastStorage(Symbol data,
CastStorageStype stype)
{
return new Operator("cast_storage")
.SetParam("stype", Util.EnumToString<CastStorageStype>(stype,CastStorageStypeConvert))
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y, depending on the elements from condition are true or false. x and y must have the same shape. If condition has the same shape as x, each element in the output array is from x if the corresponding element in the condition is true, and from y if false. If condition does not have the same shape as x, it must be a 1D array whose size is the same as x's first dimension size. Each row of the output array is from x's row if the corresponding element from condition is true, and from y's row if false.From:G:\deeplearn\mxnet\src\operator\tensor\control_flow_op.cc:39
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="condition">condition array</param>
/// <param name="x"></param>
/// <param name="y"></param>
 /// <returns>returns new symbol</returns>
public static Symbol Where(string symbol_name,
Symbol condition,
Symbol x,
Symbol y)
{
return new Operator("where")
.SetInput("condition", condition)
.SetInput("x", x)
.SetInput("y", y)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y, depending on the elements from condition are true or false. x and y must have the same shape. If condition has the same shape as x, each element in the output array is from x if the corresponding element in the condition is true, and from y if false. If condition does not have the same shape as x, it must be a 1D array whose size is the same as x's first dimension size. Each row of the output array is from x's row if the corresponding element from condition is true, and from y's row if false.From:G:\deeplearn\mxnet\src\operator\tensor\control_flow_op.cc:39
/// </summary>
/// <param name="condition">condition array</param>
/// <param name="x"></param>
/// <param name="y"></param>
 /// <returns>returns new symbol</returns>
public static Symbol Where(Symbol condition,
Symbol x,
Symbol y)
{
return new Operator("where")
.SetInput("condition", condition)
.SetInput("x", x)
.SetInput("y", y)
.CreateSymbol();
}
/// <summary>
/// Dot product of two arrays.``dot``'s behavior depends on the input array dimensions:- 1-D arrays: inner product of vectors- 2-D arrays: matrix multiplication- N-D arrays: a sum product over the last axis of the first input and the first  axis of the second input  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the  result array will have shape `(n,m,r,s)`. It is computed by::    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])  Example::    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))    dot(x,y)[0,0,1,1] = 0    sum(x[0,0,:]*y[:,1,1]) = 0The storage type of ``dot`` output depends on storage types of inputs and transpose options:- dot(csr, default) = default- dot(csr.T, default) = row_sparse- dot(csr, row_sparse) = default- otherwise, ``dot`` generates output with default storageDefined in G:\deeplearn\mxnet\src\operator\tensor\dot.cc:L61
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">The first input</param>
/// <param name="rhs">The second input</param>
/// <param name="transpose_a">If true then transpose the first input before dot.</param>
/// <param name="transpose_b">If true then transpose the second input before dot.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Dot(string symbol_name,
Symbol lhs,
Symbol rhs,
bool transpose_a=false,
bool transpose_b=false)
{
return new Operator("dot")
.SetParam("transpose_a", transpose_a)
.SetParam("transpose_b", transpose_b)
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Dot product of two arrays.``dot``'s behavior depends on the input array dimensions:- 1-D arrays: inner product of vectors- 2-D arrays: matrix multiplication- N-D arrays: a sum product over the last axis of the first input and the first  axis of the second input  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the  result array will have shape `(n,m,r,s)`. It is computed by::    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])  Example::    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))    dot(x,y)[0,0,1,1] = 0    sum(x[0,0,:]*y[:,1,1]) = 0The storage type of ``dot`` output depends on storage types of inputs and transpose options:- dot(csr, default) = default- dot(csr.T, default) = row_sparse- dot(csr, row_sparse) = default- otherwise, ``dot`` generates output with default storageDefined in G:\deeplearn\mxnet\src\operator\tensor\dot.cc:L61
/// </summary>
/// <param name="lhs">The first input</param>
/// <param name="rhs">The second input</param>
/// <param name="transpose_a">If true then transpose the first input before dot.</param>
/// <param name="transpose_b">If true then transpose the second input before dot.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Dot(Symbol lhs,
Symbol rhs,
bool transpose_a=false,
bool transpose_b=false)
{
return new Operator("dot")
.SetParam("transpose_a", transpose_a)
.SetParam("transpose_b", transpose_b)
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Batchwise dot product.``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,which is computed by::   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])Defined in G:\deeplearn\mxnet\src\operator\tensor\dot.cc:L109
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">The first input</param>
/// <param name="rhs">The second input</param>
/// <param name="transpose_a">If true then transpose the first input before dot.</param>
/// <param name="transpose_b">If true then transpose the second input before dot.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchDot(string symbol_name,
Symbol lhs,
Symbol rhs,
bool transpose_a=false,
bool transpose_b=false)
{
return new Operator("batch_dot")
.SetParam("transpose_a", transpose_a)
.SetParam("transpose_b", transpose_b)
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Batchwise dot product.``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,which is computed by::   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])Defined in G:\deeplearn\mxnet\src\operator\tensor\dot.cc:L109
/// </summary>
/// <param name="lhs">The first input</param>
/// <param name="rhs">The second input</param>
/// <param name="transpose_a">If true then transpose the first input before dot.</param>
/// <param name="transpose_b">If true then transpose the second input before dot.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchDot(Symbol lhs,
Symbol rhs,
bool transpose_a=false,
bool transpose_b=false)
{
return new Operator("batch_dot")
.SetParam("transpose_a", transpose_a)
.SetParam("transpose_b", transpose_b)
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise sum of the input arrays with broadcasting.`broadcast_plus` is an alias to the function `broadcast_add`.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_add(x, y) = [[ 1.,  1.,  1.],                          [ 2.,  2.,  2.]]   broadcast_plus(x, y) = [[ 1.,  1.,  1.],                           [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L50
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastAdd(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_add")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise sum of the input arrays with broadcasting.`broadcast_plus` is an alias to the function `broadcast_add`.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_add(x, y) = [[ 1.,  1.,  1.],                          [ 2.,  2.,  2.]]   broadcast_plus(x, y) = [[ 1.,  1.,  1.],                           [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L50
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastAdd(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_add")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise difference of the input arrays with broadcasting.`broadcast_minus` is an alias to the function `broadcast_sub`.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_sub(x, y) = [[ 1.,  1.,  1.],                          [ 0.,  0.,  0.]]   broadcast_minus(x, y) = [[ 1.,  1.,  1.],                            [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L89
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastSub(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_sub")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise difference of the input arrays with broadcasting.`broadcast_minus` is an alias to the function `broadcast_sub`.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_sub(x, y) = [[ 1.,  1.,  1.],                          [ 0.,  0.,  0.]]   broadcast_minus(x, y) = [[ 1.,  1.,  1.],                            [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L89
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastSub(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_sub")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise product of the input arrays with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_mul(x, y) = [[ 0.,  0.,  0.],                          [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L122
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMul(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise product of the input arrays with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_mul(x, y) = [[ 0.,  0.,  0.],                          [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L122
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMul(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise division of the input arrays with broadcasting.Example::   x = [[ 6.,  6.,  6.],        [ 6.,  6.,  6.]]   y = [[ 2.],        [ 3.]]   broadcast_div(x, y) = [[ 3.,  3.,  3.],                          [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L156
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastDiv(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise division of the input arrays with broadcasting.Example::   x = [[ 6.,  6.,  6.],        [ 6.,  6.,  6.]]   y = [[ 2.],        [ 3.]]   broadcast_div(x, y) = [[ 3.,  3.,  3.],                          [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L156
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastDiv(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise modulo of the input arrays with broadcasting.Example::   x = [[ 8.,  8.,  8.],        [ 8.,  8.,  8.]]   y = [[ 2.],        [ 3.]]   broadcast_mod(x, y) = [[ 0.,  0.,  0.],                          [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L189
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMod(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_mod")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise modulo of the input arrays with broadcasting.Example::   x = [[ 8.,  8.,  8.],        [ 8.,  8.,  8.]]   y = [[ 2.],        [ 3.]]   broadcast_mod(x, y) = [[ 0.,  0.,  0.],                          [ 2.,  2.,  2.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L189
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMod(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_mod")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns result of first array elements raised to powers from second array, element-wise with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_power(x, y) = [[ 2.,  2.,  2.],                            [ 4.,  4.,  4.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L44
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastPower(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_power")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns result of first array elements raised to powers from second array, element-wise with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_power(x, y) = [[ 2.,  2.,  2.],                            [ 4.,  4.,  4.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L44
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastPower(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_power")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise maximum of the input arrays with broadcasting.This function compares two input arrays and returns a new array having the element-wise maxima.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_maximum(x, y) = [[ 1.,  1.,  1.],                              [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L79
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMaximum(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_maximum")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise maximum of the input arrays with broadcasting.This function compares two input arrays and returns a new array having the element-wise maxima.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_maximum(x, y) = [[ 1.,  1.,  1.],                              [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L79
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMaximum(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_maximum")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise minimum of the input arrays with broadcasting.This function compares two input arrays and returns a new array having the element-wise minima.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_maximum(x, y) = [[ 0.,  0.,  0.],                              [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L114
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMinimum(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_minimum")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise minimum of the input arrays with broadcasting.This function compares two input arrays and returns a new array having the element-wise minima.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_maximum(x, y) = [[ 0.,  0.,  0.],                              [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L114
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastMinimum(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_minimum")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
///  Returns the hypotenuse of a right angled triangle, given its "legs"with broadcasting.It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.Example::   x = [[ 3.,  3.,  3.]]   y = [[ 4.],        [ 4.]]   broadcast_hypot(x, y) = [[ 5.,  5.,  5.],                            [ 5.,  5.,  5.]]   z = [[ 0.],        [ 4.]]   broadcast_hypot(x, z) = [[ 3.,  3.,  3.],                            [ 5.,  5.,  5.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L155
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastHypot(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_hypot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
///  Returns the hypotenuse of a right angled triangle, given its "legs"with broadcasting.It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.Example::   x = [[ 3.,  3.,  3.]]   y = [[ 4.],        [ 4.]]   broadcast_hypot(x, y) = [[ 5.,  5.,  5.],                            [ 5.,  5.,  5.]]   z = [[ 0.],        [ 4.]]   broadcast_hypot(x, z) = [[ 3.,  3.,  3.],                            [ 5.,  5.,  5.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L155
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastHypot(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_hypot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_equal(x, y) = [[ 0.,  0.,  0.],                            [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L45
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastEqual(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_equal(x, y) = [[ 0.,  0.,  0.],                            [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L45
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastEqual(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],                                [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L63
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastNotEqual(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_not_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],                                [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L63
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastNotEqual(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_not_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_greater(x, y) = [[ 1.,  1.,  1.],                              [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L81
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastGreater(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_greater")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_greater(x, y) = [[ 1.,  1.,  1.],                              [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L81
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastGreater(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_greater")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],                                    [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L99
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastGreaterEqual(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_greater_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],                                    [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L99
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastGreaterEqual(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_greater_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_lesser(x, y) = [[ 0.,  0.,  0.],                             [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L117
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastLesser(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_lesser")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_lesser(x, y) = [[ 0.,  0.,  0.],                             [ 0.,  0.,  0.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L117
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastLesser(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_lesser")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],                                   [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L135
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastLesserEqual(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_lesser_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.Example::   x = [[ 1.,  1.,  1.],        [ 1.,  1.,  1.]]   y = [[ 0.],        [ 1.]]   broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],                                   [ 1.,  1.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L135
/// </summary>
/// <param name="lhs">First input to the function</param>
/// <param name="rhs">Second input to the function</param>
 /// <returns>returns new symbol</returns>
public static Symbol BroadcastLesserEqual(Symbol lhs,
Symbol rhs)
{
return new Operator("broadcast_lesser_equal")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Adds arguments element-wise.The storage type of ``elemwise_add`` output depends on storage types of inputs   - elemwise_add(row_sparse, row_sparse) = row_sparse   - otherwise, ``elemwise_add`` generates output with default storage
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseAdd(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_add")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Adds arguments element-wise.The storage type of ``elemwise_add`` output depends on storage types of inputs   - elemwise_add(row_sparse, row_sparse) = row_sparse   - otherwise, ``elemwise_add`` generates output with default storage
/// </summary>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseAdd(Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_add")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Subtracts arguments element-wise.The storage type of ``elemwise_sub`` output depends on storage types of inputs   - elemwise_sub(row_sparse, row_sparse) = row_sparse   - otherwise, ``elemwise_add`` generates output with default storage
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseSub(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_sub")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Subtracts arguments element-wise.The storage type of ``elemwise_sub`` output depends on storage types of inputs   - elemwise_sub(row_sparse, row_sparse) = row_sparse   - otherwise, ``elemwise_add`` generates output with default storage
/// </summary>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseSub(Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_sub")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Multiplies arguments element-wise.The storage type of ``elemwise_mul`` output depends on storage types of inputs   - elemwise_mul(default, default) = default   - elemwise_mul(row_sparse, row_sparse) = row_sparse   - elemwise_mul(default, row_sparse) = row_sparse   - elemwise_mul(row_sparse, default) = row_sparse   - otherwise, ``elemwise_mul`` generates output with default storage
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseMul(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Multiplies arguments element-wise.The storage type of ``elemwise_mul`` output depends on storage types of inputs   - elemwise_mul(default, default) = default   - elemwise_mul(row_sparse, row_sparse) = row_sparse   - elemwise_mul(default, row_sparse) = row_sparse   - elemwise_mul(row_sparse, default) = row_sparse   - otherwise, ``elemwise_mul`` generates output with default storage
/// </summary>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseMul(Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Divides arguments element-wise.The storage type of ``elemwise_dev`` output is always dense
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseDiv(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Divides arguments element-wise.The storage type of ``elemwise_dev`` output is always dense
/// </summary>
/// <param name="lhs">first input</param>
/// <param name="rhs">second input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ElemwiseDiv(Symbol lhs,
Symbol rhs)
{
return new Operator("elemwise_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Calculate Smooth L1 Loss(lhs, scalar) by summing.. math::    f(x) =    \begin{cases}    (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\    |x|-0.5/\sigma^2,& \text{otherwise}    \end{cases}where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.Example::  smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_scalar_op_extended.cc:L102
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">source input</param>
/// <param name="scalar">scalar input</param>
 /// <returns>returns new symbol</returns>
public static Symbol SmoothL1(string symbol_name,
Symbol data,
float scalar)
{
return new Operator("smooth_l1")
.SetParam("scalar", scalar)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate Smooth L1 Loss(lhs, scalar) by summing.. math::    f(x) =    \begin{cases}    (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\    |x|-0.5/\sigma^2,& \text{otherwise}    \end{cases}where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.Example::  smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_binary_scalar_op_extended.cc:L102
/// </summary>
/// <param name="data">source input</param>
/// <param name="scalar">scalar input</param>
 /// <returns>returns new symbol</returns>
public static Symbol SmoothL1(Symbol data,
float scalar)
{
return new Operator("smooth_l1")
.SetParam("scalar", scalar)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Adds all input arguments element-wise... math::   add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n``add_n`` is potentially more efficient than calling ``add`` by `n` times.The storage type of ``add_n`` output depends on storage types of inputs- add_n(row_sparse, row_sparse, ..) = row_sparse- otherwise, ``add_n`` generates output with default storageDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_sum.cc:L122
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="args">Positional input arguments</param>
 /// <returns>returns new symbol</returns>
public static Symbol AddN(string symbol_name,
Symbol[] args)
{
return new Operator("add_n")
.AddInput(args)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Adds all input arguments element-wise... math::   add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n``add_n`` is potentially more efficient than calling ``add`` by `n` times.The storage type of ``add_n`` output depends on storage types of inputs- add_n(row_sparse, row_sparse, ..) = row_sparse- otherwise, ``add_n`` generates output with default storageDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_sum.cc:L122
/// </summary>
/// <param name="args">Positional input arguments</param>
 /// <returns>returns new symbol</returns>
public static Symbol AddN(Symbol[] args)
{
return new Operator("add_n")
.AddInput(args)
.CreateSymbol();
}
/// <summary>
/// Computes rectified linear... math::   max(features, 0)The storage type of ``relu`` output depends upon the input storage type:   - relu(default) = default   - relu(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L44
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Relu(string symbol_name,
Symbol data)
{
return new Operator("relu")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes rectified linear... math::   max(features, 0)The storage type of ``relu`` output depends upon the input storage type:   - relu(default) = default   - relu(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L44
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Relu(Symbol data)
{
return new Operator("relu")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes sigmoid of x element-wise... math::   y = 1 / (1 + exp(-x))The storage type of ``sigmoid`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L64
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sigmoid(string symbol_name,
Symbol data)
{
return new Operator("sigmoid")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes sigmoid of x element-wise... math::   y = 1 / (1 + exp(-x))The storage type of ``sigmoid`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L64
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sigmoid(Symbol data)
{
return new Operator("sigmoid")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Stops gradient computation.Stops the accumulated gradient of the inputs from flowing through this operatorin the backward direction. In other words, this operator prevents the contributionof its inputs to be taken into account for computing gradients.Example::  v1 = [1, 2]  v2 = [0, 1]  a = Variable('a')  b = Variable('b')  b_stop_grad = stop_gradient(3 * b)  loss = MakeLoss(b_stop_grad + a)  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))  executor.forward(is_train=True, a=v1, b=v2)  executor.outputs  [ 1.  5.]  executor.backward()  executor.grad_arrays  [ 0.  0.]  [ 1.  1.]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L128
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BlockGrad(string symbol_name,
Symbol data)
{
return new Operator("BlockGrad")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Stops gradient computation.Stops the accumulated gradient of the inputs from flowing through this operatorin the backward direction. In other words, this operator prevents the contributionof its inputs to be taken into account for computing gradients.Example::  v1 = [1, 2]  v2 = [0, 1]  a = Variable('a')  b = Variable('b')  b_stop_grad = stop_gradient(3 * b)  loss = MakeLoss(b_stop_grad + a)  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))  executor.forward(is_train=True, a=v1, b=v2)  executor.outputs  [ 1.  5.]  executor.backward()  executor.grad_arrays  [ 0.  0.]  [ 1.  1.]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L128
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol BlockGrad(Symbol data)
{
return new Operator("BlockGrad")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Stops gradient computation... note:: ``make_loss`` is deprecated, use ``MakeLoss``.The storage type of ``make_loss`` output depends upon the input storage type:   - make_loss(default) = default   - make_loss(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L148
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol MakeLoss(string symbol_name,
Symbol data)
{
return new Operator("make_loss")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Stops gradient computation... note:: ``make_loss`` is deprecated, use ``MakeLoss``.The storage type of ``make_loss`` output depends upon the input storage type:   - make_loss(default) = default   - make_loss(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L148
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol MakeLoss(Symbol data)
{
return new Operator("make_loss")
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> CastDtypeConvert = new List<string>(){"float16","float32","float64","int32","uint8"};
/// <summary>
/// Casts all elements of the input to a new type... note:: ``Cast`` is deprecated. Use ``cast`` instead.Example::   cast([0.9, 1.3], dtype='int32') = [0, 1]   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L218
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input.</param>
/// <param name="dtype">Output data type.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cast(string symbol_name,
Symbol data,
CastDtype dtype)
{
return new Operator("Cast")
.SetParam("dtype", Util.EnumToString<CastDtype>(dtype,CastDtypeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Casts all elements of the input to a new type... note:: ``Cast`` is deprecated. Use ``cast`` instead.Example::   cast([0.9, 1.3], dtype='int32') = [0, 1]   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L218
/// </summary>
/// <param name="data">The input.</param>
/// <param name="dtype">Output data type.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cast(Symbol data,
CastDtype dtype)
{
return new Operator("Cast")
.SetParam("dtype", Util.EnumToString<CastDtype>(dtype,CastDtypeConvert))
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Numerical negative of the argument, element-wise.The storage type of ``negative`` output depends upon the input storage type:   - negative(default) = default   - negative(row_sparse) = row_sparse   - negative(csr) = csr
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Negative(string symbol_name,
Symbol data)
{
return new Operator("negative")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Numerical negative of the argument, element-wise.The storage type of ``negative`` output depends upon the input storage type:   - negative(default) = default   - negative(row_sparse) = row_sparse   - negative(csr) = csr
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Negative(Symbol data)
{
return new Operator("negative")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the reciprocal of the argument, element-wise.Calculates 1/x.Example::    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L271
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reciprocal(string symbol_name,
Symbol data)
{
return new Operator("reciprocal")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the reciprocal of the argument, element-wise.Calculates 1/x.Example::    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L271
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reciprocal(Symbol data)
{
return new Operator("reciprocal")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise absolute value of the input.Example::   abs([-2, 0, 3]) = [2, 0, 3]The storage type of ``abs`` output depends upon the input storage type:   - abs(default) = default   - abs(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L293
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Abs(string symbol_name,
Symbol data)
{
return new Operator("abs")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise absolute value of the input.Example::   abs([-2, 0, 3]) = [2, 0, 3]The storage type of ``abs`` output depends upon the input storage type:   - abs(default) = default   - abs(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L293
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Abs(Symbol data)
{
return new Operator("abs")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise sign of the input.Example::   sign([-2, 0, 3]) = [-1, 0, 1]The storage type of ``sign`` output depends upon the input storage type:   - sign(default) = default   - sign(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L312
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sign(string symbol_name,
Symbol data)
{
return new Operator("sign")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise sign of the input.Example::   sign([-2, 0, 3]) = [-1, 0, 1]The storage type of ``sign`` output depends upon the input storage type:   - sign(default) = default   - sign(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L312
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sign(Symbol data)
{
return new Operator("sign")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise rounded value to the nearest integer of the input.Example::   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]The storage type of ``round`` output depends upon the input storage type:  - round(default) = default  - round(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L331
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Round(string symbol_name,
Symbol data)
{
return new Operator("round")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise rounded value to the nearest integer of the input.Example::   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]The storage type of ``round`` output depends upon the input storage type:  - round(default) = default  - round(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L331
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Round(Symbol data)
{
return new Operator("round")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise rounded value to the nearest integer of the input... note::   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.Example::   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]The storage type of ``rint`` output depends upon the input storage type:   - rint(default) = default   - rint(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L352
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rint(string symbol_name,
Symbol data)
{
return new Operator("rint")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise rounded value to the nearest integer of the input... note::   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.Example::   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]The storage type of ``rint`` output depends upon the input storage type:   - rint(default) = default   - rint(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L352
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rint(Symbol data)
{
return new Operator("rint")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise ceiling of the input.The ceil of the scalar x is the smallest integer i, such that i >= x.Example::   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]The storage type of ``ceil`` output depends upon the input storage type:   - ceil(default) = default   - ceil(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L370
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Ceil(string symbol_name,
Symbol data)
{
return new Operator("ceil")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise ceiling of the input.The ceil of the scalar x is the smallest integer i, such that i >= x.Example::   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]The storage type of ``ceil`` output depends upon the input storage type:   - ceil(default) = default   - ceil(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L370
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Ceil(Symbol data)
{
return new Operator("ceil")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise floor of the input.The floor of the scalar x is the largest integer i, such that i <= x.Example::   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]The storage type of ``floor`` output depends upon the input storage type:   - floor(default) = default   - floor(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L388
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Floor(string symbol_name,
Symbol data)
{
return new Operator("floor")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise floor of the input.The floor of the scalar x is the largest integer i, such that i <= x.Example::   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]The storage type of ``floor`` output depends upon the input storage type:   - floor(default) = default   - floor(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L388
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Floor(Symbol data)
{
return new Operator("floor")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Return the element-wise truncated value of the input.The truncated value of the scalar x is the nearest integer i which is closer tozero than x is. In short, the fractional part of the signed number x is discarded.Example::   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]The storage type of ``trunc`` output depends upon the input storage type:   - trunc(default) = default   - trunc(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L407
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Trunc(string symbol_name,
Symbol data)
{
return new Operator("trunc")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Return the element-wise truncated value of the input.The truncated value of the scalar x is the nearest integer i which is closer tozero than x is. In short, the fractional part of the signed number x is discarded.Example::   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]The storage type of ``trunc`` output depends upon the input storage type:   - trunc(default) = default   - trunc(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L407
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Trunc(Symbol data)
{
return new Operator("trunc")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise rounded value to the nearest \integer towards zero of the input.Example::   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]The storage type of ``fix`` output depends upon the input storage type:   - fix(default) = default   - fix(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L424
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Fix(string symbol_name,
Symbol data)
{
return new Operator("fix")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise rounded value to the nearest \integer towards zero of the input.Example::   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]The storage type of ``fix`` output depends upon the input storage type:   - fix(default) = default   - fix(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L424
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Fix(Symbol data)
{
return new Operator("fix")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise squared value of the input... math::   square(x) = x^2Example::   square([2, 3, 4]) = [4, 9, 16]The storage type of ``square`` output depends upon the input storage type:   - square(default) = default   - square(row_sparse) = row_sparse   - square(csr) = csrDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L444
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Square(string symbol_name,
Symbol data)
{
return new Operator("square")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise squared value of the input... math::   square(x) = x^2Example::   square([2, 3, 4]) = [4, 9, 16]The storage type of ``square`` output depends upon the input storage type:   - square(default) = default   - square(row_sparse) = row_sparse   - square(csr) = csrDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L444
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Square(Symbol data)
{
return new Operator("square")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise square-root value of the input... math::   \textrm{sqrt}(x) = \sqrt{x}Example::   sqrt([4, 9, 16]) = [2, 3, 4]The storage type of ``sqrt`` output depends upon the input storage type:   - sqrt(default) = default   - sqrt(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L467
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sqrt(string symbol_name,
Symbol data)
{
return new Operator("sqrt")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise square-root value of the input... math::   \textrm{sqrt}(x) = \sqrt{x}Example::   sqrt([4, 9, 16]) = [2, 3, 4]The storage type of ``sqrt`` output depends upon the input storage type:   - sqrt(default) = default   - sqrt(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L467
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sqrt(Symbol data)
{
return new Operator("sqrt")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise inverse square-root value of the input... math::   rsqrt(x) = 1/\sqrt{x}Example::   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]The storage type of ``rsqrt`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L487
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rsqrt(string symbol_name,
Symbol data)
{
return new Operator("rsqrt")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise inverse square-root value of the input... math::   rsqrt(x) = 1/\sqrt{x}Example::   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]The storage type of ``rsqrt`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L487
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rsqrt(Symbol data)
{
return new Operator("rsqrt")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise cube-root value of the input... math::   cbrt(x) = \sqrt[3]{x}Example::   cbrt([1, 8, -125]) = [1, 2, -5]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L504
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cbrt(string symbol_name,
Symbol data)
{
return new Operator("cbrt")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise cube-root value of the input... math::   cbrt(x) = \sqrt[3]{x}Example::   cbrt([1, 8, -125]) = [1, 2, -5]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L504
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cbrt(Symbol data)
{
return new Operator("cbrt")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise inverse cube-root value of the input... math::   rcbrt(x) = 1/\sqrt[3]{x}Example::   rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L521
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rcbrt(string symbol_name,
Symbol data)
{
return new Operator("rcbrt")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise inverse cube-root value of the input... math::   rcbrt(x) = 1/\sqrt[3]{x}Example::   rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]Defined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L521
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Rcbrt(Symbol data)
{
return new Operator("rcbrt")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise exponential value of the input... math::   exp(x) = e^x \approx 2.718^xExample::   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]The storage type of ``exp`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L543
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Exp(string symbol_name,
Symbol data)
{
return new Operator("exp")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise exponential value of the input... math::   exp(x) = e^x \approx 2.718^xExample::   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]The storage type of ``exp`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L543
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Exp(Symbol data)
{
return new Operator("exp")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise Natural logarithmic value of the input.The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``The storage type of ``log`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L555
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log(string symbol_name,
Symbol data)
{
return new Operator("log")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise Natural logarithmic value of the input.The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``The storage type of ``log`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L555
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log(Symbol data)
{
return new Operator("log")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise Base-10 logarithmic value of the input.``10**log10(x) = x``The storage type of ``log10`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L567
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log10(string symbol_name,
Symbol data)
{
return new Operator("log10")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise Base-10 logarithmic value of the input.``10**log10(x) = x``The storage type of ``log10`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L567
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log10(Symbol data)
{
return new Operator("log10")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise Base-2 logarithmic value of the input.``2**log2(x) = x``The storage type of ``log2`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L579
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log2(string symbol_name,
Symbol data)
{
return new Operator("log2")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise Base-2 logarithmic value of the input.``2**log2(x) = x``The storage type of ``log2`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L579
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log2(Symbol data)
{
return new Operator("log2")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the element-wise sine of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]The storage type of ``sin`` output depends upon the input storage type:   - sin(default) = default   - sin(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L599
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sin(string symbol_name,
Symbol data)
{
return new Operator("sin")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the element-wise sine of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]The storage type of ``sin`` output depends upon the input storage type:   - sin(default) = default   - sin(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L599
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sin(Symbol data)
{
return new Operator("sin")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise ``log(1 + x)`` value of the input.This function is more accurate than ``log(1 + x)``  for small ``x`` so that:math:`1+x\approx 1`The storage type of ``log1p`` output depends upon the input storage type:   - log1p(default) = default   - log1p(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L617
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log1P(string symbol_name,
Symbol data)
{
return new Operator("log1p")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise ``log(1 + x)`` value of the input.This function is more accurate than ``log(1 + x)``  for small ``x`` so that:math:`1+x\approx 1`The storage type of ``log1p`` output depends upon the input storage type:   - log1p(default) = default   - log1p(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L617
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Log1P(Symbol data)
{
return new Operator("log1p")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns ``exp(x) - 1`` computed element-wise on the input.This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.The storage type of ``expm1`` output depends upon the input storage type:   - expm1(default) = default   - expm1(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L635
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Expm1(string symbol_name,
Symbol data)
{
return new Operator("expm1")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns ``exp(x) - 1`` computed element-wise on the input.This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.The storage type of ``expm1`` output depends upon the input storage type:   - expm1(default) = default   - expm1(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L635
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Expm1(Symbol data)
{
return new Operator("expm1")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the element-wise cosine of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]The storage type of ``cos`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L652
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cos(string symbol_name,
Symbol data)
{
return new Operator("cos")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the element-wise cosine of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]The storage type of ``cos`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L652
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cos(Symbol data)
{
return new Operator("cos")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes the element-wise tangent of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]The storage type of ``tan`` output depends upon the input storage type:   - tan(default) = default   - tan(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L672
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tan(string symbol_name,
Symbol data)
{
return new Operator("tan")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the element-wise tangent of the input array.The input should be in radians (:math:`2\pi` rad equals 360 degrees)... math::   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]The storage type of ``tan`` output depends upon the input storage type:   - tan(default) = default   - tan(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L672
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tan(Symbol data)
{
return new Operator("tan")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise inverse sine of the input array.The input should be in the range `[-1, 1]`.The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`]... math::   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]The storage type of ``arcsin`` output depends upon the input storage type:   - arcsin(default) = default   - arcsin(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L693
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arcsin(string symbol_name,
Symbol data)
{
return new Operator("arcsin")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise inverse sine of the input array.The input should be in the range `[-1, 1]`.The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`]... math::   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]The storage type of ``arcsin`` output depends upon the input storage type:   - arcsin(default) = default   - arcsin(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L693
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arcsin(Symbol data)
{
return new Operator("arcsin")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise inverse cosine of the input array.The input should be in range `[-1, 1]`.The output is in the closed interval :math:`[0, \pi]`.. math::   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]The storage type of ``arccos`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L712
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arccos(string symbol_name,
Symbol data)
{
return new Operator("arccos")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise inverse cosine of the input array.The input should be in range `[-1, 1]`.The output is in the closed interval :math:`[0, \pi]`.. math::   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]The storage type of ``arccos`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L712
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arccos(Symbol data)
{
return new Operator("arccos")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise inverse tangent of the input array.The output is in the closed interval :math:`[-\pi/2, \pi/2]`.. math::   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]The storage type of ``arctan`` output depends upon the input storage type:   - arctan(default) = default   - arctan(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L733
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arctan(string symbol_name,
Symbol data)
{
return new Operator("arctan")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise inverse tangent of the input array.The output is in the closed interval :math:`[-\pi/2, \pi/2]`.. math::   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]The storage type of ``arctan`` output depends upon the input storage type:   - arctan(default) = default   - arctan(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L733
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arctan(Symbol data)
{
return new Operator("arctan")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Converts each element of the input array from radians to degrees... math::   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]The storage type of ``degrees`` output depends upon the input storage type:   - degrees(default) = default   - degrees(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L752
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Degrees(string symbol_name,
Symbol data)
{
return new Operator("degrees")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Converts each element of the input array from radians to degrees... math::   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]The storage type of ``degrees`` output depends upon the input storage type:   - degrees(default) = default   - degrees(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L752
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Degrees(Symbol data)
{
return new Operator("degrees")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Converts each element of the input array from degrees to radians... math::   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]The storage type of ``radians`` output depends upon the input storage type:   - radians(default) = default   - radians(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L771
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Radians(string symbol_name,
Symbol data)
{
return new Operator("radians")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Converts each element of the input array from degrees to radians... math::   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]The storage type of ``radians`` output depends upon the input storage type:   - radians(default) = default   - radians(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L771
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Radians(Symbol data)
{
return new Operator("radians")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the hyperbolic sine of the input array, computed element-wise... math::   sinh(x) = 0.5\times(exp(x) - exp(-x))The storage type of ``sinh`` output depends upon the input storage type:   - sinh(default) = default   - sinh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L790
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sinh(string symbol_name,
Symbol data)
{
return new Operator("sinh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the hyperbolic sine of the input array, computed element-wise... math::   sinh(x) = 0.5\times(exp(x) - exp(-x))The storage type of ``sinh`` output depends upon the input storage type:   - sinh(default) = default   - sinh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L790
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sinh(Symbol data)
{
return new Operator("sinh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the hyperbolic cosine  of the input array, computed element-wise... math::   cosh(x) = 0.5\times(exp(x) + exp(-x))The storage type of ``cosh`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L805
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cosh(string symbol_name,
Symbol data)
{
return new Operator("cosh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the hyperbolic cosine  of the input array, computed element-wise... math::   cosh(x) = 0.5\times(exp(x) + exp(-x))The storage type of ``cosh`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L805
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Cosh(Symbol data)
{
return new Operator("cosh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the hyperbolic tangent of the input array, computed element-wise... math::   tanh(x) = sinh(x) / cosh(x)The storage type of ``tanh`` output depends upon the input storage type:   - tanh(default) = default   - tanh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L823
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tanh(string symbol_name,
Symbol data)
{
return new Operator("tanh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the hyperbolic tangent of the input array, computed element-wise... math::   tanh(x) = sinh(x) / cosh(x)The storage type of ``tanh`` output depends upon the input storage type:   - tanh(default) = default   - tanh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L823
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tanh(Symbol data)
{
return new Operator("tanh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the element-wise inverse hyperbolic sine of the input array, \computed element-wise.The storage type of ``arcsinh`` output depends upon the input storage type:   - arcsinh(default) = default   - arcsinh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L839
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arcsinh(string symbol_name,
Symbol data)
{
return new Operator("arcsinh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the element-wise inverse hyperbolic sine of the input array, \computed element-wise.The storage type of ``arcsinh`` output depends upon the input storage type:   - arcsinh(default) = default   - arcsinh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L839
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arcsinh(Symbol data)
{
return new Operator("arcsinh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the element-wise inverse hyperbolic cosine of the input array, \computed element-wise.The storage type of ``arccosh`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L853
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arccosh(string symbol_name,
Symbol data)
{
return new Operator("arccosh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the element-wise inverse hyperbolic cosine of the input array, \computed element-wise.The storage type of ``arccosh`` output is always denseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L853
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arccosh(Symbol data)
{
return new Operator("arccosh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the element-wise inverse hyperbolic tangent of the input array, \computed element-wise.The storage type of ``arctanh`` output depends upon the input storage type:   - arctanh(default) = default   - arctanh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L870
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arctanh(string symbol_name,
Symbol data)
{
return new Operator("arctanh")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the element-wise inverse hyperbolic tangent of the input array, \computed element-wise.The storage type of ``arctanh`` output depends upon the input storage type:   - arctanh(default) = default   - arctanh(row_sparse) = row_sparseDefined in G:\deeplearn\mxnet\src\operator\tensor\elemwise_unary_op.cc:L870
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Arctanh(Symbol data)
{
return new Operator("arctanh")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the gamma function (extension of the factorial function \to the reals), computed element-wise on the input array.The storage type of ``gamma`` output is always dense
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Gamma(string symbol_name,
Symbol data)
{
return new Operator("gamma")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the gamma function (extension of the factorial function \to the reals), computed element-wise on the input array.The storage type of ``gamma`` output is always dense
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Gamma(Symbol data)
{
return new Operator("gamma")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns element-wise log of the absolute value of the gamma function \of the input.The storage type of ``gammaln`` output is always dense
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Gammaln(string symbol_name,
Symbol data)
{
return new Operator("gammaln")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns element-wise log of the absolute value of the gamma function \of the input.The storage type of ``gammaln`` output is always dense
/// </summary>
/// <param name="data">The input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Gammaln(Symbol data)
{
return new Operator("gammaln")
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> EmbeddingDtypeConvert = new List<string>(){"float16","float32","float64","int32","uint8"};
/// <summary>
/// Maps integer indices to vector representations (embeddings).This operator maps words to real-valued vectors in a high-dimensional space,called word embeddings. These embeddings can capture semantic and syntactic properties of the words.For example, it has been noted that in the learned embedding spaces, similar words tendto be close to each other and dissimilar words far apart.For an input array of shape (d1, ..., dK),the shape of an output array is (d1, ..., dK, output_dim).All the input values should be integers in the range [0, input_dim).If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be(ip0, op0).By default, if any index mentioned is too large, it is replaced by the index that addressesthe last vector in an embedding matrix.Examples::  input_dim = 4  output_dim = 5  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)  y = [[  0.,   1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.,   9.],       [ 10.,  11.,  12.,  13.,  14.],       [ 15.,  16.,  17.,  18.,  19.]]  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]  x = [[ 1.,  3.],       [ 0.,  2.]]  // Mapped input x to its vector representation y.  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],                            [ 15.,  16.,  17.,  18.,  19.]],                           [[  0.,   1.,   2.,   3.,   4.],                            [ 10.,  11.,  12.,  13.,  14.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L74
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array to the embedding operator.</param>
/// <param name="input_dim">Vocabulary size of the input indices.</param>
/// <param name="output_dim">Dimension of the embedding vectors.</param>
/// <param name="weight">The embedding weight matrix.</param>
/// <param name="dtype">Data type of weight.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Embedding(string symbol_name,
Symbol data,
int input_dim,
int output_dim,
Symbol weight=null,
EmbeddingDtype dtype=EmbeddingDtype.Float32)
{
return new Operator("Embedding")
.SetParam("input_dim", input_dim)
.SetParam("output_dim", output_dim)
.SetParam("dtype", Util.EnumToString<EmbeddingDtype>(dtype,EmbeddingDtypeConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Maps integer indices to vector representations (embeddings).This operator maps words to real-valued vectors in a high-dimensional space,called word embeddings. These embeddings can capture semantic and syntactic properties of the words.For example, it has been noted that in the learned embedding spaces, similar words tendto be close to each other and dissimilar words far apart.For an input array of shape (d1, ..., dK),the shape of an output array is (d1, ..., dK, output_dim).All the input values should be integers in the range [0, input_dim).If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be(ip0, op0).By default, if any index mentioned is too large, it is replaced by the index that addressesthe last vector in an embedding matrix.Examples::  input_dim = 4  output_dim = 5  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)  y = [[  0.,   1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.,   9.],       [ 10.,  11.,  12.,  13.,  14.],       [ 15.,  16.,  17.,  18.,  19.]]  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]  x = [[ 1.,  3.],       [ 0.,  2.]]  // Mapped input x to its vector representation y.  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],                            [ 15.,  16.,  17.,  18.,  19.]],                           [[  0.,   1.,   2.,   3.,   4.],                            [ 10.,  11.,  12.,  13.,  14.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L74
/// </summary>
/// <param name="data">The input array to the embedding operator.</param>
/// <param name="input_dim">Vocabulary size of the input indices.</param>
/// <param name="output_dim">Dimension of the embedding vectors.</param>
/// <param name="weight">The embedding weight matrix.</param>
/// <param name="dtype">Data type of weight.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Embedding(Symbol data,
int input_dim,
int output_dim,
Symbol weight=null,
EmbeddingDtype dtype=EmbeddingDtype.Float32)
{
return new Operator("Embedding")
.SetParam("input_dim", input_dim)
.SetParam("output_dim", output_dim)
.SetParam("dtype", Util.EnumToString<EmbeddingDtype>(dtype,EmbeddingDtypeConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.CreateSymbol();
}
private static readonly List<string> TakeModeConvert = new List<string>(){"clip","raise","wrap"};
/// <summary>
/// Takes elements from an input array along the given axis.This function slices the input array along a particular axis with the provided indices.Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the outputwill have shape ``(i0, i1, d1, d2)``, computed by::  output[i,j,:,:] = input[indices[i,j],:,:].. note::   - `axis`- Only slicing along axis 0 is supported for now.   - `mode`- Only `clip` mode is supported for now.Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // takes elements with specified indices along axis 0  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],                             [ 3.,  4.]],                            [[ 3.,  4.],                             [ 5.,  6.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L135
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="a">The input array.</param>
/// <param name="indices">The indices of the values to be extracted.</param>
/// <param name="axis">The axis of input array to be taken.</param>
/// <param name="mode">Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error. </param>
 /// <returns>returns new symbol</returns>
public static Symbol Take(string symbol_name,
Symbol a,
Symbol indices,
int axis=0,
TakeMode mode=TakeMode.Clip)
{
return new Operator("take")
.SetParam("axis", axis)
.SetParam("mode", Util.EnumToString<TakeMode>(mode,TakeModeConvert))
.SetInput("a", a)
.SetInput("indices", indices)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Takes elements from an input array along the given axis.This function slices the input array along a particular axis with the provided indices.Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the outputwill have shape ``(i0, i1, d1, d2)``, computed by::  output[i,j,:,:] = input[indices[i,j],:,:].. note::   - `axis`- Only slicing along axis 0 is supported for now.   - `mode`- Only `clip` mode is supported for now.Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // takes elements with specified indices along axis 0  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],                             [ 3.,  4.]],                            [[ 3.,  4.],                             [ 5.,  6.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L135
/// </summary>
/// <param name="a">The input array.</param>
/// <param name="indices">The indices of the values to be extracted.</param>
/// <param name="axis">The axis of input array to be taken.</param>
/// <param name="mode">Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error. </param>
 /// <returns>returns new symbol</returns>
public static Symbol Take(Symbol a,
Symbol indices,
int axis=0,
TakeMode mode=TakeMode.Clip)
{
return new Operator("take")
.SetParam("axis", axis)
.SetParam("mode", Util.EnumToString<TakeMode>(mode,TakeModeConvert))
.SetInput("a", a)
.SetInput("indices", indices)
.CreateSymbol();
}
/// <summary>
/// Takes elements from a data batch... note::  `batch_take` is deprecated. Use `pick` instead.Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will bean output array of shape ``(i0,)`` with::  output[i] = input[i, indices[i]]Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // takes elements with specified indices  batch_take(x, [0,1,0]) = [ 1.  4.  5.]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L190
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="a">The input array</param>
/// <param name="indices">The index array</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchTake(string symbol_name,
Symbol a,
Symbol indices)
{
return new Operator("batch_take")
.SetInput("a", a)
.SetInput("indices", indices)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Takes elements from a data batch... note::  `batch_take` is deprecated. Use `pick` instead.Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will bean output array of shape ``(i0,)`` with::  output[i] = input[i, indices[i]]Examples::  x = [[ 1.,  2.],       [ 3.,  4.],       [ 5.,  6.]]  // takes elements with specified indices  batch_take(x, [0,1,0]) = [ 1.  4.  5.]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L190
/// </summary>
/// <param name="a">The input array</param>
/// <param name="indices">The index array</param>
 /// <returns>returns new symbol</returns>
public static Symbol BatchTake(Symbol a,
Symbol indices)
{
return new Operator("batch_take")
.SetInput("a", a)
.SetInput("indices", indices)
.CreateSymbol();
}
private static readonly List<string> OneHotDtypeConvert = new List<string>(){"float16","float32","float64","int32","uint8"};
/// <summary>
/// Returns a one-hot array.The locations represented by `indices` take value `on_value`, while allother locations take value `off_value`.`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would resultin an output array of shape ``(i0, i1, d)`` with::  output[i,j,:] = off_value  output[i,j,indices[i,j]] = on_valueExamples::  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]                           [ 1.  0.  0.]                           [ 0.  0.  1.]                           [ 1.  0.  0.]]  one_hot([1,0,2,0], 3, on_value=8, off_value=1,          dtype='int32') = [[1 8 1]                            [8 1 1]                            [1 1 8]                            [8 1 1]]  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]                                      [ 1.  0.  0.]]                                     [[ 0.  1.  0.]                                      [ 1.  0.  0.]]                                     [[ 0.  0.  1.]                                      [ 1.  0.  0.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L236
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="indices">array of locations where to set on_value</param>
/// <param name="depth">Depth of the one hot dimension.</param>
/// <param name="on_value">The value assigned to the locations represented by indices.</param>
/// <param name="off_value">The value assigned to the locations not represented by indices.</param>
/// <param name="dtype">DType of the output</param>
 /// <returns>returns new symbol</returns>
public static Symbol OneHot(string symbol_name,
Symbol indices,
int depth,
double on_value=1,
double off_value=0,
OneHotDtype dtype=OneHotDtype.Float32)
{
return new Operator("one_hot")
.SetParam("depth", depth)
.SetParam("on_value", on_value)
.SetParam("off_value", off_value)
.SetParam("dtype", Util.EnumToString<OneHotDtype>(dtype,OneHotDtypeConvert))
.SetInput("indices", indices)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns a one-hot array.The locations represented by `indices` take value `on_value`, while allother locations take value `off_value`.`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would resultin an output array of shape ``(i0, i1, d)`` with::  output[i,j,:] = off_value  output[i,j,indices[i,j]] = on_valueExamples::  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]                           [ 1.  0.  0.]                           [ 0.  0.  1.]                           [ 1.  0.  0.]]  one_hot([1,0,2,0], 3, on_value=8, off_value=1,          dtype='int32') = [[1 8 1]                            [8 1 1]                            [1 1 8]                            [8 1 1]]  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]                                      [ 1.  0.  0.]]                                     [[ 0.  1.  0.]                                      [ 1.  0.  0.]]                                     [[ 0.  0.  1.]                                      [ 1.  0.  0.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\indexing_op.cc:L236
/// </summary>
/// <param name="indices">array of locations where to set on_value</param>
/// <param name="depth">Depth of the one hot dimension.</param>
/// <param name="on_value">The value assigned to the locations represented by indices.</param>
/// <param name="off_value">The value assigned to the locations not represented by indices.</param>
/// <param name="dtype">DType of the output</param>
 /// <returns>returns new symbol</returns>
public static Symbol OneHot(Symbol indices,
int depth,
double on_value=1,
double off_value=0,
OneHotDtype dtype=OneHotDtype.Float32)
{
return new Operator("one_hot")
.SetParam("depth", depth)
.SetParam("on_value", on_value)
.SetParam("off_value", off_value)
.SetParam("dtype", Util.EnumToString<OneHotDtype>(dtype,OneHotDtypeConvert))
.SetInput("indices", indices)
.CreateSymbol();
}
/// <summary>
/// Gather elements or slices from `data` and store to a tensor whoseshape is defined by `indices`. `gather_nd` and `scatter_nd` are inverse functionsto each other.Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.The elements in output is defined as follows::  output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],                                                      ...,                                                      indices[M-1, y_0, ..., y_{K-1}],                                                      x_M, ..., x_{N-1}]Examples::  data = [[0, 1], [2, 3]]  indices = [[1, 1, 0], [0, 1, 0]]  gather_nd(data, indices) = [2, 3, 0]
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">data</param>
/// <param name="indices">indices</param>
 /// <returns>returns new symbol</returns>
public static Symbol GatherNd(string symbol_name,
Symbol data,
Symbol indices)
{
return new Operator("gather_nd")
.SetInput("data", data)
.SetInput("indices", indices)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Gather elements or slices from `data` and store to a tensor whoseshape is defined by `indices`. `gather_nd` and `scatter_nd` are inverse functionsto each other.Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.The elements in output is defined as follows::  output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],                                                      ...,                                                      indices[M-1, y_0, ..., y_{K-1}],                                                      x_M, ..., x_{N-1}]Examples::  data = [[0, 1], [2, 3]]  indices = [[1, 1, 0], [0, 1, 0]]  gather_nd(data, indices) = [2, 3, 0]
/// </summary>
/// <param name="data">data</param>
/// <param name="indices">indices</param>
 /// <returns>returns new symbol</returns>
public static Symbol GatherNd(Symbol data,
Symbol indices)
{
return new Operator("gather_nd")
.SetInput("data", data)
.SetInput("indices", indices)
.CreateSymbol();
}
/// <summary>
/// Scatters data into a new tensor according to indices.`gather_nd` and `scatter_nd` are inverse functions to each other.Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.The elements in output is defined as follows::  output[indices[0, y_0, ..., y_{K-1}],         ...,         indices[M-1, y_0, ..., y_{K-1}],         x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]all other entries in output are 0.Examples::  data = [2, 3, 0]  indices = [[1, 1, 0], [0, 1, 0]]  scatter_nd(data, indices) = [[0, 0], [2, 3]]
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">data</param>
/// <param name="indices">indices</param>
/// <param name="shape">Shape of output.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ScatterNd(string symbol_name,
Symbol data,
Symbol indices,
Shape shape)
{
return new Operator("scatter_nd")
.SetParam("shape", shape)
.SetInput("data", data)
.SetInput("indices", indices)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Scatters data into a new tensor according to indices.`gather_nd` and `scatter_nd` are inverse functions to each other.Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.The elements in output is defined as follows::  output[indices[0, y_0, ..., y_{K-1}],         ...,         indices[M-1, y_0, ..., y_{K-1}],         x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]all other entries in output are 0.Examples::  data = [2, 3, 0]  indices = [[1, 1, 0], [0, 1, 0]]  scatter_nd(data, indices) = [[0, 0], [2, 3]]
/// </summary>
/// <param name="data">data</param>
/// <param name="indices">indices</param>
/// <param name="shape">Shape of output.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ScatterNd(Symbol data,
Symbol indices,
Shape shape)
{
return new Operator("scatter_nd")
.SetParam("shape", shape)
.SetInput("data", data)
.SetInput("indices", indices)
.CreateSymbol();
}
/// <summary>
/// Return an array of zeros with the same shape and typeas the input array.The storage type of ``zeros_like`` output depends on the storage type of the input- zeros_like(row_sparse) = row_sparse- zeros_like(csr) = csr- zeros_like(default) = defaultExamples::  x = [[ 1.,  1.,  1.],       [ 1.,  1.,  1.]]  zeros_like(x) = [[ 0.,  0.,  0.],                   [ 0.,  0.,  0.]]
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ZerosLike(string symbol_name,
Symbol data)
{
return new Operator("zeros_like")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Return an array of zeros with the same shape and typeas the input array.The storage type of ``zeros_like`` output depends on the storage type of the input- zeros_like(row_sparse) = row_sparse- zeros_like(csr) = csr- zeros_like(default) = defaultExamples::  x = [[ 1.,  1.,  1.],       [ 1.,  1.,  1.]]  zeros_like(x) = [[ 0.,  0.,  0.],                   [ 0.,  0.,  0.]]
/// </summary>
/// <param name="data">The input</param>
 /// <returns>returns new symbol</returns>
public static Symbol ZerosLike(Symbol data)
{
return new Operator("zeros_like")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Return an array of ones with the same shape and typeas the input array.Examples::  x = [[ 0.,  0.,  0.],       [ 0.,  0.,  0.]]  ones_like(x) = [[ 1.,  1.,  1.],                  [ 1.,  1.,  1.]]
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input</param>
 /// <returns>returns new symbol</returns>
public static Symbol OnesLike(string symbol_name,
Symbol data)
{
return new Operator("ones_like")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Return an array of ones with the same shape and typeas the input array.Examples::  x = [[ 0.,  0.,  0.],       [ 0.,  0.,  0.]]  ones_like(x) = [[ 1.,  1.,  1.],                  [ 1.,  1.,  1.]]
/// </summary>
/// <param name="data">The input</param>
 /// <returns>returns new symbol</returns>
public static Symbol OnesLike(Symbol data)
{
return new Operator("ones_like")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Reshapes the input array... note:: ``Reshape`` is deprecated, use ``reshape``Given an array and a shape, this function returns a copy of the array in the new shape.The shape is a tuple of integers such as (2,3,4).The size of the new shape should be same as the size of the input array.Example::  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:- ``0``  copy this dimension from the input to the output shape.  Example::  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions  keeping the size of the new array same as that of the input array.  At most one dimension of shape can be -1.  Example::  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)  - input shape = (2,3,4), shape=(-1,), output shape = (24,)- ``-2`` copy all/remainder of the input dimensions to the output shape.  Example::  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.  Example::  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).  Example::  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)If the argument `reverse` is set to 1, then the special values are inferred from right to left.  Example::  - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)  - with reverse=1, output shape will be (50,4).Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L105
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to reshape.</param>
/// <param name="shape">The target shape</param>
/// <param name="reverse">If true then the special values are inferred from right to left</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reshape(string symbol_name,
Symbol data,
Shape shape=null,
bool reverse=false)
{
return new Operator("Reshape")
.SetParam("shape", shape)
.SetParam("reverse", reverse)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Reshapes the input array... note:: ``Reshape`` is deprecated, use ``reshape``Given an array and a shape, this function returns a copy of the array in the new shape.The shape is a tuple of integers such as (2,3,4).The size of the new shape should be same as the size of the input array.Example::  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:- ``0``  copy this dimension from the input to the output shape.  Example::  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions  keeping the size of the new array same as that of the input array.  At most one dimension of shape can be -1.  Example::  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)  - input shape = (2,3,4), shape=(-1,), output shape = (24,)- ``-2`` copy all/remainder of the input dimensions to the output shape.  Example::  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.  Example::  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).  Example::  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)If the argument `reverse` is set to 1, then the special values are inferred from right to left.  Example::  - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)  - with reverse=1, output shape will be (50,4).Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L105
/// </summary>
/// <param name="data">Input data to reshape.</param>
/// <param name="shape">The target shape</param>
/// <param name="reverse">If true then the special values are inferred from right to left</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reshape(Symbol data,
Shape shape=null,
bool reverse=false)
{
return new Operator("Reshape")
.SetParam("shape", shape)
.SetParam("reverse", reverse)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Flattens the input array into a 2-D array by collapsing the higher dimensions... note:: `Flatten` is deprecated. Use `flatten` instead.For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapesthe input array into an output array of shape ``(d1, d2*...*dk)``.Example::    x = [[        [1,2,3],        [4,5,6],        [7,8,9]    ],    [    [1,2,3],        [4,5,6],        [7,8,9]    ]],    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L149
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Flatten(string symbol_name,
Symbol data)
{
return new Operator("Flatten")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Flattens the input array into a 2-D array by collapsing the higher dimensions... note:: `Flatten` is deprecated. Use `flatten` instead.For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapesthe input array into an output array of shape ``(d1, d2*...*dk)``.Example::    x = [[        [1,2,3],        [4,5,6],        [7,8,9]    ],    [    [1,2,3],        [4,5,6],        [7,8,9]    ]],    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L149
/// </summary>
/// <param name="data">Input array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Flatten(Symbol data)
{
return new Operator("Flatten")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Permutes the dimensions of an array.Examples::  x = [[ 1, 2],       [ 3, 4]]  transpose(x) = [[ 1.,  3.],                  [ 2.,  4.]]  x = [[[ 1.,  2.],        [ 3.,  4.]],       [[ 5.,  6.],        [ 7.,  8.]]]  transpose(x) = [[[ 1.,  5.],                   [ 3.,  7.]],                  [[ 2.,  6.],                   [ 4.,  8.]]]  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],                                 [ 5.,  6.]],                                [[ 3.,  4.],                                 [ 7.,  8.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L194
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Source input</param>
/// <param name="axes">Target axis order. By default the axes will be inverted.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Transpose(string symbol_name,
Symbol data,
Shape axes=null)
{
return new Operator("transpose")
.SetParam("axes", axes)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Permutes the dimensions of an array.Examples::  x = [[ 1, 2],       [ 3, 4]]  transpose(x) = [[ 1.,  3.],                  [ 2.,  4.]]  x = [[[ 1.,  2.],        [ 3.,  4.]],       [[ 5.,  6.],        [ 7.,  8.]]]  transpose(x) = [[[ 1.,  5.],                   [ 3.,  7.]],                  [[ 2.,  6.],                   [ 4.,  8.]]]  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],                                 [ 5.,  6.]],                                [[ 3.,  4.],                                 [ 7.,  8.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L194
/// </summary>
/// <param name="data">Source input</param>
/// <param name="axes">Target axis order. By default the axes will be inverted.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Transpose(Symbol data,
Shape axes=null)
{
return new Operator("transpose")
.SetParam("axes", axes)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Inserts a new axis of size 1 into the array shapeFor example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``will return a new array with shape ``(2,1,3,4)``.Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L230
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Source input</param>
/// <param name="axis">Position where new axis is to be inserted. Suppose that the input `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim, ndim]`</param>
 /// <returns>returns new symbol</returns>
public static Symbol ExpandDims(string symbol_name,
Symbol data,
int axis)
{
return new Operator("expand_dims")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Inserts a new axis of size 1 into the array shapeFor example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``will return a new array with shape ``(2,1,3,4)``.Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L230
/// </summary>
/// <param name="data">Source input</param>
/// <param name="axis">Position where new axis is to be inserted. Suppose that the input `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim, ndim]`</param>
 /// <returns>returns new symbol</returns>
public static Symbol ExpandDims(Symbol data,
int axis)
{
return new Operator("expand_dims")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Slices a contiguous region of the array... note:: ``crop`` is deprecated. Use ``slice`` instead.This function returns a sliced continuous region of the array between the indices givenby `begin` and `end`.For an input array of `n` dimensions, slice operation with ``begin=(b_0, b_1...b_n-1)`` indicesand ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape``(e_1-b_0, ..., e_n-b_n-1)``.The resulting array's *k*-th dimension contains elementsfrom the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.For an input array of non-default storage type(e.g. `csr` or `row_sparse`), it only supportsslicing on the first dimension.Example::  x = [[  1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.],       [  9.,  10.,  11.,  12.]]  slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],                                     [ 6.,  7.,  8.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L278
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Source input</param>
/// <param name="begin">starting indices for the slice operation, supports negative indices.</param>
/// <param name="end">ending indices for the slice operation, supports negative indices.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Slice(string symbol_name,
Symbol data,
Shape begin,
Shape end)
{
return new Operator("slice")
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Slices a contiguous region of the array... note:: ``crop`` is deprecated. Use ``slice`` instead.This function returns a sliced continuous region of the array between the indices givenby `begin` and `end`.For an input array of `n` dimensions, slice operation with ``begin=(b_0, b_1...b_n-1)`` indicesand ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape``(e_1-b_0, ..., e_n-b_n-1)``.The resulting array's *k*-th dimension contains elementsfrom the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.For an input array of non-default storage type(e.g. `csr` or `row_sparse`), it only supportsslicing on the first dimension.Example::  x = [[  1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.],       [  9.,  10.,  11.,  12.]]  slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],                                     [ 6.,  7.,  8.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L278
/// </summary>
/// <param name="data">Source input</param>
/// <param name="begin">starting indices for the slice operation, supports negative indices.</param>
/// <param name="end">ending indices for the slice operation, supports negative indices.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Slice(Symbol data,
Shape begin,
Shape end)
{
return new Operator("slice")
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Slices along a given axis.Returns an array slice along a given `axis` starting from the `begin` indexto the `end` index.Examples::  x = [[  1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.],       [  9.,  10.,  11.,  12.]]  slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],                                           [  9.,  10.,  11.,  12.]]  slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],                                           [  5.,   6.],                                           [  9.,  10.]]  slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],                                             [  6.,   7.],                                             [ 10.,  11.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L360
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Source input</param>
/// <param name="axis">Axis along which to be sliced, supports negative indexes.</param>
/// <param name="begin">The beginning index along the axis to be sliced,  supports negative indexes.</param>
/// <param name="end">The ending index along the axis to be sliced,  supports negative indexes.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SliceAxis(string symbol_name,
Symbol data,
int axis,
int begin,
int end)
{
return new Operator("slice_axis")
.SetParam("axis", axis)
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Slices along a given axis.Returns an array slice along a given `axis` starting from the `begin` indexto the `end` index.Examples::  x = [[  1.,   2.,   3.,   4.],       [  5.,   6.,   7.,   8.],       [  9.,  10.,  11.,  12.]]  slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],                                           [  9.,  10.,  11.,  12.]]  slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],                                           [  5.,   6.],                                           [  9.,  10.]]  slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],                                             [  6.,   7.],                                             [ 10.,  11.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L360
/// </summary>
/// <param name="data">Source input</param>
/// <param name="axis">Axis along which to be sliced, supports negative indexes.</param>
/// <param name="begin">The beginning index along the axis to be sliced,  supports negative indexes.</param>
/// <param name="end">The ending index along the axis to be sliced,  supports negative indexes.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SliceAxis(Symbol data,
int axis,
int begin,
int end)
{
return new Operator("slice_axis")
.SetParam("axis", axis)
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Clips (limits) the values in an array.Given an interval, values outside the interval are clipped to the interval edges.Clipping ``x`` between `a_min` and `a_x` would be::   clip(x, a_min, a_max) = max(min(x, a_max), a_min))Example::    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L392
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
/// <param name="a_min">Minimum value</param>
/// <param name="a_max">Maximum value</param>
 /// <returns>returns new symbol</returns>
public static Symbol Clip(string symbol_name,
Symbol data,
float a_min,
float a_max)
{
return new Operator("clip")
.SetParam("a_min", a_min)
.SetParam("a_max", a_max)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Clips (limits) the values in an array.Given an interval, values outside the interval are clipped to the interval edges.Clipping ``x`` between `a_min` and `a_x` would be::   clip(x, a_min, a_max) = max(min(x, a_max), a_min))Example::    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L392
/// </summary>
/// <param name="data">Input array.</param>
/// <param name="a_min">Minimum value</param>
/// <param name="a_max">Maximum value</param>
 /// <returns>returns new symbol</returns>
public static Symbol Clip(Symbol data,
float a_min,
float a_max)
{
return new Operator("clip")
.SetParam("a_min", a_min)
.SetParam("a_max", a_max)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Repeats elements of an array.By default, ``repeat`` flattens the input array into 1-D and then repeats theelements::  x = [[ 1, 2],       [ 3, 4]]  repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]The parameter ``axis`` specifies the axis along which to perform repeat::  repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],                                  [ 3.,  3.,  4.,  4.]]  repeat(x, repeats=2, axis=0) = [[ 1.,  2.],                                  [ 1.,  2.],                                  [ 3.,  4.],                                  [ 3.,  4.]]  repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],                                   [ 3.,  3.,  4.,  4.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L434
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data array</param>
/// <param name="repeats">The number of repetitions for each element.</param>
/// <param name="axis">The axis along which to repeat values. The negative numbers are interpreted counting from the backward. By default, use the flattened input array, and return a flat output array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Repeat(string symbol_name,
Symbol data,
int repeats,
int? axis=null)
{
return new Operator("repeat")
.SetParam("repeats", repeats)
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Repeats elements of an array.By default, ``repeat`` flattens the input array into 1-D and then repeats theelements::  x = [[ 1, 2],       [ 3, 4]]  repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]The parameter ``axis`` specifies the axis along which to perform repeat::  repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],                                  [ 3.,  3.,  4.,  4.]]  repeat(x, repeats=2, axis=0) = [[ 1.,  2.],                                  [ 1.,  2.],                                  [ 3.,  4.],                                  [ 3.,  4.]]  repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],                                   [ 3.,  3.,  4.,  4.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L434
/// </summary>
/// <param name="data">Input data array</param>
/// <param name="repeats">The number of repetitions for each element.</param>
/// <param name="axis">The axis along which to repeat values. The negative numbers are interpreted counting from the backward. By default, use the flattened input array, and return a flat output array.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Repeat(Symbol data,
int repeats,
int? axis=null)
{
return new Operator("repeat")
.SetParam("repeats", repeats)
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Repeats the whole array multiple times.If ``reps`` has length *d*, and input array has dimension of *n*. There arethere cases:- **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::    x = [[1, 2],         [3, 4]]    tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],                           [ 3.,  4.,  3.,  4.,  3.,  4.],                           [ 1.,  2.,  1.,  2.,  1.,  2.],                           [ 3.,  4.,  3.,  4.,  3.,  4.]]- **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for  an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::    tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],                          [ 3.,  4.,  3.,  4.]]- **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a  shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::    tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.],                              [ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.]],                             [[ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.],                              [ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L495
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data array</param>
/// <param name="reps">The number of times for repeating the tensor a. If reps has length d, the result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tile(string symbol_name,
Symbol data,
Shape reps)
{
return new Operator("tile")
.SetParam("reps", reps)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Repeats the whole array multiple times.If ``reps`` has length *d*, and input array has dimension of *n*. There arethere cases:- **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::    x = [[1, 2],         [3, 4]]    tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],                           [ 3.,  4.,  3.,  4.,  3.,  4.],                           [ 1.,  2.,  1.,  2.,  1.,  2.],                           [ 3.,  4.,  3.,  4.,  3.,  4.]]- **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for  an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::    tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],                          [ 3.,  4.,  3.,  4.]]- **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a  shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::    tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.],                              [ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.]],                             [[ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.],                              [ 1.,  2.,  1.,  2.,  1.,  2.],                              [ 3.,  4.,  3.,  4.,  3.,  4.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L495
/// </summary>
/// <param name="data">Input data array</param>
/// <param name="reps">The number of times for repeating the tensor a. If reps has length d, the result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Tile(Symbol data,
Shape reps)
{
return new Operator("tile")
.SetParam("reps", reps)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Reverses the order of elements along given axis while preserving array shape.Note: reverse and flip are equivalent. We use reverse in the following examples.Examples::  x = [[ 0.,  1.,  2.,  3.,  4.],       [ 5.,  6.,  7.,  8.,  9.]]  reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],                        [ 0.,  1.,  2.,  3.,  4.]]  reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],                        [ 9.,  8.,  7.,  6.,  5.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L536
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data array</param>
/// <param name="axis">The axis which to reverse elements.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reverse(string symbol_name,
Symbol data,
Shape axis)
{
return new Operator("reverse")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Reverses the order of elements along given axis while preserving array shape.Note: reverse and flip are equivalent. We use reverse in the following examples.Examples::  x = [[ 0.,  1.,  2.,  3.,  4.],       [ 5.,  6.,  7.,  8.,  9.]]  reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],                        [ 0.,  1.,  2.,  3.,  4.]]  reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],                        [ 9.,  8.,  7.,  6.,  5.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\matrix_op.cc:L536
/// </summary>
/// <param name="data">Input data array</param>
/// <param name="axis">The axis which to reverse elements.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Reverse(Symbol data,
Shape axis)
{
return new Operator("reverse")
.SetParam("axis", axis)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Join a sequence of arrays along a new axis.The axis parameter specifies the index of the new axis in the dimensions of theresult. For example, if axis=0 it will be the first dimension and if axis=-1 itwill be the last dimension.Examples::  x = [1, 2]  y = [3, 4]  stack(x, y) = [[1, 2],                 [3, 4]]  stack(x, y, axis=1) = [[1, 3],                         [2, 4]]
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">List of arrays to stack</param>
/// <param name="num_args">Number of inputs to be stacked.</param>
/// <param name="axis">The axis in the result array along which the input arrays are stacked.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Stack(string symbol_name,
Symbol[] data,
int num_args,
int axis=0)
{
return new Operator("stack")
.SetParam("num_args", num_args)
.SetParam("axis", axis)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Join a sequence of arrays along a new axis.The axis parameter specifies the index of the new axis in the dimensions of theresult. For example, if axis=0 it will be the first dimension and if axis=-1 itwill be the last dimension.Examples::  x = [1, 2]  y = [3, 4]  stack(x, y) = [[1, 2],                 [3, 4]]  stack(x, y, axis=1) = [[1, 3],                         [2, 4]]
/// </summary>
/// <param name="data">List of arrays to stack</param>
/// <param name="num_args">Number of inputs to be stacked.</param>
/// <param name="axis">The axis in the result array along which the input arrays are stacked.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Stack(Symbol[] data,
int num_args,
int axis=0)
{
return new Operator("stack")
.SetParam("num_args", num_args)
.SetParam("axis", axis)
.AddInput(data)
.CreateSymbol();
}
private static readonly List<string> TopkRetTypConvert = new List<string>(){"both","indices","mask","value"};
/// <summary>
/// Returns the top *k* elements in an input array along the given axis.Examples::  x = [[ 0.3,  0.2,  0.4],       [ 0.1,  0.3,  0.2]]  // returns an index of the largest element on last axis  topk(x) = [[ 2.],             [ 1.]]  // returns the value of top-2 largest elements on last axis  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],                                   [ 0.3,  0.2]]  // returns the value of top-2 smallest elements on last axis  topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],                                               [ 0.1 ,  0.2]]  // returns the value of top-2 largest elements on axis 0  topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],                                           [ 0.1,  0.2,  0.2]]  // flattens and then returns list of both values and indices  topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L62
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to choose the top k indices. If not given, the flattened array is used. Default is -1.</param>
/// <param name="k">Number of top elements to select, should be always smaller than or equal to the element number in the given axis. A global sort is performed if set k < 1.</param>
/// <param name="ret_typ">The return type. "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.</param>
/// <param name="is_ascend">Whether to choose k largest or k smallest elements. Top K largest elements will be chosen if set to false.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Topk(string symbol_name,
Symbol data,
int axis=-1,
int k=1,
TopkRetTyp ret_typ=TopkRetTyp.Indices,
bool is_ascend=false)
{
return new Operator("topk")
.SetParam("axis", axis)
.SetParam("k", k)
.SetParam("ret_typ", Util.EnumToString<TopkRetTyp>(ret_typ,TopkRetTypConvert))
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the top *k* elements in an input array along the given axis.Examples::  x = [[ 0.3,  0.2,  0.4],       [ 0.1,  0.3,  0.2]]  // returns an index of the largest element on last axis  topk(x) = [[ 2.],             [ 1.]]  // returns the value of top-2 largest elements on last axis  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],                                   [ 0.3,  0.2]]  // returns the value of top-2 smallest elements on last axis  topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],                                               [ 0.1 ,  0.2]]  // returns the value of top-2 largest elements on axis 0  topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],                                           [ 0.1,  0.2,  0.2]]  // flattens and then returns list of both values and indices  topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L62
/// </summary>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to choose the top k indices. If not given, the flattened array is used. Default is -1.</param>
/// <param name="k">Number of top elements to select, should be always smaller than or equal to the element number in the given axis. A global sort is performed if set k < 1.</param>
/// <param name="ret_typ">The return type. "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.</param>
/// <param name="is_ascend">Whether to choose k largest or k smallest elements. Top K largest elements will be chosen if set to false.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Topk(Symbol data,
int axis=-1,
int k=1,
TopkRetTyp ret_typ=TopkRetTyp.Indices,
bool is_ascend=false)
{
return new Operator("topk")
.SetParam("axis", axis)
.SetParam("k", k)
.SetParam("ret_typ", Util.EnumToString<TopkRetTyp>(ret_typ,TopkRetTypConvert))
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns a sorted copy of an input array along the given axis.Examples::  x = [[ 1, 4],       [ 3, 1]]  // sorts along the last axis  sort(x) = [[ 1.,  4.],             [ 1.,  3.]]  // flattens and then sorts  sort(x) = [ 1.,  1.,  3.,  4.]  // sorts along the first axis  sort(x, axis=0) = [[ 1.,  1.],                     [ 3.,  4.]]  // in a descend order  sort(x, is_ascend=0) = [[ 4.,  1.],                          [ 3.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L125
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to choose sort the input tensor. If not given, the flattened array is used. Default is -1.</param>
/// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sort(string symbol_name,
Symbol data,
int axis=-1,
bool is_ascend=true)
{
return new Operator("sort")
.SetParam("axis", axis)
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns a sorted copy of an input array along the given axis.Examples::  x = [[ 1, 4],       [ 3, 1]]  // sorts along the last axis  sort(x) = [[ 1.,  4.],             [ 1.,  3.]]  // flattens and then sorts  sort(x) = [ 1.,  1.,  3.,  4.]  // sorts along the first axis  sort(x, axis=0) = [[ 1.,  1.],                     [ 3.,  4.]]  // in a descend order  sort(x, is_ascend=0) = [[ 4.,  1.],                          [ 3.,  1.]]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L125
/// </summary>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to choose sort the input tensor. If not given, the flattened array is used. Default is -1.</param>
/// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Sort(Symbol data,
int axis=-1,
bool is_ascend=true)
{
return new Operator("sort")
.SetParam("axis", axis)
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Returns the indices that would sort an input array along the given axis.This function performs sorting along the given axis and returns an array of indices having same shapeas an input array that index data in sorted order.Examples::  x = [[ 0.3,  0.2,  0.4],       [ 0.1,  0.3,  0.2]]  // sort along axis -1  argsort(x) = [[ 1.,  0.,  2.],                [ 0.,  2.,  1.]]  // sort along axis 0  argsort(x, axis=0) = [[ 1.,  0.,  1.]                        [ 0.,  1.,  0.]]  // flatten and then sort  argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L175
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to sort the input tensor. If not given, the flattened array is used. Default is -1.</param>
/// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argsort(string symbol_name,
Symbol data,
int axis=-1,
bool is_ascend=true)
{
return new Operator("argsort")
.SetParam("axis", axis)
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Returns the indices that would sort an input array along the given axis.This function performs sorting along the given axis and returns an array of indices having same shapeas an input array that index data in sorted order.Examples::  x = [[ 0.3,  0.2,  0.4],       [ 0.1,  0.3,  0.2]]  // sort along axis -1  argsort(x) = [[ 1.,  0.,  2.],                [ 0.,  2.,  1.]]  // sort along axis 0  argsort(x, axis=0) = [[ 1.,  0.,  1.]                        [ 0.,  1.,  0.]]  // flatten and then sort  argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]Defined in G:\deeplearn\mxnet\src\operator\tensor\ordering_op.cc:L175
/// </summary>
/// <param name="data">The input array</param>
/// <param name="axis">Axis along which to sort the input tensor. If not given, the flattened array is used. Default is -1.</param>
/// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Argsort(Symbol data,
int axis=-1,
bool is_ascend=true)
{
return new Operator("argsort")
.SetParam("axis", axis)
.SetParam("is_ascend", is_ascend)
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> UpsamplingSampleTypeConvert = new List<string>(){"bilinear","nearest"};
private static readonly List<string> UpsamplingMultiInputModeConvert = new List<string>(){"concat","sum"};
/// <summary>
/// Performs nearest neighbor/bilinear up sampling to inputs.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Array of tensors to upsample</param>
/// <param name="scale">Up sampling scale</param>
/// <param name="sample_type">upsampling method</param>
/// <param name="num_args">Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.</param>
/// <param name="num_filter">Input filter. Only used by bilinear sample_type.</param>
/// <param name="multi_input_mode">How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.</param>
/// <param name="workspace">Tmp workspace for deconvolution (MB)</param>
 /// <returns>returns new symbol</returns>
public static Symbol UpSampling(string symbol_name,
Symbol[] data,
int scale,
UpsamplingSampleType sample_type,
int num_args,
int num_filter=0,
UpsamplingMultiInputMode multi_input_mode=UpsamplingMultiInputMode.Concat,
long workspace=512)
{
return new Operator("UpSampling")
.SetParam("scale", scale)
.SetParam("sample_type", Util.EnumToString<UpsamplingSampleType>(sample_type,UpsamplingSampleTypeConvert))
.SetParam("num_args", num_args)
.SetParam("num_filter", num_filter)
.SetParam("multi_input_mode", Util.EnumToString<UpsamplingMultiInputMode>(multi_input_mode,UpsamplingMultiInputModeConvert))
.SetParam("workspace", workspace)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Performs nearest neighbor/bilinear up sampling to inputs.
/// </summary>
/// <param name="data">Array of tensors to upsample</param>
/// <param name="scale">Up sampling scale</param>
/// <param name="sample_type">upsampling method</param>
/// <param name="num_args">Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.</param>
/// <param name="num_filter">Input filter. Only used by bilinear sample_type.</param>
/// <param name="multi_input_mode">How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.</param>
/// <param name="workspace">Tmp workspace for deconvolution (MB)</param>
 /// <returns>returns new symbol</returns>
public static Symbol UpSampling(Symbol[] data,
int scale,
UpsamplingSampleType sample_type,
int num_args,
int num_filter=0,
UpsamplingMultiInputMode multi_input_mode=UpsamplingMultiInputMode.Concat,
long workspace=512)
{
return new Operator("UpSampling")
.SetParam("scale", scale)
.SetParam("sample_type", Util.EnumToString<UpsamplingSampleType>(sample_type,UpsamplingSampleTypeConvert))
.SetParam("num_args", num_args)
.SetParam("num_filter", num_filter)
.SetParam("multi_input_mode", Util.EnumToString<UpsamplingMultiInputMode>(multi_input_mode,UpsamplingMultiInputModeConvert))
.SetParam("workspace", workspace)
.AddInput(data)
.CreateSymbol();
}
private static readonly List<string> ActivationActTypeConvert = new List<string>(){"relu","sigmoid","softrelu","tanh"};
/// <summary>
/// Applies an activation function element-wise to the input.The following activation functions are supported:- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`Defined in G:\deeplearn\mxnet\src\operator\activation.cc:L91
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array to activation function.</param>
/// <param name="act_type">Activation function to be applied.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Activation(string symbol_name,
Symbol data,
ActivationActType act_type)
{
return new Operator("Activation")
.SetParam("act_type", Util.EnumToString<ActivationActType>(act_type,ActivationActTypeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies an activation function element-wise to the input.The following activation functions are supported:- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`Defined in G:\deeplearn\mxnet\src\operator\activation.cc:L91
/// </summary>
/// <param name="data">Input array to activation function.</param>
/// <param name="act_type">Activation function to be applied.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Activation(Symbol data,
ActivationActType act_type)
{
return new Operator("Activation")
.SetParam("act_type", Util.EnumToString<ActivationActType>(act_type,ActivationActTypeConvert))
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Applies bilinear sampling to input feature map.Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,except that the operator has the backward pass.Given :math:`data` and :math:`grid`, then the output is computed by.. math::  x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\  y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\  output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src}):math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.Example 1::  ## Zoom out data two times  data = array([[[[1, 4, 3, 6],                  [1, 8, 8, 9],                  [0, 4, 1, 5],                  [1, 0, 1, 3]]]])  affine_matrix = array([[2, 0, 0],                         [0, 2, 0]])  affine_matrix = reshape(affine_matrix, shape=(1, 6))  grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))  out = BilinearSampler(data, grid)  out  [[[[ 0,   0,     0,   0],     [ 0,   3.5,   6.5, 0],     [ 0,   1.25,  2.5, 0],     [ 0,   0,     0,   0]]]Example 2::  ## shift data horizontally by -1 pixel  data = array([[[[1, 4, 3, 6],                  [1, 8, 8, 9],                  [0, 4, 1, 5],                  [1, 0, 1, 3]]]])  warp_maxtrix = array([[[[1, 1, 1, 1],                          [1, 1, 1, 1],                          [1, 1, 1, 1],                          [1, 1, 1, 1]],                         [[0, 0, 0, 0],                          [0, 0, 0, 0],                          [0, 0, 0, 0],                          [0, 0, 0, 0]]]])  grid = GridGenerator(data=warp_matrix, transform_type='warp')  out = BilinearSampler(data, grid)  out  [[[[ 4,  3,  6,  0],     [ 8,  8,  9,  0],     [ 4,  1,  5,  0],     [ 0,  1,  3,  0]]]Defined in G:\deeplearn\mxnet\src\operator\bilinear_sampler.cc:L244
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the BilinearsamplerOp.</param>
/// <param name="grid">Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src</param>
 /// <returns>returns new symbol</returns>
public static Symbol BilinearSampler(string symbol_name,
Symbol data,
Symbol grid)
{
return new Operator("BilinearSampler")
.SetInput("data", data)
.SetInput("grid", grid)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies bilinear sampling to input feature map.Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,except that the operator has the backward pass.Given :math:`data` and :math:`grid`, then the output is computed by.. math::  x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\  y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\  output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src}):math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.Example 1::  ## Zoom out data two times  data = array([[[[1, 4, 3, 6],                  [1, 8, 8, 9],                  [0, 4, 1, 5],                  [1, 0, 1, 3]]]])  affine_matrix = array([[2, 0, 0],                         [0, 2, 0]])  affine_matrix = reshape(affine_matrix, shape=(1, 6))  grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))  out = BilinearSampler(data, grid)  out  [[[[ 0,   0,     0,   0],     [ 0,   3.5,   6.5, 0],     [ 0,   1.25,  2.5, 0],     [ 0,   0,     0,   0]]]Example 2::  ## shift data horizontally by -1 pixel  data = array([[[[1, 4, 3, 6],                  [1, 8, 8, 9],                  [0, 4, 1, 5],                  [1, 0, 1, 3]]]])  warp_maxtrix = array([[[[1, 1, 1, 1],                          [1, 1, 1, 1],                          [1, 1, 1, 1],                          [1, 1, 1, 1]],                         [[0, 0, 0, 0],                          [0, 0, 0, 0],                          [0, 0, 0, 0],                          [0, 0, 0, 0]]]])  grid = GridGenerator(data=warp_matrix, transform_type='warp')  out = BilinearSampler(data, grid)  out  [[[[ 4,  3,  6,  0],     [ 8,  8,  9,  0],     [ 4,  1,  5,  0],     [ 0,  1,  3,  0]]]Defined in G:\deeplearn\mxnet\src\operator\bilinear_sampler.cc:L244
/// </summary>
/// <param name="data">Input data to the BilinearsamplerOp.</param>
/// <param name="grid">Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src</param>
 /// <returns>returns new symbol</returns>
public static Symbol BilinearSampler(Symbol data,
Symbol grid)
{
return new Operator("BilinearSampler")
.SetInput("data", data)
.SetInput("grid", grid)
.CreateSymbol();
}
private static readonly List<string> ConvolutionCudnnTuneConvert = new List<string>(){"fastest","limited_workspace","off"};
private static readonly List<string> ConvolutionLayoutConvert = new List<string>(){"NCDHW","NCHW","NCW","NDHWC","NHWC"};
/// <summary>
/// Compute *N*-D convolution on *(N+2)*-D input.In the 2-D convolution, given input data with shape *(batch_size,channel, height, width)*, the output is computed by.. math::   out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star   weight[i,j,:,:]where :math:`\star` is the 2-D cross-correlation operator.For general 2-D convolution, the shapes are- **data**: *(batch_size, channel, height, width)*- **weight**: *(num_filter, channel, kernel[0], kernel[1])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_height, out_width)*.Define::  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1then we have::  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])If ``no_bias`` is set to be true, then the ``bias`` term is ignored.The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,width)*. We can choose other layouts such as *NHWC*.If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``evenly into *g* parts along the channel axis, and also evenly split ``weight``along the first dimension. Next compute the convolution on the *i*-th part ofthe data with the *i*-th weight part. The output is obtained by concatenating allthe *g* results.1-D convolution does not have *height* dimension but only *width* in space.- **data**: *(batch_size, channel, width)*- **weight**: *(num_filter, channel, kernel[0])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_width)*.3-D convolution adds an additional *depth* dimension besides *height* and*width*. The shapes are- **data**: *(batch_size, channel, depth, height, width)*- **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.Both ``weight`` and ``bias`` are learnable parameters.There are other options to tune the performance.- **cudnn_tune**: enable this option leads to higher startup time but may give  faster speed. Options are  - **off**: no tuning  - **limited_workspace**:run test and pick the fastest algorithm that doesn't    exceed workspace limit.  - **fastest**: pick the fastest algorithm and ignore workspace limit.  - **None** (default): the behavior is determined by environment variable    ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace    (default), 2 for fastest.- **workspace**: A large number leads to more (GPU) memory usage but may improve  the performance.Defined in G:\deeplearn\mxnet\src\operator\convolution.cc:L169
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the ConvolutionOp.</param>
/// <param name="kernel">convolution kernel size: (h, w) or (d, h, w)</param>
/// <param name="num_filter">convolution filter(channel) number</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="stride">convolution stride: (h, w) or (d, h, w)</param>
/// <param name="dilate">convolution dilate: (h, w) or (d, h, w)</param>
/// <param name="pad">pad for convolution: (h, w) or (d, h, w)</param>
/// <param name="num_group">Number of group partitions.</param>
/// <param name="workspace">Maximum temporary workspace allowed for convolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algo by running performance test.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Convolution(string symbol_name,
Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
ConvolutionCudnnTune? cudnn_tune=null,
bool cudnn_off=false,
ConvolutionLayout? layout=null)
{
return new Operator("Convolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<ConvolutionCudnnTune>(cudnn_tune,ConvolutionCudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<ConvolutionLayout>(layout,ConvolutionLayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Compute *N*-D convolution on *(N+2)*-D input.In the 2-D convolution, given input data with shape *(batch_size,channel, height, width)*, the output is computed by.. math::   out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star   weight[i,j,:,:]where :math:`\star` is the 2-D cross-correlation operator.For general 2-D convolution, the shapes are- **data**: *(batch_size, channel, height, width)*- **weight**: *(num_filter, channel, kernel[0], kernel[1])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_height, out_width)*.Define::  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1then we have::  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])If ``no_bias`` is set to be true, then the ``bias`` term is ignored.The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,width)*. We can choose other layouts such as *NHWC*.If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``evenly into *g* parts along the channel axis, and also evenly split ``weight``along the first dimension. Next compute the convolution on the *i*-th part ofthe data with the *i*-th weight part. The output is obtained by concatenating allthe *g* results.1-D convolution does not have *height* dimension but only *width* in space.- **data**: *(batch_size, channel, width)*- **weight**: *(num_filter, channel, kernel[0])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_width)*.3-D convolution adds an additional *depth* dimension besides *height* and*width*. The shapes are- **data**: *(batch_size, channel, depth, height, width)*- **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*- **bias**: *(num_filter,)*- **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.Both ``weight`` and ``bias`` are learnable parameters.There are other options to tune the performance.- **cudnn_tune**: enable this option leads to higher startup time but may give  faster speed. Options are  - **off**: no tuning  - **limited_workspace**:run test and pick the fastest algorithm that doesn't    exceed workspace limit.  - **fastest**: pick the fastest algorithm and ignore workspace limit.  - **None** (default): the behavior is determined by environment variable    ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace    (default), 2 for fastest.- **workspace**: A large number leads to more (GPU) memory usage but may improve  the performance.Defined in G:\deeplearn\mxnet\src\operator\convolution.cc:L169
/// </summary>
/// <param name="data">Input data to the ConvolutionOp.</param>
/// <param name="kernel">convolution kernel size: (h, w) or (d, h, w)</param>
/// <param name="num_filter">convolution filter(channel) number</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="stride">convolution stride: (h, w) or (d, h, w)</param>
/// <param name="dilate">convolution dilate: (h, w) or (d, h, w)</param>
/// <param name="pad">pad for convolution: (h, w) or (d, h, w)</param>
/// <param name="num_group">Number of group partitions.</param>
/// <param name="workspace">Maximum temporary workspace allowed for convolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algo by running performance test.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Convolution(Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
ConvolutionCudnnTune? cudnn_tune=null,
bool cudnn_off=false,
ConvolutionLayout? layout=null)
{
return new Operator("Convolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<ConvolutionCudnnTune>(cudnn_tune,ConvolutionCudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<ConvolutionLayout>(layout,ConvolutionLayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
private static readonly List<string> ConvolutionV1CudnnTuneConvert = new List<string>(){"fastest","limited_workspace","off"};
private static readonly List<string> ConvolutionV1LayoutConvert = new List<string>(){"NCDHW","NCHW","NDHWC","NHWC"};
/// <summary>
/// This operator is DEPRECATED. Apply convolution to input then add a bias.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the ConvolutionV1Op.</param>
/// <param name="kernel">convolution kernel size: (h, w) or (d, h, w)</param>
/// <param name="num_filter">convolution filter(channel) number</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="stride">convolution stride: (h, w) or (d, h, w)</param>
/// <param name="dilate">convolution dilate: (h, w) or (d, h, w)</param>
/// <param name="pad">pad for convolution: (h, w) or (d, h, w)</param>
/// <param name="num_group">Number of group partitions. Equivalent to slicing input into num_group    partitions, apply convolution on each, then concatenate the results</param>
/// <param name="workspace">Maximum tmp workspace allowed for convolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algo by running performance test.    Leads to higher startup time but may give faster speed. Options are:    'off': no tuning    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.    'fastest': pick the fastest algorithm and ignore workspace limit.    If set to None (default), behavior is determined by environment    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,    1 for limited workspace (default), 2 for fastest.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for    default layout: NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ConvolutionV1(string symbol_name,
Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
ConvolutionV1CudnnTune? cudnn_tune=null,
bool cudnn_off=false,
ConvolutionV1Layout? layout=null)
{
return new Operator("Convolution_v1")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<ConvolutionV1CudnnTune>(cudnn_tune,ConvolutionV1CudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<ConvolutionV1Layout>(layout,ConvolutionV1LayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// This operator is DEPRECATED. Apply convolution to input then add a bias.
/// </summary>
/// <param name="data">Input data to the ConvolutionV1Op.</param>
/// <param name="kernel">convolution kernel size: (h, w) or (d, h, w)</param>
/// <param name="num_filter">convolution filter(channel) number</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="stride">convolution stride: (h, w) or (d, h, w)</param>
/// <param name="dilate">convolution dilate: (h, w) or (d, h, w)</param>
/// <param name="pad">pad for convolution: (h, w) or (d, h, w)</param>
/// <param name="num_group">Number of group partitions. Equivalent to slicing input into num_group    partitions, apply convolution on each, then concatenate the results</param>
/// <param name="workspace">Maximum tmp workspace allowed for convolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algo by running performance test.    Leads to higher startup time but may give faster speed. Options are:    'off': no tuning    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.    'fastest': pick the fastest algorithm and ignore workspace limit.    If set to None (default), behavior is determined by environment    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,    1 for limited workspace (default), 2 for fastest.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for    default layout: NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ConvolutionV1(Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
ConvolutionV1CudnnTune? cudnn_tune=null,
bool cudnn_off=false,
ConvolutionV1Layout? layout=null)
{
return new Operator("Convolution_v1")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<ConvolutionV1CudnnTune>(cudnn_tune,ConvolutionV1CudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<ConvolutionV1Layout>(layout,ConvolutionV1LayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
/// <summary>
/// Applies correlation to inputs.The correlation layer performs multiplicative patch comparisons between two feature maps.Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and:math:`x_{2}` in the second map is then defined as:.. math::   c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>for a square patch of size :math:`K:=2k+1`.Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with otherdata. For this reason, it has no training weights.Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhoodcentered around :math:`x_{1}`.The final output is defined by the following expression:.. math::  out[n, q, i, j] = c(x_{i, j}, x_{q})where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.Defined in G:\deeplearn\mxnet\src\operator\correlation.cc:L191
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data1">Input data1 to the correlation.</param>
/// <param name="data2">Input data2 to the correlation.</param>
/// <param name="kernel_size">kernel size for Correlation must be an odd number</param>
/// <param name="max_displacement">Max displacement of Correlation </param>
/// <param name="stride1">stride1 quantize data1 globally</param>
/// <param name="stride2">stride2 quantize data2 within the neighborhood centered around data1</param>
/// <param name="pad_size">pad for Correlation</param>
/// <param name="is_multiply">operation type is either multiplication or subduction</param>
 /// <returns>returns new symbol</returns>
public static Symbol Correlation(string symbol_name,
Symbol data1,
Symbol data2,
int kernel_size=1,
int max_displacement=1,
int stride1=1,
int stride2=1,
int pad_size=0,
bool is_multiply=true)
{
return new Operator("Correlation")
.SetParam("kernel_size", kernel_size)
.SetParam("max_displacement", max_displacement)
.SetParam("stride1", stride1)
.SetParam("stride2", stride2)
.SetParam("pad_size", pad_size)
.SetParam("is_multiply", is_multiply)
.SetInput("data1", data1)
.SetInput("data2", data2)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies correlation to inputs.The correlation layer performs multiplicative patch comparisons between two feature maps.Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and:math:`x_{2}` in the second map is then defined as:.. math::   c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>for a square patch of size :math:`K:=2k+1`.Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with otherdata. For this reason, it has no training weights.Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhoodcentered around :math:`x_{1}`.The final output is defined by the following expression:.. math::  out[n, q, i, j] = c(x_{i, j}, x_{q})where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.Defined in G:\deeplearn\mxnet\src\operator\correlation.cc:L191
/// </summary>
/// <param name="data1">Input data1 to the correlation.</param>
/// <param name="data2">Input data2 to the correlation.</param>
/// <param name="kernel_size">kernel size for Correlation must be an odd number</param>
/// <param name="max_displacement">Max displacement of Correlation </param>
/// <param name="stride1">stride1 quantize data1 globally</param>
/// <param name="stride2">stride2 quantize data2 within the neighborhood centered around data1</param>
/// <param name="pad_size">pad for Correlation</param>
/// <param name="is_multiply">operation type is either multiplication or subduction</param>
 /// <returns>returns new symbol</returns>
public static Symbol Correlation(Symbol data1,
Symbol data2,
int kernel_size=1,
int max_displacement=1,
int stride1=1,
int stride2=1,
int pad_size=0,
bool is_multiply=true)
{
return new Operator("Correlation")
.SetParam("kernel_size", kernel_size)
.SetParam("max_displacement", max_displacement)
.SetParam("stride1", stride1)
.SetParam("stride2", stride2)
.SetParam("pad_size", pad_size)
.SetParam("is_multiply", is_multiply)
.SetInput("data1", data1)
.SetInput("data2", data2)
.CreateSymbol();
}
/// <summary>
/// .. note:: `Crop` is deprecated. Use `slice` instead.Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w orwith width and height of the second input symbol, i.e., with one input, we need h_w tospecify the crop height and width, otherwise the second input symbol's size will be usedDefined in G:\deeplearn\mxnet\src\operator\crop.cc:L49
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Tensor or List of Tensors, the second input will be used as crop_like shape reference</param>
/// <param name="num_args">Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here</param>
/// <param name="offset">crop offset coordinate: (y, x)</param>
/// <param name="h_w">crop height and width: (h, w)</param>
/// <param name="center_crop">If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like</param>
 /// <returns>returns new symbol</returns>
public static Symbol Crop(string symbol_name,
Symbol data,
int num_args,
Shape offset=null,
Shape h_w=null,
bool center_crop=false)
{if(offset==null){ offset= new Shape(0,0);}
if(h_w==null){ h_w= new Shape(0,0);}

return new Operator("Crop")
.SetParam("data", data)
.SetParam("num_args", num_args)
.SetParam("offset", offset)
.SetParam("h_w", h_w)
.SetParam("center_crop", center_crop)
.CreateSymbol(symbol_name);
}
/// <summary>
/// .. note:: `Crop` is deprecated. Use `slice` instead.Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w orwith width and height of the second input symbol, i.e., with one input, we need h_w tospecify the crop height and width, otherwise the second input symbol's size will be usedDefined in G:\deeplearn\mxnet\src\operator\crop.cc:L49
/// </summary>
/// <param name="data">Tensor or List of Tensors, the second input will be used as crop_like shape reference</param>
/// <param name="num_args">Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here</param>
/// <param name="offset">crop offset coordinate: (y, x)</param>
/// <param name="h_w">crop height and width: (h, w)</param>
/// <param name="center_crop">If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like</param>
 /// <returns>returns new symbol</returns>
public static Symbol Crop(Symbol data,
int num_args,
Shape offset=null,
Shape h_w=null,
bool center_crop=false)
{if(offset==null){ offset= new Shape(0,0);}
if(h_w==null){ h_w= new Shape(0,0);}

return new Operator("Crop")
.SetParam("data", data)
.SetParam("num_args", num_args)
.SetParam("offset", offset)
.SetParam("h_w", h_w)
.SetParam("center_crop", center_crop)
.CreateSymbol();
}
private static readonly List<string> DeconvolutionCudnnTuneConvert = new List<string>(){"fastest","limited_workspace","off"};
private static readonly List<string> DeconvolutionLayoutConvert = new List<string>(){"NCDHW","NCHW","NCW","NDHWC","NHWC"};
/// <summary>
/// Computes 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input tensor to the deconvolution operation.</param>
/// <param name="kernel">Deconvolution kernel size: (h, w) or (d, h, w). This is same as the kernel size used for the corresponding convolution</param>
/// <param name="num_filter">Number of output filters.</param>
/// <param name="weight">Weights representing the kernel.</param>
/// <param name="bias">Bias added to the result after the deconvolution operation.</param>
/// <param name="stride">The stride used for the corresponding convolution: (h, w) or (d, h, w).</param>
/// <param name="dilate">Dilation factor for each dimension of the input: (h, w) or (d, h, w).</param>
/// <param name="pad">The amount of implicit zero padding added during convolution for each dimension of the input: (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice. If `target_shape` is set, `pad` will be ignored and a padding that will generate the target shape will be used.</param>
/// <param name="adj">Adjustment for output shape: (h, w) or (d, h, w). If `target_shape` is set, `adj` will be ignored and computed accordingly.</param>
/// <param name="target_shape">Shape of the output tensor: (h, w) or (d, h, w).</param>
/// <param name="num_group">Number of groups partition.</param>
/// <param name="workspace">Maximum temporal workspace allowed for deconvolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algorithm by running performance test.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Deconvolution(string symbol_name,
Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
Shape adj=null,
Shape target_shape=null,
int num_group=1,
long workspace=512,
bool no_bias=true,
DeconvolutionCudnnTune? cudnn_tune=null,
bool cudnn_off=false,
DeconvolutionLayout? layout=null)
{
return new Operator("Deconvolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("adj", adj)
.SetParam("target_shape", target_shape)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<DeconvolutionCudnnTune>(cudnn_tune,DeconvolutionCudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<DeconvolutionLayout>(layout,DeconvolutionLayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.
/// </summary>
/// <param name="data">Input tensor to the deconvolution operation.</param>
/// <param name="kernel">Deconvolution kernel size: (h, w) or (d, h, w). This is same as the kernel size used for the corresponding convolution</param>
/// <param name="num_filter">Number of output filters.</param>
/// <param name="weight">Weights representing the kernel.</param>
/// <param name="bias">Bias added to the result after the deconvolution operation.</param>
/// <param name="stride">The stride used for the corresponding convolution: (h, w) or (d, h, w).</param>
/// <param name="dilate">Dilation factor for each dimension of the input: (h, w) or (d, h, w).</param>
/// <param name="pad">The amount of implicit zero padding added during convolution for each dimension of the input: (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice. If `target_shape` is set, `pad` will be ignored and a padding that will generate the target shape will be used.</param>
/// <param name="adj">Adjustment for output shape: (h, w) or (d, h, w). If `target_shape` is set, `adj` will be ignored and computed accordingly.</param>
/// <param name="target_shape">Shape of the output tensor: (h, w) or (d, h, w).</param>
/// <param name="num_group">Number of groups partition.</param>
/// <param name="workspace">Maximum temporal workspace allowed for deconvolution (MB).</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="cudnn_tune">Whether to pick convolution algorithm by running performance test.</param>
/// <param name="cudnn_off">Turn off cudnn for this layer.</param>
/// <param name="layout">Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Deconvolution(Symbol data,
Shape kernel,
int num_filter,
Symbol weight=null,
Symbol bias=null,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
Shape adj=null,
Shape target_shape=null,
int num_group=1,
long workspace=512,
bool no_bias=true,
DeconvolutionCudnnTune? cudnn_tune=null,
bool cudnn_off=false,
DeconvolutionLayout? layout=null)
{
return new Operator("Deconvolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("adj", adj)
.SetParam("target_shape", target_shape)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", Util.EnumToString<DeconvolutionCudnnTune>(cudnn_tune,DeconvolutionCudnnTuneConvert))
.SetParam("cudnn_off", cudnn_off)
.SetParam("layout", Util.EnumToString<DeconvolutionLayout>(layout,DeconvolutionLayoutConvert))
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
private static readonly List<string> DropoutModeConvert = new List<string>(){"always","training"};
/// <summary>
/// Applies dropout operation to input array.- During training, each element of the input is set to zero with probability p.  The whole array is rescaled by :math:`1/(1-p)` to keep the expected  sum of the input unchanged.- During testing, this operator does not change the input if mode is 'training'.  If mode is 'always', the same computaion as during training will be applied.Example::  random.seed(998)  input_array = array([[3., 0.5,  -0.5,  2., 7.],                      [2., -0.4,   7.,  3., 0.2]])  a = symbol.Variable('a')  dropout = symbol.Dropout(a, p = 0.2)  executor = dropout.simple_bind(a = input_array.shape)  ## If training  executor.forward(is_train = True, a = input_array)  executor.outputs  [[ 3.75   0.625 -0.     2.5    8.75 ]   [ 2.5   -0.5    8.75   3.75   0.   ]]  ## If testing  executor.forward(is_train = False, a = input_array)  executor.outputs  [[ 3.     0.5   -0.5    2.     7.   ]   [ 2.    -0.4    7.     3.     0.2  ]]Defined in G:\deeplearn\mxnet\src\operator\dropout.cc:L77
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array to which dropout will be applied.</param>
/// <param name="p">Fraction of the input that gets dropped out during training time.</param>
/// <param name="mode">Whether to only turn on dropout during training or to also turn on for inference.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Dropout(string symbol_name,
Symbol data,
float p=0.5f,
DropoutMode mode=DropoutMode.Training)
{
return new Operator("Dropout")
.SetParam("p", p)
.SetParam("mode", Util.EnumToString<DropoutMode>(mode,DropoutModeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies dropout operation to input array.- During training, each element of the input is set to zero with probability p.  The whole array is rescaled by :math:`1/(1-p)` to keep the expected  sum of the input unchanged.- During testing, this operator does not change the input if mode is 'training'.  If mode is 'always', the same computaion as during training will be applied.Example::  random.seed(998)  input_array = array([[3., 0.5,  -0.5,  2., 7.],                      [2., -0.4,   7.,  3., 0.2]])  a = symbol.Variable('a')  dropout = symbol.Dropout(a, p = 0.2)  executor = dropout.simple_bind(a = input_array.shape)  ## If training  executor.forward(is_train = True, a = input_array)  executor.outputs  [[ 3.75   0.625 -0.     2.5    8.75 ]   [ 2.5   -0.5    8.75   3.75   0.   ]]  ## If testing  executor.forward(is_train = False, a = input_array)  executor.outputs  [[ 3.     0.5   -0.5    2.     7.   ]   [ 2.    -0.4    7.     3.     0.2  ]]Defined in G:\deeplearn\mxnet\src\operator\dropout.cc:L77
/// </summary>
/// <param name="data">Input array to which dropout will be applied.</param>
/// <param name="p">Fraction of the input that gets dropped out during training time.</param>
/// <param name="mode">Whether to only turn on dropout during training or to also turn on for inference.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Dropout(Symbol data,
float p=0.5f,
DropoutMode mode=DropoutMode.Training)
{
return new Operator("Dropout")
.SetParam("p", p)
.SetParam("mode", Util.EnumToString<DropoutMode>(mode,DropoutModeConvert))
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Applies a linear transformation: :math:`Y = XW^T + b`.If ``flatten`` is set to be true, then the shapes are:- **data**: `(batch_size, x1, x2, ..., xn)`- **weight**: `(num_hidden, x1 * x2 * ... * xn)`- **bias**: `(num_hidden,)`- **out**: `(batch_size, num_hidden)`If ``flatten`` is set to be false, then the shapes are:- **data**: `(x1, x2, ..., xn, input_dim)`- **weight**: `(num_hidden, input_dim)`- **bias**: `(num_hidden,)`- **out**: `(x1, x2, ..., xn, num_hidden)`The learnable parameters include both ``weight`` and ``bias``.If ``no_bias`` is set to be true, then the ``bias`` term is ignored.Defined in G:\deeplearn\mxnet\src\operator\fully_connected.cc:L97
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data.</param>
/// <param name="num_hidden">Number of hidden nodes of the output.</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="flatten">Whether to collapse all but the first axis of the input data tensor.</param>
 /// <returns>returns new symbol</returns>
public static Symbol FullyConnected(string symbol_name,
Symbol data,
int num_hidden,
Symbol weight=null,
Symbol bias=null,
bool no_bias=false,
bool flatten=true)
{
return new Operator("FullyConnected")
.SetParam("num_hidden", num_hidden)
.SetParam("no_bias", no_bias)
.SetParam("flatten", flatten)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies a linear transformation: :math:`Y = XW^T + b`.If ``flatten`` is set to be true, then the shapes are:- **data**: `(batch_size, x1, x2, ..., xn)`- **weight**: `(num_hidden, x1 * x2 * ... * xn)`- **bias**: `(num_hidden,)`- **out**: `(batch_size, num_hidden)`If ``flatten`` is set to be false, then the shapes are:- **data**: `(x1, x2, ..., xn, input_dim)`- **weight**: `(num_hidden, input_dim)`- **bias**: `(num_hidden,)`- **out**: `(x1, x2, ..., xn, num_hidden)`The learnable parameters include both ``weight`` and ``bias``.If ``no_bias`` is set to be true, then the ``bias`` term is ignored.Defined in G:\deeplearn\mxnet\src\operator\fully_connected.cc:L97
/// </summary>
/// <param name="data">Input data.</param>
/// <param name="num_hidden">Number of hidden nodes of the output.</param>
/// <param name="weight">Weight matrix.</param>
/// <param name="bias">Bias parameter.</param>
/// <param name="no_bias">Whether to disable bias parameter.</param>
/// <param name="flatten">Whether to collapse all but the first axis of the input data tensor.</param>
 /// <returns>returns new symbol</returns>
public static Symbol FullyConnected(Symbol data,
int num_hidden,
Symbol weight=null,
Symbol bias=null,
bool no_bias=false,
bool flatten=true)
{
return new Operator("FullyConnected")
.SetParam("num_hidden", num_hidden)
.SetParam("no_bias", no_bias)
.SetParam("flatten", flatten)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
private static readonly List<string> GridgeneratorTransformTypeConvert = new List<string>(){"affine","warp"};
/// <summary>
/// Generates 2D sampling grid for bilinear sampling.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the function.</param>
/// <param name="transform_type">The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).</param>
/// <param name="target_shape">Specifies the output shape (H, W). This is required if transformation type is `affine`. If transformation type is `warp`, this parameter is ignored.</param>
 /// <returns>returns new symbol</returns>
public static Symbol GridGenerator(string symbol_name,
Symbol data,
GridgeneratorTransformType transform_type,
Shape target_shape=null)
{if(target_shape==null){ target_shape= new Shape(0,0);}

return new Operator("GridGenerator")
.SetParam("transform_type", Util.EnumToString<GridgeneratorTransformType>(transform_type,GridgeneratorTransformTypeConvert))
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Generates 2D sampling grid for bilinear sampling.
/// </summary>
/// <param name="data">Input data to the function.</param>
/// <param name="transform_type">The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).</param>
/// <param name="target_shape">Specifies the output shape (H, W). This is required if transformation type is `affine`. If transformation type is `warp`, this parameter is ignored.</param>
 /// <returns>returns new symbol</returns>
public static Symbol GridGenerator(Symbol data,
GridgeneratorTransformType transform_type,
Shape target_shape=null)
{if(target_shape==null){ target_shape= new Shape(0,0);}

return new Operator("GridGenerator")
.SetParam("transform_type", Util.EnumToString<GridgeneratorTransformType>(transform_type,GridgeneratorTransformTypeConvert))
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Applies instance normalization to the n-dimensional input array.This operator takes an n-dimensional input array where (n>2) and normalizesthe input using the following formula:.. math::  out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + betaThis layer is similar to batch normalization layer (`BatchNorm`)with two differences: first, the normalization iscarried out per example (instance), not over a batch. Second, thesame normalization is applied both at test and train time. Thisoperation is also known as `contrast normalization`.If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],`gamma` and `beta` parameters must be vectors of shape [channel].This implementation is based on paper:.. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,   D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).Examples::  // Input of shape (2,1,2)  x = [[[ 1.1,  2.2]],       [[ 3.3,  4.4]]]  // gamma parameter of length 1  gamma = [1.5]  // beta parameter of length 1  beta = [0.5]  // Instance normalization is calculated with the above formula  InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],                                [[-0.99752653,  1.99752724]]]Defined in G:\deeplearn\mxnet\src\operator\instance_norm.cc:L94
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">An n-dimensional input array (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].</param>
/// <param name="gamma">A vector of length 'channel', which multiplies the normalized input.</param>
/// <param name="beta">A vector of length 'channel', which is added to the product of the normalized input and the weight.</param>
/// <param name="eps">An `epsilon` parameter to prevent division by 0.</param>
 /// <returns>returns new symbol</returns>
public static Symbol InstanceNorm(string symbol_name,
Symbol data,
Symbol gamma,
Symbol beta,
float eps=0.001f)
{
return new Operator("InstanceNorm")
.SetParam("eps", eps)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies instance normalization to the n-dimensional input array.This operator takes an n-dimensional input array where (n>2) and normalizesthe input using the following formula:.. math::  out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + betaThis layer is similar to batch normalization layer (`BatchNorm`)with two differences: first, the normalization iscarried out per example (instance), not over a batch. Second, thesame normalization is applied both at test and train time. Thisoperation is also known as `contrast normalization`.If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],`gamma` and `beta` parameters must be vectors of shape [channel].This implementation is based on paper:.. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,   D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).Examples::  // Input of shape (2,1,2)  x = [[[ 1.1,  2.2]],       [[ 3.3,  4.4]]]  // gamma parameter of length 1  gamma = [1.5]  // beta parameter of length 1  beta = [0.5]  // Instance normalization is calculated with the above formula  InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],                                [[-0.99752653,  1.99752724]]]Defined in G:\deeplearn\mxnet\src\operator\instance_norm.cc:L94
/// </summary>
/// <param name="data">An n-dimensional input array (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].</param>
/// <param name="gamma">A vector of length 'channel', which multiplies the normalized input.</param>
/// <param name="beta">A vector of length 'channel', which is added to the product of the normalized input and the weight.</param>
/// <param name="eps">An `epsilon` parameter to prevent division by 0.</param>
 /// <returns>returns new symbol</returns>
public static Symbol InstanceNorm(Symbol data,
Symbol gamma,
Symbol beta,
float eps=0.001f)
{
return new Operator("InstanceNorm")
.SetParam("eps", eps)
.SetInput("data", data)
.SetInput("gamma", gamma)
.SetInput("beta", beta)
.CreateSymbol();
}
private static readonly List<string> L2normalizationModeConvert = new List<string>(){"channel","instance","spatial"};
/// <summary>
/// Normalize the input array using the L2 norm.For 1-D NDArray, it computes::  out = data / sqrt(sum(data ** 2) + eps)For N-D NDArray, if the input array has shape (N, N, ..., N),with ``mode`` = ``instance``, it normalizes each instance in the multidimensionalarray by its L2 norm.::  for i in 0...N    out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::  for i in 0...N    out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)with ``mode`` = ``spatial``, it normalizes the cross channel norm for each positionin the array by its L2 norm.::  for dim in 2...N    for i in 0...N      out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)          -dim-Example::  x = [[[1,2],        [3,4]],       [[2,2],        [5,6]]]  L2Normalization(x, mode='instance')  =[[[ 0.18257418  0.36514837]     [ 0.54772252  0.73029673]]    [[ 0.24077171  0.24077171]     [ 0.60192931  0.72231513]]]  L2Normalization(x, mode='channel')  =[[[ 0.31622776  0.44721359]     [ 0.94868326  0.89442718]]    [[ 0.37139067  0.31622776]     [ 0.92847669  0.94868326]]]  L2Normalization(x, mode='spatial')  =[[[ 0.44721359  0.89442718]     [ 0.60000002  0.80000001]]    [[ 0.70710677  0.70710677]     [ 0.6401844   0.76822126]]]Defined in G:\deeplearn\mxnet\src\operator\l2_normalization.cc:L92
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array to normalize.</param>
/// <param name="eps">A small constant for numerical stability.</param>
/// <param name="mode">Specify the dimension along which to compute L2 norm.</param>
 /// <returns>returns new symbol</returns>
public static Symbol L2Normalization(string symbol_name,
Symbol data,
float eps=1e-10f,
L2normalizationMode mode=L2normalizationMode.Instance)
{
return new Operator("L2Normalization")
.SetParam("eps", eps)
.SetParam("mode", Util.EnumToString<L2normalizationMode>(mode,L2normalizationModeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Normalize the input array using the L2 norm.For 1-D NDArray, it computes::  out = data / sqrt(sum(data ** 2) + eps)For N-D NDArray, if the input array has shape (N, N, ..., N),with ``mode`` = ``instance``, it normalizes each instance in the multidimensionalarray by its L2 norm.::  for i in 0...N    out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::  for i in 0...N    out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)with ``mode`` = ``spatial``, it normalizes the cross channel norm for each positionin the array by its L2 norm.::  for dim in 2...N    for i in 0...N      out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)          -dim-Example::  x = [[[1,2],        [3,4]],       [[2,2],        [5,6]]]  L2Normalization(x, mode='instance')  =[[[ 0.18257418  0.36514837]     [ 0.54772252  0.73029673]]    [[ 0.24077171  0.24077171]     [ 0.60192931  0.72231513]]]  L2Normalization(x, mode='channel')  =[[[ 0.31622776  0.44721359]     [ 0.94868326  0.89442718]]    [[ 0.37139067  0.31622776]     [ 0.92847669  0.94868326]]]  L2Normalization(x, mode='spatial')  =[[[ 0.44721359  0.89442718]     [ 0.60000002  0.80000001]]    [[ 0.70710677  0.70710677]     [ 0.6401844   0.76822126]]]Defined in G:\deeplearn\mxnet\src\operator\l2_normalization.cc:L92
/// </summary>
/// <param name="data">Input array to normalize.</param>
/// <param name="eps">A small constant for numerical stability.</param>
/// <param name="mode">Specify the dimension along which to compute L2 norm.</param>
 /// <returns>returns new symbol</returns>
public static Symbol L2Normalization(Symbol data,
float eps=1e-10f,
L2normalizationMode mode=L2normalizationMode.Instance)
{
return new Operator("L2Normalization")
.SetParam("eps", eps)
.SetParam("mode", Util.EnumToString<L2normalizationMode>(mode,L2normalizationModeConvert))
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Applies local response normalization to the input.The local response normalization layer performs "lateral inhibition" by normalizingover local input regions.If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalizedactivity :math:`b_{x,y}^{i}` is given by the expression:.. math::   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the totalnumber of kernels in the layer.Defined in G:\deeplearn\mxnet\src\operator\lrn.cc:L72
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data.</param>
/// <param name="nsize">normalization window width in elements.</param>
/// <param name="alpha">The variance scaling parameter :math:`lpha` in the LRN expression.</param>
/// <param name="beta">The power parameter :math:`eta` in the LRN expression.</param>
/// <param name="knorm">The parameter :math:`k` in the LRN expression.</param>
 /// <returns>returns new symbol</returns>
public static Symbol LRN(string symbol_name,
Symbol data,
int nsize,
float alpha=0.0001f,
float beta=0.75f,
float knorm=2f)
{
return new Operator("LRN")
.SetParam("nsize", nsize)
.SetParam("alpha", alpha)
.SetParam("beta", beta)
.SetParam("knorm", knorm)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies local response normalization to the input.The local response normalization layer performs "lateral inhibition" by normalizingover local input regions.If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalizedactivity :math:`b_{x,y}^{i}` is given by the expression:.. math::   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the totalnumber of kernels in the layer.Defined in G:\deeplearn\mxnet\src\operator\lrn.cc:L72
/// </summary>
/// <param name="data">Input data.</param>
/// <param name="nsize">normalization window width in elements.</param>
/// <param name="alpha">The variance scaling parameter :math:`lpha` in the LRN expression.</param>
/// <param name="beta">The power parameter :math:`eta` in the LRN expression.</param>
/// <param name="knorm">The parameter :math:`k` in the LRN expression.</param>
 /// <returns>returns new symbol</returns>
public static Symbol LRN(Symbol data,
int nsize,
float alpha=0.0001f,
float beta=0.75f,
float knorm=2f)
{
return new Operator("LRN")
.SetParam("nsize", nsize)
.SetParam("alpha", alpha)
.SetParam("beta", beta)
.SetParam("knorm", knorm)
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> MakelossNormalizationConvert = new List<string>(){"batch","null","valid"};
/// <summary>
/// Make your own loss function in network construction.This operator accepts a customized loss function symbol as a terminal loss andthe symbol should be an operator with no backward dependency.The output of this function is the gradient of loss with respect to the input data.For example, if you are a making a cross entropy loss function. Assume ``out`` is thepredicted output and ``label`` is the true label, then the cross entropy can be defined as::  cross_entropy = label * log(out) + (1 - label) * log(1 - out)  loss = MakeLoss(cross_entropy)We will need to use ``MakeLoss`` when we are creating our own loss function or we want tocombine multiple loss functions. Also we may want to stop some variables' gradientsfrom backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.In addition, we can give a scale to the loss by setting ``grad_scale``,so that the gradient of the loss will be rescaled in the backpropagation... note:: This operator should be used as a Symbol instead of NDArray.Defined in G:\deeplearn\mxnet\src\operator\make_loss.cc:L70
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
/// <param name="grad_scale">Gradient scale as a supplement to unary and binary operators</param>
/// <param name="valid_thresh">clip each element in the array to 0 when it is less than ``valid_thresh``. This is used when ``normalization`` is set to ``'valid'``.</param>
/// <param name="normalization">If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.</param>
 /// <returns>returns new symbol</returns>
public static Symbol MakeLoss(string symbol_name,
Symbol data,
float grad_scale=1f,
float valid_thresh=0f,
MakelossNormalization normalization=MakelossNormalization.Null)
{
return new Operator("MakeLoss")
.SetParam("grad_scale", grad_scale)
.SetParam("valid_thresh", valid_thresh)
.SetParam("normalization", Util.EnumToString<MakelossNormalization>(normalization,MakelossNormalizationConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Make your own loss function in network construction.This operator accepts a customized loss function symbol as a terminal loss andthe symbol should be an operator with no backward dependency.The output of this function is the gradient of loss with respect to the input data.For example, if you are a making a cross entropy loss function. Assume ``out`` is thepredicted output and ``label`` is the true label, then the cross entropy can be defined as::  cross_entropy = label * log(out) + (1 - label) * log(1 - out)  loss = MakeLoss(cross_entropy)We will need to use ``MakeLoss`` when we are creating our own loss function or we want tocombine multiple loss functions. Also we may want to stop some variables' gradientsfrom backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.In addition, we can give a scale to the loss by setting ``grad_scale``,so that the gradient of the loss will be rescaled in the backpropagation... note:: This operator should be used as a Symbol instead of NDArray.Defined in G:\deeplearn\mxnet\src\operator\make_loss.cc:L70
/// </summary>
/// <param name="data">Input array.</param>
/// <param name="grad_scale">Gradient scale as a supplement to unary and binary operators</param>
/// <param name="valid_thresh">clip each element in the array to 0 when it is less than ``valid_thresh``. This is used when ``normalization`` is set to ``'valid'``.</param>
/// <param name="normalization">If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.</param>
 /// <returns>returns new symbol</returns>
public static Symbol MakeLoss(Symbol data,
float grad_scale=1f,
float valid_thresh=0f,
MakelossNormalization normalization=MakelossNormalization.Null)
{
return new Operator("MakeLoss")
.SetParam("grad_scale", grad_scale)
.SetParam("valid_thresh", valid_thresh)
.SetParam("normalization", Util.EnumToString<MakelossNormalization>(normalization,MakelossNormalizationConvert))
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> PoolingPoolTypeConvert = new List<string>(){"avg","max","sum"};
private static readonly List<string> PoolingPoolingConventionConvert = new List<string>(){"full","valid"};
/// <summary>
/// Performs pooling on the input.The shapes for 1-D pooling are- **data**: *(batch_size, channel, width)*,- **out**: *(batch_size, num_filter, out_width)*.The shapes for 2-D pooling are- **data**: *(batch_size, channel, height, width)*- **out**: *(batch_size, num_filter, out_height, out_width)*, with::    out_height = f(height, kernel[0], pad[0], stride[0])    out_width = f(width, kernel[1], pad[1], stride[1])The definition of *f* depends on ``pooling_convention``, which has two options:- **valid** (default)::    f(x, k, p, s) = floor((x+2*p-k)/s)+1- **full**, which is compatible with Caffe::    f(x, k, p, s) = ceil((x+2*p-k)/s)+1But ``global_pool`` is set to be true, then do a global pooling, namely reset``kernel=(height, width)``.Three pooling options are supported by ``pool_type``:- **avg**: average pooling- **max**: max pooling- **sum**: sum poolingFor 3-D pooling, an additional *depth* dimension is added before*height*. Namely the input data will have shape *(batch_size, channel, depth,height, width)*.Defined in G:\deeplearn\mxnet\src\operator\pooling.cc:L134
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the pooling operator.</param>
/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>
/// <param name="pool_type">Pooling type to be applied.</param>
/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
/// <param name="cudnn_off">Turn off cudnn pooling and use MXNet pooling operator. </param>
/// <param name="pooling_convention">Pooling convention to be applied.</param>
/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>
/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pooling(string symbol_name,
Symbol data,
Shape kernel,
PoolingPoolType pool_type,
bool global_pool=false,
bool cudnn_off=false,
PoolingPoolingConvention pooling_convention=PoolingPoolingConvention.Valid,
Shape stride=null,
Shape pad=null)
{
return new Operator("Pooling")
.SetParam("kernel", kernel)
.SetParam("pool_type", Util.EnumToString<PoolingPoolType>(pool_type,PoolingPoolTypeConvert))
.SetParam("global_pool", global_pool)
.SetParam("cudnn_off", cudnn_off)
.SetParam("pooling_convention", Util.EnumToString<PoolingPoolingConvention>(pooling_convention,PoolingPoolingConventionConvert))
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Performs pooling on the input.The shapes for 1-D pooling are- **data**: *(batch_size, channel, width)*,- **out**: *(batch_size, num_filter, out_width)*.The shapes for 2-D pooling are- **data**: *(batch_size, channel, height, width)*- **out**: *(batch_size, num_filter, out_height, out_width)*, with::    out_height = f(height, kernel[0], pad[0], stride[0])    out_width = f(width, kernel[1], pad[1], stride[1])The definition of *f* depends on ``pooling_convention``, which has two options:- **valid** (default)::    f(x, k, p, s) = floor((x+2*p-k)/s)+1- **full**, which is compatible with Caffe::    f(x, k, p, s) = ceil((x+2*p-k)/s)+1But ``global_pool`` is set to be true, then do a global pooling, namely reset``kernel=(height, width)``.Three pooling options are supported by ``pool_type``:- **avg**: average pooling- **max**: max pooling- **sum**: sum poolingFor 3-D pooling, an additional *depth* dimension is added before*height*. Namely the input data will have shape *(batch_size, channel, depth,height, width)*.Defined in G:\deeplearn\mxnet\src\operator\pooling.cc:L134
/// </summary>
/// <param name="data">Input data to the pooling operator.</param>
/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>
/// <param name="pool_type">Pooling type to be applied.</param>
/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
/// <param name="cudnn_off">Turn off cudnn pooling and use MXNet pooling operator. </param>
/// <param name="pooling_convention">Pooling convention to be applied.</param>
/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>
/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol Pooling(Symbol data,
Shape kernel,
PoolingPoolType pool_type,
bool global_pool=false,
bool cudnn_off=false,
PoolingPoolingConvention pooling_convention=PoolingPoolingConvention.Valid,
Shape stride=null,
Shape pad=null)
{
return new Operator("Pooling")
.SetParam("kernel", kernel)
.SetParam("pool_type", Util.EnumToString<PoolingPoolType>(pool_type,PoolingPoolTypeConvert))
.SetParam("global_pool", global_pool)
.SetParam("cudnn_off", cudnn_off)
.SetParam("pooling_convention", Util.EnumToString<PoolingPoolingConvention>(pooling_convention,PoolingPoolingConventionConvert))
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> PoolingV1PoolTypeConvert = new List<string>(){"avg","max","sum"};
private static readonly List<string> PoolingV1PoolingConventionConvert = new List<string>(){"full","valid"};
/// <summary>
/// This operator is DEPRECATED.Perform pooling on the input.The shapes for 2-D pooling is- **data**: *(batch_size, channel, height, width)*- **out**: *(batch_size, num_filter, out_height, out_width)*, with::    out_height = f(height, kernel[0], pad[0], stride[0])    out_width = f(width, kernel[1], pad[1], stride[1])The definition of *f* depends on ``pooling_convention``, which has two options:- **valid** (default)::    f(x, k, p, s) = floor((x+2*p-k)/s)+1- **full**, which is compatible with Caffe::    f(x, k, p, s) = ceil((x+2*p-k)/s)+1But ``global_pool`` is set to be true, then do a global pooling, namely reset``kernel=(height, width)``.Three pooling options are supported by ``pool_type``:- **avg**: average pooling- **max**: max pooling- **sum**: sum pooling1-D pooling is special case of 2-D pooling with *weight=1* and*kernel[1]=1*.For 3-D pooling, an additional *depth* dimension is added before*height*. Namely the input data will have shape *(batch_size, channel, depth,height, width)*.Defined in G:\deeplearn\mxnet\src\operator\pooling_v1.cc:L103
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the pooling operator.</param>
/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>
/// <param name="pool_type">Pooling type to be applied.</param>
/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
/// <param name="pooling_convention">Pooling convention to be applied.</param>
/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>
/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol PoolingV1(string symbol_name,
Symbol data,
Shape kernel,
PoolingV1PoolType pool_type,
bool global_pool=false,
PoolingV1PoolingConvention pooling_convention=PoolingV1PoolingConvention.Valid,
Shape stride=null,
Shape pad=null)
{
return new Operator("Pooling_v1")
.SetParam("kernel", kernel)
.SetParam("pool_type", Util.EnumToString<PoolingV1PoolType>(pool_type,PoolingV1PoolTypeConvert))
.SetParam("global_pool", global_pool)
.SetParam("pooling_convention", Util.EnumToString<PoolingV1PoolingConvention>(pooling_convention,PoolingV1PoolingConventionConvert))
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// This operator is DEPRECATED.Perform pooling on the input.The shapes for 2-D pooling is- **data**: *(batch_size, channel, height, width)*- **out**: *(batch_size, num_filter, out_height, out_width)*, with::    out_height = f(height, kernel[0], pad[0], stride[0])    out_width = f(width, kernel[1], pad[1], stride[1])The definition of *f* depends on ``pooling_convention``, which has two options:- **valid** (default)::    f(x, k, p, s) = floor((x+2*p-k)/s)+1- **full**, which is compatible with Caffe::    f(x, k, p, s) = ceil((x+2*p-k)/s)+1But ``global_pool`` is set to be true, then do a global pooling, namely reset``kernel=(height, width)``.Three pooling options are supported by ``pool_type``:- **avg**: average pooling- **max**: max pooling- **sum**: sum pooling1-D pooling is special case of 2-D pooling with *weight=1* and*kernel[1]=1*.For 3-D pooling, an additional *depth* dimension is added before*height*. Namely the input data will have shape *(batch_size, channel, depth,height, width)*.Defined in G:\deeplearn\mxnet\src\operator\pooling_v1.cc:L103
/// </summary>
/// <param name="data">Input data to the pooling operator.</param>
/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>
/// <param name="pool_type">Pooling type to be applied.</param>
/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
/// <param name="pooling_convention">Pooling convention to be applied.</param>
/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>
/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol PoolingV1(Symbol data,
Shape kernel,
PoolingV1PoolType pool_type,
bool global_pool=false,
PoolingV1PoolingConvention pooling_convention=PoolingV1PoolingConvention.Valid,
Shape stride=null,
Shape pad=null)
{
return new Operator("Pooling_v1")
.SetParam("kernel", kernel)
.SetParam("pool_type", Util.EnumToString<PoolingV1PoolType>(pool_type,PoolingV1PoolTypeConvert))
.SetParam("global_pool", global_pool)
.SetParam("pooling_convention", Util.EnumToString<PoolingV1PoolingConvention>(pooling_convention,PoolingV1PoolingConventionConvert))
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Computes and optimizes for squared loss during backward propagation.Just outputs ``data`` during forward propagation.If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,then the squared loss estimated over :math:`n` samples is defined as:math:`\text{SquaredLoss}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left( y_i - \hat{y}_i \right)^2`.. note::   Use the LinearRegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L69
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol LinearRegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("LinearRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes and optimizes for squared loss during backward propagation.Just outputs ``data`` during forward propagation.If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,then the squared loss estimated over :math:`n` samples is defined as:math:`\text{SquaredLoss}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left( y_i - \hat{y}_i \right)^2`.. note::   Use the LinearRegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L69
/// </summary>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol LinearRegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("LinearRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Computes mean absolute error of the input.MAE is a risk metric corresponding to the expected value of the absolute error.If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,then the mean absolute error (MAE) estimated over :math:`n` samples is defined as:math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i - \hat{y}_i \right|`.. note::   Use the MAERegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L90
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol MAERegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("MAERegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes mean absolute error of the input.MAE is a risk metric corresponding to the expected value of the absolute error.If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,then the mean absolute error (MAE) estimated over :math:`n` samples is defined as:math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i - \hat{y}_i \right|`.. note::   Use the MAERegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L90
/// </summary>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol MAERegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("MAERegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Applies a logistic function to the input.The logistic function, also known as the sigmoid function, is computed as:math:`\frac{1}{1+exp(-x)}`.Commonly, the sigmoid is used to squash the real-valued output of a linear model:math:wTx+b into the [0,1] range so that it can be interpreted as a probability.It is suitable for binary classification or probability prediction tasks... note::   Use the LogisticRegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L111
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol LogisticRegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("LogisticRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies a logistic function to the input.The logistic function, also known as the sigmoid function, is computed as:math:`\frac{1}{1+exp(-x)}`.Commonly, the sigmoid is used to squash the real-valued output of a linear model:math:wTx+b into the [0,1] range so that it can be interpreted as a probability.It is suitable for binary classification or probability prediction tasks... note::   Use the LogisticRegressionOutput as the final output layer of a net.By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.Defined in G:\deeplearn\mxnet\src\operator\regression_output.cc:L111
/// </summary>
/// <param name="data">Input data to the function.</param>
/// <param name="label">Input label to the function.</param>
/// <param name="grad_scale">Scale the gradient by a float factor</param>
 /// <returns>returns new symbol</returns>
public static Symbol LogisticRegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{
return new Operator("LogisticRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
private static readonly List<string> RNNModeConvert = new List<string>(){"gru","lstm","rnn_relu","rnn_tanh"};
/// <summary>
/// Applies a recurrent layer to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to RNN</param>
/// <param name="parameters">Vector of all RNN trainable parameters concatenated</param>
/// <param name="state">initial hidden state of the RNN</param>
/// <param name="state_cell">initial cell state for LSTM networks (only for LSTM)</param>
/// <param name="state_size">size of the state for each layer</param>
/// <param name="num_layers">number of stacked layers</param>
/// <param name="mode">the type of RNN to compute</param>
/// <param name="bidirectional">whether to use bidirectional recurrent layers</param>
/// <param name="p">Dropout probability, fraction of the input that gets dropped out at training time</param>
/// <param name="state_outputs">Whether to have the states as symbol outputs.</param>
 /// <returns>returns new symbol</returns>
public static Symbol RNN(string symbol_name,
Symbol data,
Symbol parameters,
Symbol state,
Symbol state_cell,
int state_size,
int num_layers,
RNNMode mode,
bool bidirectional=false,
float p=0f,
bool state_outputs=false)
{
return new Operator("RNN")
.SetParam("state_size", state_size)
.SetParam("num_layers", num_layers)
.SetParam("mode", Util.EnumToString<RNNMode>(mode,RNNModeConvert))
.SetParam("bidirectional", bidirectional)
.SetParam("p", p)
.SetParam("state_outputs", state_outputs)
.SetInput("data", data)
.SetInput("parameters", parameters)
.SetInput("state", state)
.SetInput("state_cell", state_cell)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies a recurrent layer to input.
/// </summary>
/// <param name="data">Input data to RNN</param>
/// <param name="parameters">Vector of all RNN trainable parameters concatenated</param>
/// <param name="state">initial hidden state of the RNN</param>
/// <param name="state_cell">initial cell state for LSTM networks (only for LSTM)</param>
/// <param name="state_size">size of the state for each layer</param>
/// <param name="num_layers">number of stacked layers</param>
/// <param name="mode">the type of RNN to compute</param>
/// <param name="bidirectional">whether to use bidirectional recurrent layers</param>
/// <param name="p">Dropout probability, fraction of the input that gets dropped out at training time</param>
/// <param name="state_outputs">Whether to have the states as symbol outputs.</param>
 /// <returns>returns new symbol</returns>
public static Symbol RNN(Symbol data,
Symbol parameters,
Symbol state,
Symbol state_cell,
int state_size,
int num_layers,
RNNMode mode,
bool bidirectional=false,
float p=0f,
bool state_outputs=false)
{
return new Operator("RNN")
.SetParam("state_size", state_size)
.SetParam("num_layers", num_layers)
.SetParam("mode", Util.EnumToString<RNNMode>(mode,RNNModeConvert))
.SetParam("bidirectional", bidirectional)
.SetParam("p", p)
.SetParam("state_outputs", state_outputs)
.SetInput("data", data)
.SetInput("parameters", parameters)
.SetInput("state", state)
.SetInput("state_cell", state_cell)
.CreateSymbol();
}
/// <summary>
/// Performs region of interest(ROI) pooling on the input array.ROI pooling is a variant of a max pooling layer, in which the output size is fixed andregion of interest is a parameter. Its purpose is to perform max pooling on the inputsof non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-netlayer mostly used in training a `Fast R-CNN` network for object detection.This operator takes a 4D feature map as an input array and region proposals as `rois`,then it pools over sub-regions of input and produces a fixed-sized output arrayregardless of the ROI size.To crop the feature map accordingly, you can resize the bounding box coordinatesby changing the parameters `rois` and `spatial_scale`.The cropped feature maps are pooled by standard max pooling operation to a fixed size outputindicated by a `pooled_size` parameter. batch_size will change to the number of regionbounding boxes after `ROIPooling`.The size of each region of interest doesn't have to be perfectly divisible bythe number of pooling sections(`pooled_size`).Example::  x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],         [  6.,   7.,   8.,   9.,  10.,  11.],         [ 12.,  13.,  14.,  15.,  16.,  17.],         [ 18.,  19.,  20.,  21.,  22.,  23.],         [ 24.,  25.,  26.,  27.,  28.,  29.],         [ 30.,  31.,  32.,  33.,  34.,  35.],         [ 36.,  37.,  38.,  39.,  40.,  41.],         [ 42.,  43.,  44.,  45.,  46.,  47.]]]]  // region of interest i.e. bounding box coordinates.  y = [[0,0,0,4,4]]  // returns array of shape (2,2) according to the given roi with max pooling.  ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],                                    [ 26.,  28.]]]]  // region of interest is changed due to the change in `spacial_scale` parameter.  ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],                                    [ 19.,  21.]]]]Defined in G:\deeplearn\mxnet\src\operator\roi_pooling.cc:L287
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">The input array to the pooling operator,  a 4D Feature maps </param>
/// <param name="rois">Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right corners of designated region of interest. `batch_index` indicates the index of corresponding image in the input array</param>
/// <param name="pooled_size">ROI pooling output shape (h,w) </param>
/// <param name="spatial_scale">Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers</param>
 /// <returns>returns new symbol</returns>
public static Symbol ROIPooling(string symbol_name,
Symbol data,
Symbol rois,
Shape pooled_size,
float spatial_scale)
{
return new Operator("ROIPooling")
.SetParam("pooled_size", pooled_size)
.SetParam("spatial_scale", spatial_scale)
.SetInput("data", data)
.SetInput("rois", rois)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Performs region of interest(ROI) pooling on the input array.ROI pooling is a variant of a max pooling layer, in which the output size is fixed andregion of interest is a parameter. Its purpose is to perform max pooling on the inputsof non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-netlayer mostly used in training a `Fast R-CNN` network for object detection.This operator takes a 4D feature map as an input array and region proposals as `rois`,then it pools over sub-regions of input and produces a fixed-sized output arrayregardless of the ROI size.To crop the feature map accordingly, you can resize the bounding box coordinatesby changing the parameters `rois` and `spatial_scale`.The cropped feature maps are pooled by standard max pooling operation to a fixed size outputindicated by a `pooled_size` parameter. batch_size will change to the number of regionbounding boxes after `ROIPooling`.The size of each region of interest doesn't have to be perfectly divisible bythe number of pooling sections(`pooled_size`).Example::  x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],         [  6.,   7.,   8.,   9.,  10.,  11.],         [ 12.,  13.,  14.,  15.,  16.,  17.],         [ 18.,  19.,  20.,  21.,  22.,  23.],         [ 24.,  25.,  26.,  27.,  28.,  29.],         [ 30.,  31.,  32.,  33.,  34.,  35.],         [ 36.,  37.,  38.,  39.,  40.,  41.],         [ 42.,  43.,  44.,  45.,  46.,  47.]]]]  // region of interest i.e. bounding box coordinates.  y = [[0,0,0,4,4]]  // returns array of shape (2,2) according to the given roi with max pooling.  ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],                                    [ 26.,  28.]]]]  // region of interest is changed due to the change in `spacial_scale` parameter.  ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],                                    [ 19.,  21.]]]]Defined in G:\deeplearn\mxnet\src\operator\roi_pooling.cc:L287
/// </summary>
/// <param name="data">The input array to the pooling operator,  a 4D Feature maps </param>
/// <param name="rois">Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right corners of designated region of interest. `batch_index` indicates the index of corresponding image in the input array</param>
/// <param name="pooled_size">ROI pooling output shape (h,w) </param>
/// <param name="spatial_scale">Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers</param>
 /// <returns>returns new symbol</returns>
public static Symbol ROIPooling(Symbol data,
Symbol rois,
Shape pooled_size,
float spatial_scale)
{
return new Operator("ROIPooling")
.SetParam("pooled_size", pooled_size)
.SetParam("spatial_scale", spatial_scale)
.SetInput("data", data)
.SetInput("rois", rois)
.CreateSymbol();
}
/// <summary>
/// Takes the last element of a sequence.This function takes an n-dimensional input array of the form[max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional arrayof the form [batch_size, other_feature_dims].Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should bean input array of positive ints of dimension [batch_size]. To use this parameter,set `use_sequence_length` to `True`, otherwise each example in the batch is assumedto have the max sequence length... note:: Alternatively, you can also use `take` operator.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.],         [  7.,   8.,   9.]],        [[ 10.,   11.,   12.],         [ 13.,   14.,   15.],         [ 16.,   17.,   18.]],        [[  19.,   20.,   21.],         [  22.,   23.,   24.],         [  25.,   26.,   27.]]]   // returns last sequence when sequence_length parameter is not used   SequenceLast(x) = [[  19.,   20.,   21.],                      [  22.,   23.,   24.],                      [  25.,   26.,   27.]]   // sequence_length y is used   SequenceLast(x, y=[1,1,1], use_sequence_length=True) =            [[  1.,   2.,   3.],             [  4.,   5.,   6.],             [  7.,   8.,   9.]]   // sequence_length y is used   SequenceLast(x, y=[1,2,3], use_sequence_length=True) =            [[  1.,    2.,   3.],             [  13.,  14.,  15.],             [  25.,  26.,  27.]]Defined in G:\deeplearn\mxnet\src\operator\sequence_last.cc:L91
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2</param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceLast(string symbol_name,
Symbol data,
Symbol sequence_length,
bool use_sequence_length=false)
{
return new Operator("SequenceLast")
.SetParam("use_sequence_length", use_sequence_length)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Takes the last element of a sequence.This function takes an n-dimensional input array of the form[max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional arrayof the form [batch_size, other_feature_dims].Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should bean input array of positive ints of dimension [batch_size]. To use this parameter,set `use_sequence_length` to `True`, otherwise each example in the batch is assumedto have the max sequence length... note:: Alternatively, you can also use `take` operator.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.],         [  7.,   8.,   9.]],        [[ 10.,   11.,   12.],         [ 13.,   14.,   15.],         [ 16.,   17.,   18.]],        [[  19.,   20.,   21.],         [  22.,   23.,   24.],         [  25.,   26.,   27.]]]   // returns last sequence when sequence_length parameter is not used   SequenceLast(x) = [[  19.,   20.,   21.],                      [  22.,   23.,   24.],                      [  25.,   26.,   27.]]   // sequence_length y is used   SequenceLast(x, y=[1,1,1], use_sequence_length=True) =            [[  1.,   2.,   3.],             [  4.,   5.,   6.],             [  7.,   8.,   9.]]   // sequence_length y is used   SequenceLast(x, y=[1,2,3], use_sequence_length=True) =            [[  1.,    2.,   3.],             [  13.,  14.,  15.],             [  25.,  26.,  27.]]Defined in G:\deeplearn\mxnet\src\operator\sequence_last.cc:L91
/// </summary>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2</param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceLast(Symbol data,
Symbol sequence_length,
bool use_sequence_length=false)
{
return new Operator("SequenceLast")
.SetParam("use_sequence_length", use_sequence_length)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol();
}
/// <summary>
/// Sets all elements outside the sequence to a constant value.This function takes an n-dimensional input array of the form[max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`should be an input array of positive ints of dimension [batch_size].To use this parameter, set `use_sequence_length` to `True`,otherwise each example in the batch is assumed to have the max sequence length andthis operator works as the `identity` operator.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.]],        [[  7.,   8.,   9.],         [ 10.,  11.,  12.]],        [[ 13.,  14.,   15.],         [ 16.,  17.,   18.]]]   // Batch 1   B1 = [[  1.,   2.,   3.],         [  7.,   8.,   9.],         [ 13.,  14.,  15.]]   // Batch 2   B2 = [[  4.,   5.,   6.],         [ 10.,  11.,  12.],         [ 16.,  17.,  18.]]   // works as identity operator when sequence_length parameter is not used   SequenceMask(x) = [[[  1.,   2.,   3.],                       [  4.,   5.,   6.]],                      [[  7.,   8.,   9.],                       [ 10.,  11.,  12.]],                      [[ 13.,  14.,   15.],                       [ 16.,  17.,   18.]]]   // sequence_length [1,1] means 1 of each batch will be kept   // and other rows are masked with default mask value = 0   SequenceMask(x, y=[1,1], use_sequence_length=True) =                [[[  1.,   2.,   3.],                  [  4.,   5.,   6.]],                 [[  0.,   0.,   0.],                  [  0.,   0.,   0.]],                 [[  0.,   0.,   0.],                  [  0.,   0.,   0.]]]   // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept   // and other rows are masked with value = 1   SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =                [[[  1.,   2.,   3.],                  [  4.,   5.,   6.]],                 [[  7.,   8.,   9.],                  [  10.,  11.,  12.]],                 [[   1.,   1.,   1.],                  [  16.,  17.,  18.]]]Defined in G:\deeplearn\mxnet\src\operator\sequence_mask.cc:L113
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2</param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
/// <param name="value">The value to be used as a mask.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceMask(string symbol_name,
Symbol data,
Symbol sequence_length,
bool use_sequence_length=false,
float value=0f)
{
return new Operator("SequenceMask")
.SetParam("use_sequence_length", use_sequence_length)
.SetParam("value", value)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Sets all elements outside the sequence to a constant value.This function takes an n-dimensional input array of the form[max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`should be an input array of positive ints of dimension [batch_size].To use this parameter, set `use_sequence_length` to `True`,otherwise each example in the batch is assumed to have the max sequence length andthis operator works as the `identity` operator.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.]],        [[  7.,   8.,   9.],         [ 10.,  11.,  12.]],        [[ 13.,  14.,   15.],         [ 16.,  17.,   18.]]]   // Batch 1   B1 = [[  1.,   2.,   3.],         [  7.,   8.,   9.],         [ 13.,  14.,  15.]]   // Batch 2   B2 = [[  4.,   5.,   6.],         [ 10.,  11.,  12.],         [ 16.,  17.,  18.]]   // works as identity operator when sequence_length parameter is not used   SequenceMask(x) = [[[  1.,   2.,   3.],                       [  4.,   5.,   6.]],                      [[  7.,   8.,   9.],                       [ 10.,  11.,  12.]],                      [[ 13.,  14.,   15.],                       [ 16.,  17.,   18.]]]   // sequence_length [1,1] means 1 of each batch will be kept   // and other rows are masked with default mask value = 0   SequenceMask(x, y=[1,1], use_sequence_length=True) =                [[[  1.,   2.,   3.],                  [  4.,   5.,   6.]],                 [[  0.,   0.,   0.],                  [  0.,   0.,   0.]],                 [[  0.,   0.,   0.],                  [  0.,   0.,   0.]]]   // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept   // and other rows are masked with value = 1   SequenceMask(x, y=[2,3], use_sequence_length=True, value=1) =                [[[  1.,   2.,   3.],                  [  4.,   5.,   6.]],                 [[  7.,   8.,   9.],                  [  10.,  11.,  12.]],                 [[   1.,   1.,   1.],                  [  16.,  17.,  18.]]]Defined in G:\deeplearn\mxnet\src\operator\sequence_mask.cc:L113
/// </summary>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2</param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
/// <param name="value">The value to be used as a mask.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceMask(Symbol data,
Symbol sequence_length,
bool use_sequence_length=false,
float value=0f)
{
return new Operator("SequenceMask")
.SetParam("use_sequence_length", use_sequence_length)
.SetParam("value", value)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol();
}
/// <summary>
/// Reverses the elements of each sequence.This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]and returns an array of the same shape.Parameter `sequence_length` is used to handle variable-length sequences.`sequence_length` should be an input array of positive ints of dimension [batch_size].To use this parameter, set `use_sequence_length` to `True`,otherwise each example in the batch is assumed to have the max sequence length.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.]],        [[  7.,   8.,   9.],         [ 10.,  11.,  12.]],        [[ 13.,  14.,   15.],         [ 16.,  17.,   18.]]]   // Batch 1   B1 = [[  1.,   2.,   3.],         [  7.,   8.,   9.],         [ 13.,  14.,  15.]]   // Batch 2   B2 = [[  4.,   5.,   6.],         [ 10.,  11.,  12.],         [ 16.,  17.,  18.]]   // returns reverse sequence when sequence_length parameter is not used   SequenceReverse(x) = [[[ 13.,  14.,   15.],                          [ 16.,  17.,   18.]],                         [[  7.,   8.,   9.],                          [ 10.,  11.,  12.]],                         [[  1.,   2.,   3.],                          [  4.,   5.,   6.]]]   // sequence_length [2,2] means 2 rows of   // both batch B1 and B2 will be reversed.   SequenceReverse(x, y=[2,2], use_sequence_length=True) =                     [[[  7.,   8.,   9.],                       [ 10.,  11.,  12.]],                      [[  1.,   2.,   3.],                       [  4.,   5.,   6.]],                      [[ 13.,  14.,   15.],                       [ 16.,  17.,   18.]]]   // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3   // will be reversed.   SequenceReverse(x, y=[2,3], use_sequence_length=True) =                    [[[  7.,   8.,   9.],                      [ 16.,  17.,  18.]],                     [[  1.,   2.,   3.],                      [ 10.,  11.,  12.]],                     [[ 13.,  14,   15.],                      [  4.,   5.,   6.]]]Defined in G:\deeplearn\mxnet\src\operator\sequence_reverse.cc:L112
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other dims] where n>2 </param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceReverse(string symbol_name,
Symbol data,
Symbol sequence_length,
bool use_sequence_length=false)
{
return new Operator("SequenceReverse")
.SetParam("use_sequence_length", use_sequence_length)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Reverses the elements of each sequence.This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]and returns an array of the same shape.Parameter `sequence_length` is used to handle variable-length sequences.`sequence_length` should be an input array of positive ints of dimension [batch_size].To use this parameter, set `use_sequence_length` to `True`,otherwise each example in the batch is assumed to have the max sequence length.Example::   x = [[[  1.,   2.,   3.],         [  4.,   5.,   6.]],        [[  7.,   8.,   9.],         [ 10.,  11.,  12.]],        [[ 13.,  14.,   15.],         [ 16.,  17.,   18.]]]   // Batch 1   B1 = [[  1.,   2.,   3.],         [  7.,   8.,   9.],         [ 13.,  14.,  15.]]   // Batch 2   B2 = [[  4.,   5.,   6.],         [ 10.,  11.,  12.],         [ 16.,  17.,  18.]]   // returns reverse sequence when sequence_length parameter is not used   SequenceReverse(x) = [[[ 13.,  14.,   15.],                          [ 16.,  17.,   18.]],                         [[  7.,   8.,   9.],                          [ 10.,  11.,  12.]],                         [[  1.,   2.,   3.],                          [  4.,   5.,   6.]]]   // sequence_length [2,2] means 2 rows of   // both batch B1 and B2 will be reversed.   SequenceReverse(x, y=[2,2], use_sequence_length=True) =                     [[[  7.,   8.,   9.],                       [ 10.,  11.,  12.]],                      [[  1.,   2.,   3.],                       [  4.,   5.,   6.]],                      [[ 13.,  14.,   15.],                       [ 16.,  17.,   18.]]]   // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3   // will be reversed.   SequenceReverse(x, y=[2,3], use_sequence_length=True) =                    [[[  7.,   8.,   9.],                      [ 16.,  17.,  18.]],                     [[  1.,   2.,   3.],                      [ 10.,  11.,  12.]],                     [[ 13.,  14,   15.],                      [  4.,   5.,   6.]]]Defined in G:\deeplearn\mxnet\src\operator\sequence_reverse.cc:L112
/// </summary>
/// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other dims] where n>2 </param>
/// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
/// <param name="use_sequence_length">If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence</param>
 /// <returns>returns new symbol</returns>
public static Symbol SequenceReverse(Symbol data,
Symbol sequence_length,
bool use_sequence_length=false)
{
return new Operator("SequenceReverse")
.SetParam("use_sequence_length", use_sequence_length)
.SetInput("data", data)
.SetInput("sequence_length", sequence_length)
.CreateSymbol();
}
private static readonly List<string> SoftmaxactivationModeConvert = new List<string>(){"channel","instance"};
/// <summary>
/// Applies softmax activation to input. This is intended for internal layers... note::  This operator has been deprecated, please use `softmax`.If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.This is the default mode.If `mode` = ``channel``, this operator will compute a k-class softmax at each positionof each instance, where `k` = ``num_channel``. This mode can only be used when the input arrayhas at least 3 dimensions.This can be used for `fully convolutional network`, `image segmentation`, etc.Example::  >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],  >>>                            [2., -.4, 7.,   3., 0.2]])  >>> softmax_act = mx.nd.SoftmaxActivation(input_array)  >>> print softmax_act.asnumpy()  [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]   [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]Defined in G:\deeplearn\mxnet\src\operator\softmax_activation.cc:L66
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array to activation function.</param>
/// <param name="mode">Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxActivation(string symbol_name,
Symbol data,
SoftmaxactivationMode mode=SoftmaxactivationMode.Instance)
{
return new Operator("SoftmaxActivation")
.SetParam("mode", Util.EnumToString<SoftmaxactivationMode>(mode,SoftmaxactivationModeConvert))
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies softmax activation to input. This is intended for internal layers... note::  This operator has been deprecated, please use `softmax`.If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.This is the default mode.If `mode` = ``channel``, this operator will compute a k-class softmax at each positionof each instance, where `k` = ``num_channel``. This mode can only be used when the input arrayhas at least 3 dimensions.This can be used for `fully convolutional network`, `image segmentation`, etc.Example::  >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],  >>>                            [2., -.4, 7.,   3., 0.2]])  >>> softmax_act = mx.nd.SoftmaxActivation(input_array)  >>> print softmax_act.asnumpy()  [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]   [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]Defined in G:\deeplearn\mxnet\src\operator\softmax_activation.cc:L66
/// </summary>
/// <param name="data">Input array to activation function.</param>
/// <param name="mode">Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxActivation(Symbol data,
SoftmaxactivationMode mode=SoftmaxactivationMode.Instance)
{
return new Operator("SoftmaxActivation")
.SetParam("mode", Util.EnumToString<SoftmaxactivationMode>(mode,SoftmaxactivationModeConvert))
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> SoftmaxoutputNormalizationConvert = new List<string>(){"batch","null","valid"};
/// <summary>
/// Computes the gradient of cross entropy loss with respect to softmax output.- This operator computes the gradient in two steps.  The cross entropy loss does not actually need to be computed.  - Applies softmax function on the input array.  - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.- The softmax function, cross entropy loss and gradient is given by:  - Softmax Function:    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}  - Cross Entropy Function:    .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)  - The gradient of cross entropy loss w.r.t softmax output:    .. math:: \text{gradient} = \text{output} - \text{label}- During forward propagation, the softmax function is computed for each instance in the input array.  For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is  :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`  and `multi_output` to specify the way to compute softmax:  - By default, `preserve_shape` is ``false``. This operator will reshape the input array    into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for    each row in the reshaped array, and afterwards reshape it back to the original shape    :math:`(d_1, d_2, ..., d_n)`.  - If `preserve_shape` is ``true``, the softmax function will be computed along    the last axis (`axis` = ``-1``).  - If `multi_output` is ``true``, the softmax function will be computed along    the second axis (`axis` = ``1``).- During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.  The provided label can be a one-hot label array or a probability label array.  - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances    with a particular label to be ignored during backward propagation. **This has no effect when    softmax `output` has same shape as `label`**.    Example::      data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]      label = [1,0,2,3]      ignore_label = 1      SoftmaxOutput(data=data, label = label,\                    multi_output=true, use_ignore=true,\                    ignore_label=ignore_label)      ## forward softmax output      [[ 0.0320586   0.08714432  0.23688284  0.64391428]       [ 0.25        0.25        0.25        0.25      ]       [ 0.25        0.25        0.25        0.25      ]       [ 0.25        0.25        0.25        0.25      ]]      ## backward gradient output      [[ 0.    0.    0.    0.  ]       [-0.75  0.25  0.25  0.25]       [ 0.25  0.25 -0.75  0.25]       [ 0.25  0.25  0.25 -0.75]]      ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.  - The parameter `grad_scale` can be used to rescale the gradient, which is often used to    give each loss function different weights.  - This operator also supports various ways to normalize the gradient by `normalization`,    The `normalization` is applied if softmax output has different shape than the labels.    The `normalization` mode can be set to the followings:    - ``'null'``: do nothing.    - ``'batch'``: divide the gradient by the batch size.    - ``'valid'``: divide the gradient by the number of instances which are not ignored.Defined in G:\deeplearn\mxnet\src\operator\softmax_output.cc:L122
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
/// <param name="label">Ground truth label.</param>
/// <param name="grad_scale">Scales the gradient by a float factor.</param>
/// <param name="ignore_label">The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).</param>
/// <param name="multi_output">If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.</param>
/// <param name="use_ignore">If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.</param>
/// <param name="preserve_shape">If set to ``true``, the softmax function will be computed along the last axis (``-1``).</param>
/// <param name="normalization">Normalizes the gradient.</param>
/// <param name="out_grad">Multiplies gradient with output gradient element-wise.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
bool preserve_shape=false,
SoftmaxoutputNormalization normalization=SoftmaxoutputNormalization.Null,
bool out_grad=false)
{
return new Operator("SoftmaxOutput")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("preserve_shape", preserve_shape)
.SetParam("normalization", Util.EnumToString<SoftmaxoutputNormalization>(normalization,SoftmaxoutputNormalizationConvert))
.SetParam("out_grad", out_grad)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes the gradient of cross entropy loss with respect to softmax output.- This operator computes the gradient in two steps.  The cross entropy loss does not actually need to be computed.  - Applies softmax function on the input array.  - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.- The softmax function, cross entropy loss and gradient is given by:  - Softmax Function:    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}  - Cross Entropy Function:    .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)  - The gradient of cross entropy loss w.r.t softmax output:    .. math:: \text{gradient} = \text{output} - \text{label}- During forward propagation, the softmax function is computed for each instance in the input array.  For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is  :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`  and `multi_output` to specify the way to compute softmax:  - By default, `preserve_shape` is ``false``. This operator will reshape the input array    into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for    each row in the reshaped array, and afterwards reshape it back to the original shape    :math:`(d_1, d_2, ..., d_n)`.  - If `preserve_shape` is ``true``, the softmax function will be computed along    the last axis (`axis` = ``-1``).  - If `multi_output` is ``true``, the softmax function will be computed along    the second axis (`axis` = ``1``).- During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.  The provided label can be a one-hot label array or a probability label array.  - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances    with a particular label to be ignored during backward propagation. **This has no effect when    softmax `output` has same shape as `label`**.    Example::      data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]      label = [1,0,2,3]      ignore_label = 1      SoftmaxOutput(data=data, label = label,\                    multi_output=true, use_ignore=true,\                    ignore_label=ignore_label)      ## forward softmax output      [[ 0.0320586   0.08714432  0.23688284  0.64391428]       [ 0.25        0.25        0.25        0.25      ]       [ 0.25        0.25        0.25        0.25      ]       [ 0.25        0.25        0.25        0.25      ]]      ## backward gradient output      [[ 0.    0.    0.    0.  ]       [-0.75  0.25  0.25  0.25]       [ 0.25  0.25 -0.75  0.25]       [ 0.25  0.25  0.25 -0.75]]      ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.  - The parameter `grad_scale` can be used to rescale the gradient, which is often used to    give each loss function different weights.  - This operator also supports various ways to normalize the gradient by `normalization`,    The `normalization` is applied if softmax output has different shape than the labels.    The `normalization` mode can be set to the followings:    - ``'null'``: do nothing.    - ``'batch'``: divide the gradient by the batch size.    - ``'valid'``: divide the gradient by the number of instances which are not ignored.Defined in G:\deeplearn\mxnet\src\operator\softmax_output.cc:L122
/// </summary>
/// <param name="data">Input array.</param>
/// <param name="label">Ground truth label.</param>
/// <param name="grad_scale">Scales the gradient by a float factor.</param>
/// <param name="ignore_label">The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).</param>
/// <param name="multi_output">If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.</param>
/// <param name="use_ignore">If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.</param>
/// <param name="preserve_shape">If set to ``true``, the softmax function will be computed along the last axis (``-1``).</param>
/// <param name="normalization">Normalizes the gradient.</param>
/// <param name="out_grad">Multiplies gradient with output gradient element-wise.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SoftmaxOutput(Symbol data,
Symbol label,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
bool preserve_shape=false,
SoftmaxoutputNormalization normalization=SoftmaxoutputNormalization.Null,
bool out_grad=false)
{
return new Operator("SoftmaxOutput")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("preserve_shape", preserve_shape)
.SetParam("normalization", Util.EnumToString<SoftmaxoutputNormalization>(normalization,SoftmaxoutputNormalizationConvert))
.SetParam("out_grad", out_grad)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
private static readonly List<string> SoftmaxNormalizationConvert = new List<string>(){"batch","null","valid"};
/// <summary>
/// Please use `SoftmaxOutput`... note::  This operator has been renamed to `SoftmaxOutput`, which  computes the gradient of cross-entropy loss w.r.t softmax output.  To just compute softmax output, use the `softmax` operator.Defined in G:\deeplearn\mxnet\src\operator\softmax_output.cc:L137
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input array.</param>
/// <param name="grad_scale">Scales the gradient by a float factor.</param>
/// <param name="ignore_label">The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).</param>
/// <param name="multi_output">If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.</param>
/// <param name="use_ignore">If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.</param>
/// <param name="preserve_shape">If set to ``true``, the softmax function will be computed along the last axis (``-1``).</param>
/// <param name="normalization">Normalizes the gradient.</param>
/// <param name="out_grad">Multiplies gradient with output gradient element-wise.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Softmax(string symbol_name,
Symbol data,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
bool preserve_shape=false,
SoftmaxNormalization normalization=SoftmaxNormalization.Null,
bool out_grad=false)
{
return new Operator("Softmax")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("preserve_shape", preserve_shape)
.SetParam("normalization", Util.EnumToString<SoftmaxNormalization>(normalization,SoftmaxNormalizationConvert))
.SetParam("out_grad", out_grad)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Please use `SoftmaxOutput`... note::  This operator has been renamed to `SoftmaxOutput`, which  computes the gradient of cross-entropy loss w.r.t softmax output.  To just compute softmax output, use the `softmax` operator.Defined in G:\deeplearn\mxnet\src\operator\softmax_output.cc:L137
/// </summary>
/// <param name="data">Input array.</param>
/// <param name="grad_scale">Scales the gradient by a float factor.</param>
/// <param name="ignore_label">The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).</param>
/// <param name="multi_output">If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.</param>
/// <param name="use_ignore">If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.</param>
/// <param name="preserve_shape">If set to ``true``, the softmax function will be computed along the last axis (``-1``).</param>
/// <param name="normalization">Normalizes the gradient.</param>
/// <param name="out_grad">Multiplies gradient with output gradient element-wise.</param>
 /// <returns>returns new symbol</returns>
public static Symbol Softmax(Symbol data,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
bool preserve_shape=false,
SoftmaxNormalization normalization=SoftmaxNormalization.Null,
bool out_grad=false)
{
return new Operator("Softmax")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("preserve_shape", preserve_shape)
.SetParam("normalization", Util.EnumToString<SoftmaxNormalization>(normalization,SoftmaxNormalizationConvert))
.SetParam("out_grad", out_grad)
.SetInput("data", data)
.CreateSymbol();
}
private static readonly List<string> SpatialtransformerTransformTypeConvert = new List<string>(){"affine"};
private static readonly List<string> SpatialtransformerSamplerTypeConvert = new List<string>(){"bilinear"};
/// <summary>
/// Applies a spatial transformer to input feature map.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data to the SpatialTransformerOp.</param>
/// <param name="loc">localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the weight and bias with identity tranform.</param>
/// <param name="transform_type">transformation type</param>
/// <param name="sampler_type">sampling type</param>
/// <param name="target_shape">output shape(h, w) of spatial transformer: (y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol SpatialTransformer(string symbol_name,
Symbol data,
Symbol loc,
SpatialtransformerTransformType transform_type,
SpatialtransformerSamplerType sampler_type,
Shape target_shape=null)
{if(target_shape==null){ target_shape= new Shape(0,0);}

return new Operator("SpatialTransformer")
.SetParam("transform_type", Util.EnumToString<SpatialtransformerTransformType>(transform_type,SpatialtransformerTransformTypeConvert))
.SetParam("sampler_type", Util.EnumToString<SpatialtransformerSamplerType>(sampler_type,SpatialtransformerSamplerTypeConvert))
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.SetInput("loc", loc)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Applies a spatial transformer to input feature map.
/// </summary>
/// <param name="data">Input data to the SpatialTransformerOp.</param>
/// <param name="loc">localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the weight and bias with identity tranform.</param>
/// <param name="transform_type">transformation type</param>
/// <param name="sampler_type">sampling type</param>
/// <param name="target_shape">output shape(h, w) of spatial transformer: (y, x)</param>
 /// <returns>returns new symbol</returns>
public static Symbol SpatialTransformer(Symbol data,
Symbol loc,
SpatialtransformerTransformType transform_type,
SpatialtransformerSamplerType sampler_type,
Shape target_shape=null)
{if(target_shape==null){ target_shape= new Shape(0,0);}

return new Operator("SpatialTransformer")
.SetParam("transform_type", Util.EnumToString<SpatialtransformerTransformType>(transform_type,SpatialtransformerTransformTypeConvert))
.SetParam("sampler_type", Util.EnumToString<SpatialtransformerSamplerType>(sampler_type,SpatialtransformerSamplerTypeConvert))
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.SetInput("loc", loc)
.CreateSymbol();
}
/// <summary>
/// Computes support vector machine based transformation of the input.This tutorial demonstrates using SVM as output layer for classification instead of softmax:https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="data">Input data for SVM transformation.</param>
/// <param name="label">Class label for the input data.</param>
/// <param name="margin">The loss function penalizes outputs that lie outside this margin. Default margin is 1.</param>
/// <param name="regularization_coefficient">Regularization parameter for the SVM. This balances the tradeoff between coefficient size and error.</param>
/// <param name="use_linear">Whether to use L1-SVM objective. L2-SVM objective is used by default.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SVMOutput(string symbol_name,
Symbol data,
Symbol label,
float margin=1f,
float regularization_coefficient=1f,
bool use_linear=false)
{
return new Operator("SVMOutput")
.SetParam("margin", margin)
.SetParam("regularization_coefficient", regularization_coefficient)
.SetParam("use_linear", use_linear)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Computes support vector machine based transformation of the input.This tutorial demonstrates using SVM as output layer for classification instead of softmax:https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
/// </summary>
/// <param name="data">Input data for SVM transformation.</param>
/// <param name="label">Class label for the input data.</param>
/// <param name="margin">The loss function penalizes outputs that lie outside this margin. Default margin is 1.</param>
/// <param name="regularization_coefficient">Regularization parameter for the SVM. This balances the tradeoff between coefficient size and error.</param>
/// <param name="use_linear">Whether to use L1-SVM objective. L2-SVM objective is used by default.</param>
 /// <returns>returns new symbol</returns>
public static Symbol SVMOutput(Symbol data,
Symbol label,
float margin=1f,
float regularization_coefficient=1f,
bool use_linear=false)
{
return new Operator("SVMOutput")
.SetParam("margin", margin)
.SetParam("regularization_coefficient", regularization_coefficient)
.SetParam("use_linear", use_linear)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">Left operand to the function.</param>
/// <param name="rhs">Right operand to the function.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ChooseElement0Index(string symbol_name,
Symbol lhs,
Symbol rhs)
{
return new Operator("choose_element_0index")
.SetParam("lhs", lhs)
.SetParam("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
/// </summary>
/// <param name="lhs">Left operand to the function.</param>
/// <param name="rhs">Right operand to the function.</param>
 /// <returns>returns new symbol</returns>
public static Symbol ChooseElement0Index(Symbol lhs,
Symbol rhs)
{
return new Operator("choose_element_0index")
.SetParam("lhs", lhs)
.SetParam("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>
/// <param name="lhs">Left operand to the function.</param>
/// <param name="mhs">Middle operand to the function.</param>
/// <param name="rhs">Right operand to the function.</param>
 /// <returns>returns new symbol</returns>
public static Symbol FillElement0Index(string symbol_name,
Symbol lhs,
Symbol mhs,
Symbol rhs)
{
return new Operator("fill_element_0index")
.SetParam("lhs", lhs)
.SetParam("mhs", mhs)
.SetParam("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
/// </summary>
/// <param name="lhs">Left operand to the function.</param>
/// <param name="mhs">Middle operand to the function.</param>
/// <param name="rhs">Right operand to the function.</param>
 /// <returns>returns new symbol</returns>
public static Symbol FillElement0Index(Symbol lhs,
Symbol mhs,
Symbol rhs)
{
return new Operator("fill_element_0index")
.SetParam("lhs", lhs)
.SetParam("mhs", mhs)
.SetParam("rhs", rhs)
.CreateSymbol();
}
}
}
