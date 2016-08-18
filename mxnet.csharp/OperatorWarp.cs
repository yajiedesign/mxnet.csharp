using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class OperatorWarp
    {/// <summary>
/// Activation function to be applied.
/// </summary>
public enum Activationact_type
{relu,
sigmoid,
softrelu,
tanh
};/// <summary>
/// Apply activation function to input.Softmax Activation is only available with CUDNN on GPUand will be computed at each location across channel if input is 4D.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to activation function.</param>/// <param name="act_type">Activation function to be applied.</param> /// <returns>returns new symbol</returns>
public Symbol Activation(string symbol_name,
Symbol data,
Activationact_type act_type)
{return new Operator("Activation")
.SetParam("act_type", act_type)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply activation function to input.Softmax Activation is only available with CUDNN on GPUand will be computed at each location across channel if input is 4D.
/// </summary>
/// <param name="data">Input data to activation function.</param>/// <param name="act_type">Activation function to be applied.</param> /// <returns>returns new symbol</returns>
public Symbol Activation(Symbol data,
Activationact_type act_type)
{return new Operator("Activation")
.SetParam("act_type", act_type)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to batch normalization</param>/// <param name="eps">Epsilon to prevent div 0</param>/// <param name="momentum">Momentum for moving average</param>/// <param name="fix_gamma">Fix gamma while training</param>/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param> /// <returns>returns new symbol</returns>
public Symbol BatchNorm(string symbol_name,
Symbol data,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false)
{return new Operator("BatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="data">Input data to batch normalization</param>/// <param name="eps">Epsilon to prevent div 0</param>/// <param name="momentum">Momentum for moving average</param>/// <param name="fix_gamma">Fix gamma while training</param>/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param> /// <returns>returns new symbol</returns>
public Symbol BatchNorm(Symbol data,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false)
{return new Operator("BatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Get output from a symbol and pass 0 gradient back
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data.</param> /// <returns>returns new symbol</returns>
public Symbol BlockGrad(string symbol_name,
Symbol data)
{return new Operator("BlockGrad")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Get output from a symbol and pass 0 gradient back
/// </summary>
/// <param name="data">Input data.</param> /// <returns>returns new symbol</returns>
public Symbol BlockGrad(Symbol data)
{return new Operator("BlockGrad")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.</param>/// <param name="keepdims">Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.</param> /// <returns>returns new symbol</returns>
public Symbol sum(string symbol_name,
Symbol src,
Shape axis=null,
bool keepdims=false)
{axis= new Shape();return new Operator("sum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.</param>/// <param name="keepdims">Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.</param> /// <returns>returns new symbol</returns>
public Symbol sum(Symbol src,
Shape axis=null,
bool keepdims=false)
{axis= new Shape();return new Operator("sum")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// (Depreciated! Use sum instead!) Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.</param>/// <param name="keepdims">Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.</param> /// <returns>returns new symbol</returns>
public Symbol sum_axis(string symbol_name,
Symbol src,
Shape axis=null,
bool keepdims=false)
{axis= new Shape();return new Operator("sum_axis")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// (Depreciated! Use sum instead!) Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.</param>/// <param name="keepdims">Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.</param> /// <returns>returns new symbol</returns>
public Symbol sum_axis(Symbol src,
Shape axis=null,
bool keepdims=false)
{axis= new Shape();return new Operator("sum_axis")
.SetParam("axis", axis)
.SetParam("keepdims", keepdims)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Broadcast data in the given axis to the given size. The original size of the broadcasting axis must be 1.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">The axes to perform the broadcasting.</param>/// <param name="size">Target sizes of the broadcasting axes.</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_axis(string symbol_name,
Symbol src,
Shape axis=null,
Shape size=null)
{axis= new Shape();size= new Shape();return new Operator("broadcast_axis")
.SetParam("axis", axis)
.SetParam("size", size)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Broadcast data in the given axis to the given size. The original size of the broadcasting axis must be 1.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">The axes to perform the broadcasting.</param>/// <param name="size">Target sizes of the broadcasting axes.</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_axis(Symbol src,
Shape axis=null,
Shape size=null)
{axis= new Shape();size= new Shape();return new Operator("broadcast_axis")
.SetParam("axis", axis)
.SetParam("size", size)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Broadcast data to the target shape. The original size of the broadcasting axis must be 1.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="shape">The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_to(string symbol_name,
Symbol src,
Shape shape=null)
{shape= new Shape();return new Operator("broadcast_to")
.SetParam("shape", shape)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Broadcast data to the target shape. The original size of the broadcasting axis must be 1.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="shape">The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_to(Symbol src,
Shape shape=null)
{shape= new Shape();return new Operator("broadcast_to")
.SetParam("shape", shape)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Target data type.
/// </summary>
public enum Castdtype
{float16,
float32,
float64,
int32,
uint8
};/// <summary>
/// Cast array to a different data type.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to cast function.</param>/// <param name="dtype">Target data type.</param> /// <returns>returns new symbol</returns>
public Symbol Cast(string symbol_name,
Symbol data,
Castdtype dtype)
{return new Operator("Cast")
.SetParam("dtype", dtype)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Cast array to a different data type.
/// </summary>
/// <param name="data">Input data to cast function.</param>/// <param name="dtype">Target data type.</param> /// <returns>returns new symbol</returns>
public Symbol Cast(Symbol data,
Castdtype dtype)
{return new Operator("Cast")
.SetParam("dtype", dtype)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Perform an feature concat on channel dim (defaut is 1) over all
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">List of tensors to concatenate</param>/// <param name="num_args">Number of inputs to be concated.</param>/// <param name="dim">the dimension to be concated.</param> /// <returns>returns new symbol</returns>
public Symbol Concat(string symbol_name,
Symbol[] data,
int num_args,
int dim=1)
{return new Operator("Concat")
.SetParam("num_args", num_args)
.SetParam("dim", dim)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Perform an feature concat on channel dim (defaut is 1) over all
/// </summary>
/// <param name="data">List of tensors to concatenate</param>/// <param name="num_args">Number of inputs to be concated.</param>/// <param name="dim">the dimension to be concated.</param> /// <returns>returns new symbol</returns>
public Symbol Concat(Symbol[] data,
int num_args,
int dim=1)
{return new Operator("Concat")
.SetParam("num_args", num_args)
.SetParam("dim", dim)
.AddInput(data)
.CreateSymbol();
}
/// <summary>
/// Whether to find convolution algo by running performance test.Leads to higher startup time but may give better speed.auto tune is turned off by default.Set environment varialbe MXNET_CUDNN_AUTOTUNE_DEFAULT=1 to turn on by default.
/// </summary>
public enum Convolutioncudnn_tune
{fastest,
limited_workspace,
off
};/// <summary>
/// Apply convolution to input then add a bias.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the ConvolutionOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="kernel">convolution kernel size: (y, x) or (d, y, x)</param>/// <param name="num_filter">convolution filter(channel) number</param>/// <param name="stride">convolution stride: (y, x) or (d, y, x)</param>/// <param name="dilate">convolution dilate: (y, x)</param>/// <param name="pad">pad for convolution: (y, x) or (d, y, x)</param>/// <param name="num_group">Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.</param>/// <param name="workspace">Tmp workspace for convolution (MB).</param>/// <param name="no_bias">Whether to disable bias parameter.</param>/// <param name="cudnn_tune">Whether to find convolution algo by running performance test.Leads to higher startup time but may give better speed.auto tune is turned off by default.Set environment varialbe MXNET_CUDNN_AUTOTUNE_DEFAULT=1 to turn on by default.</param> /// <returns>returns new symbol</returns>
public Symbol Convolution(string symbol_name,
Symbol data,
Symbol weight,
Symbol bias,
Shape kernel,
int num_filter,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
Convolutioncudnn_tune cudnn_tune=Convolutioncudnn_tune.off)
{stride= new Shape(1,1);dilate= new Shape(1,1);pad= new Shape(0,0);return new Operator("Convolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", cudnn_tune)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply convolution to input then add a bias.
/// </summary>
/// <param name="data">Input data to the ConvolutionOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="kernel">convolution kernel size: (y, x) or (d, y, x)</param>/// <param name="num_filter">convolution filter(channel) number</param>/// <param name="stride">convolution stride: (y, x) or (d, y, x)</param>/// <param name="dilate">convolution dilate: (y, x)</param>/// <param name="pad">pad for convolution: (y, x) or (d, y, x)</param>/// <param name="num_group">Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.</param>/// <param name="workspace">Tmp workspace for convolution (MB).</param>/// <param name="no_bias">Whether to disable bias parameter.</param>/// <param name="cudnn_tune">Whether to find convolution algo by running performance test.Leads to higher startup time but may give better speed.auto tune is turned off by default.Set environment varialbe MXNET_CUDNN_AUTOTUNE_DEFAULT=1 to turn on by default.</param> /// <returns>returns new symbol</returns>
public Symbol Convolution(Symbol data,
Symbol weight,
Symbol bias,
Shape kernel,
int num_filter,
Shape stride=null,
Shape dilate=null,
Shape pad=null,
int num_group=1,
long workspace=1024,
bool no_bias=false,
Convolutioncudnn_tune cudnn_tune=Convolutioncudnn_tune.off)
{stride= new Shape(1,1);dilate= new Shape(1,1);pad= new Shape(0,0);return new Operator("Convolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("dilate", dilate)
.SetParam("pad", pad)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetParam("cudnn_tune", cudnn_tune)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
/// <summary>
/// Apply correlation to inputs
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data1">Input data1 to the correlation.</param>/// <param name="data2">Input data2 to the correlation.</param>/// <param name="kernel_size">kernel size for Correlation must be an odd number</param>/// <param name="max_displacement">Max displacement of Correlation </param>/// <param name="stride1">stride1 quantize data1 globally</param>/// <param name="stride2">stride2 quantize data2 within the neighborhood centered around data1</param>/// <param name="pad_size">pad for Correlation</param>/// <param name="is_multiply">operation type is either multiplication or subduction</param> /// <returns>returns new symbol</returns>
public Symbol Correlation(string symbol_name,
Symbol data1,
Symbol data2,
int kernel_size=1,
int max_displacement=1,
int stride1=1,
int stride2=1,
int pad_size=0,
bool is_multiply=true)
{return new Operator("Correlation")
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
/// Apply correlation to inputs
/// </summary>
/// <param name="data1">Input data1 to the correlation.</param>/// <param name="data2">Input data2 to the correlation.</param>/// <param name="kernel_size">kernel size for Correlation must be an odd number</param>/// <param name="max_displacement">Max displacement of Correlation </param>/// <param name="stride1">stride1 quantize data1 globally</param>/// <param name="stride2">stride2 quantize data2 within the neighborhood centered around data1</param>/// <param name="pad_size">pad for Correlation</param>/// <param name="is_multiply">operation type is either multiplication or subduction</param> /// <returns>returns new symbol</returns>
public Symbol Correlation(Symbol data1,
Symbol data2,
int kernel_size=1,
int max_displacement=1,
int stride1=1,
int stride2=1,
int pad_size=0,
bool is_multiply=true)
{return new Operator("Correlation")
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
/// Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or with width and height of the second input symbol, i.e., with one input, we need h_w to specify the crop height and width, otherwise the second input symbol's size will be used
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Tensor or List of Tensors, the second input will be used as crop_like shape reference</param>/// <param name="num_args">Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here</param>/// <param name="offset">crop offset coordinate: (y, x)</param>/// <param name="h_w">crop height and weight: (h, w)</param>/// <param name="center_crop">If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like</param> /// <returns>returns new symbol</returns>
public Symbol Crop(string symbol_name,
Symbol data,
int num_args,
Shape offset=null,
Shape h_w=null,
bool center_crop=false)
{offset= new Shape(0,0);h_w= new Shape(0,0);return new Operator("Crop")
.SetParam("num_args", num_args)
.SetParam("offset", offset)
.SetParam("h_w", h_w)
.SetParam("center_crop", center_crop)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or with width and height of the second input symbol, i.e., with one input, we need h_w to specify the crop height and width, otherwise the second input symbol's size will be used
/// </summary>
/// <param name="data">Tensor or List of Tensors, the second input will be used as crop_like shape reference</param>/// <param name="num_args">Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here</param>/// <param name="offset">crop offset coordinate: (y, x)</param>/// <param name="h_w">crop height and weight: (h, w)</param>/// <param name="center_crop">If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like</param> /// <returns>returns new symbol</returns>
public Symbol Crop(Symbol data,
int num_args,
Shape offset=null,
Shape h_w=null,
bool center_crop=false)
{offset= new Shape(0,0);h_w= new Shape(0,0);return new Operator("Crop")
.SetParam("num_args", num_args)
.SetParam("offset", offset)
.SetParam("h_w", h_w)
.SetParam("center_crop", center_crop)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to batch normalization</param>/// <param name="eps">Epsilon to prevent div 0</param>/// <param name="momentum">Momentum for moving average</param>/// <param name="fix_gamma">Fix gamma while training</param>/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param> /// <returns>returns new symbol</returns>
public Symbol CuDNNBatchNorm(string symbol_name,
Symbol data,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false)
{return new Operator("CuDNNBatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply batch normalization to input.
/// </summary>
/// <param name="data">Input data to batch normalization</param>/// <param name="eps">Epsilon to prevent div 0</param>/// <param name="momentum">Momentum for moving average</param>/// <param name="fix_gamma">Fix gamma while training</param>/// <param name="use_global_stats">Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.</param> /// <returns>returns new symbol</returns>
public Symbol CuDNNBatchNorm(Symbol data,
float eps=0.001f,
float momentum=0.9f,
bool fix_gamma=true,
bool use_global_stats=false)
{return new Operator("CuDNNBatchNorm")
.SetParam("eps", eps)
.SetParam("momentum", momentum)
.SetParam("fix_gamma", fix_gamma)
.SetParam("use_global_stats", use_global_stats)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Custom operator implemented in frontend.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="op_type">Type of custom operator. Must be registered first.</param> /// <returns>returns new symbol</returns>
public Symbol Custom(string symbol_name,
string op_type)
{return new Operator("Custom")
.CreateSymbol(symbol_name);
}
/// <summary>
/// Custom operator implemented in frontend.
/// </summary>
/// <param name="op_type">Type of custom operator. Must be registered first.</param> /// <returns>returns new symbol</returns>
public Symbol Custom(string op_type)
{return new Operator("Custom")
.CreateSymbol();
}
/// <summary>
/// Apply deconvolution to input then add a bias.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the DeconvolutionOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="kernel">deconvolution kernel size: (y, x)</param>/// <param name="num_filter">deconvolution filter(channel) number</param>/// <param name="stride">deconvolution stride: (y, x)</param>/// <param name="pad">pad for deconvolution: (y, x), a good number is : (kernel-1)/2, if target_shape set, pad will be ignored and will be computed automatically</param>/// <param name="adj">adjustment for output shape: (y, x), if target_shape set, adj will be ignored and will be computed automatically</param>/// <param name="target_shape">output shape with targe shape : (y, x)</param>/// <param name="num_group">number of groups partition</param>/// <param name="workspace">Tmp workspace for deconvolution (MB)</param>/// <param name="no_bias">Whether to disable bias parameter.</param> /// <returns>returns new symbol</returns>
public Symbol Deconvolution(string symbol_name,
Symbol data,
Symbol weight,
Symbol bias,
Shape kernel,
int num_filter,
Shape stride=null,
Shape pad=null,
Shape adj=null,
Shape target_shape=null,
int num_group=1,
long workspace=512,
bool no_bias=true)
{stride= new Shape(1,1);pad= new Shape(0,0);adj= new Shape(0,0);target_shape= new Shape(0,0);return new Operator("Deconvolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetParam("adj", adj)
.SetParam("target_shape", target_shape)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply deconvolution to input then add a bias.
/// </summary>
/// <param name="data">Input data to the DeconvolutionOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="kernel">deconvolution kernel size: (y, x)</param>/// <param name="num_filter">deconvolution filter(channel) number</param>/// <param name="stride">deconvolution stride: (y, x)</param>/// <param name="pad">pad for deconvolution: (y, x), a good number is : (kernel-1)/2, if target_shape set, pad will be ignored and will be computed automatically</param>/// <param name="adj">adjustment for output shape: (y, x), if target_shape set, adj will be ignored and will be computed automatically</param>/// <param name="target_shape">output shape with targe shape : (y, x)</param>/// <param name="num_group">number of groups partition</param>/// <param name="workspace">Tmp workspace for deconvolution (MB)</param>/// <param name="no_bias">Whether to disable bias parameter.</param> /// <returns>returns new symbol</returns>
public Symbol Deconvolution(Symbol data,
Symbol weight,
Symbol bias,
Shape kernel,
int num_filter,
Shape stride=null,
Shape pad=null,
Shape adj=null,
Shape target_shape=null,
int num_group=1,
long workspace=512,
bool no_bias=true)
{stride= new Shape(1,1);pad= new Shape(0,0);adj= new Shape(0,0);target_shape= new Shape(0,0);return new Operator("Deconvolution")
.SetParam("kernel", kernel)
.SetParam("num_filter", num_filter)
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetParam("adj", adj)
.SetParam("target_shape", target_shape)
.SetParam("num_group", num_group)
.SetParam("workspace", workspace)
.SetParam("no_bias", no_bias)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
/// <summary>
/// Apply dropout to input
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to dropout.</param>/// <param name="p">Fraction of the input that gets dropped out at training time</param> /// <returns>returns new symbol</returns>
public Symbol Dropout(string symbol_name,
Symbol data,
float p=0.5f)
{return new Operator("Dropout")
.SetParam("p", p)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply dropout to input
/// </summary>
/// <param name="data">Input data to dropout.</param>/// <param name="p">Fraction of the input that gets dropped out at training time</param> /// <returns>returns new symbol</returns>
public Symbol Dropout(Symbol data,
float p=0.5f)
{return new Operator("Dropout")
.SetParam("p", p)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// lhs add rhs with broadcast
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_plus(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_plus")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// lhs add rhs with broadcast
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_plus(Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_plus")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// lhs minus rhs with broadcast
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_minus(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_minus")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// lhs minus rhs with broadcast
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_minus(Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_minus")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// lhs multiple rhs with broadcast
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_mul(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// lhs multiple rhs with broadcast
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_mul(Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_mul")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// lhs divide rhs with broadcast
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_div(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// lhs divide rhs with broadcast
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_div(Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_div")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// lhs power rhs with broadcast
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_power(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_power")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// lhs power rhs with broadcast
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol broadcast_power(Symbol lhs,
Symbol rhs)
{return new Operator("broadcast_power")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Perform an elementwise sum over all the inputs.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="num_args">Number of inputs to be summed.</param> /// <returns>returns new symbol</returns>
public Symbol ElementWiseSum(string symbol_name,
int num_args)
{return new Operator("ElementWiseSum")
.SetParam("num_args", num_args)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Perform an elementwise sum over all the inputs.
/// </summary>
/// <param name="num_args">Number of inputs to be summed.</param> /// <returns>returns new symbol</returns>
public Symbol ElementWiseSum(int num_args)
{return new Operator("ElementWiseSum")
.SetParam("num_args", num_args)
.CreateSymbol();
}
/// <summary>
/// Take absolute value of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol abs(string symbol_name,
Symbol src)
{return new Operator("abs")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take absolute value of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol abs(Symbol src)
{return new Operator("abs")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take sign value of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sign(string symbol_name,
Symbol src)
{return new Operator("sign")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take sign value of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sign(Symbol src)
{return new Operator("sign")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take round value of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol round(string symbol_name,
Symbol src)
{return new Operator("round")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take round value of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol round(Symbol src)
{return new Operator("round")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take ceil value of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol ceil(string symbol_name,
Symbol src)
{return new Operator("ceil")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take ceil value of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol ceil(Symbol src)
{return new Operator("ceil")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take floor value of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol floor(string symbol_name,
Symbol src)
{return new Operator("floor")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take floor value of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol floor(Symbol src)
{return new Operator("floor")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take square of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol square(string symbol_name,
Symbol src)
{return new Operator("square")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take square of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol square(Symbol src)
{return new Operator("square")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take sqrt of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sqrt(string symbol_name,
Symbol src)
{return new Operator("sqrt")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take sqrt of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sqrt(Symbol src)
{return new Operator("sqrt")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take rsqrt of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol rsqrt(string symbol_name,
Symbol src)
{return new Operator("rsqrt")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take rsqrt of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol rsqrt(Symbol src)
{return new Operator("rsqrt")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take exp of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol exp(string symbol_name,
Symbol src)
{return new Operator("exp")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take exp of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol exp(Symbol src)
{return new Operator("exp")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take log of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol log(string symbol_name,
Symbol src)
{return new Operator("log")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take log of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol log(Symbol src)
{return new Operator("log")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take cos of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol cos(string symbol_name,
Symbol src)
{return new Operator("cos")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take cos of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol cos(Symbol src)
{return new Operator("cos")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Take sin of the src
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sin(string symbol_name,
Symbol src)
{return new Operator("sin")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Take sin of the src
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol sin(Symbol src)
{return new Operator("sin")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Get embedding for one-hot input. A n-dimensional input tensor will be trainsformed into a (n+1)-dimensional tensor, where a new dimension is added for the embedding results.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the EmbeddingOp.</param>/// <param name="weight">Enbedding weight matrix.</param>/// <param name="input_dim">input dim of one-hot encoding</param>/// <param name="output_dim">output dim of embedding</param> /// <returns>returns new symbol</returns>
public Symbol Embedding(string symbol_name,
Symbol data,
Symbol weight,
int input_dim,
int output_dim)
{return new Operator("Embedding")
.SetParam("input_dim", input_dim)
.SetParam("output_dim", output_dim)
.SetInput("data", data)
.SetInput("weight", weight)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Get embedding for one-hot input. A n-dimensional input tensor will be trainsformed into a (n+1)-dimensional tensor, where a new dimension is added for the embedding results.
/// </summary>
/// <param name="data">Input data to the EmbeddingOp.</param>/// <param name="weight">Enbedding weight matrix.</param>/// <param name="input_dim">input dim of one-hot encoding</param>/// <param name="output_dim">output dim of embedding</param> /// <returns>returns new symbol</returns>
public Symbol Embedding(Symbol data,
Symbol weight,
int input_dim,
int output_dim)
{return new Operator("Embedding")
.SetParam("input_dim", input_dim)
.SetParam("output_dim", output_dim)
.SetInput("data", data)
.SetInput("weight", weight)
.CreateSymbol();
}
/// <summary>
/// Apply matrix multiplication to input then add a bias.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the FullyConnectedOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="num_hidden">Number of hidden nodes of the output.</param>/// <param name="no_bias">Whether to disable bias parameter.</param> /// <returns>returns new symbol</returns>
public Symbol FullyConnected(string symbol_name,
Symbol data,
Symbol weight,
Symbol bias,
int num_hidden,
bool no_bias=false)
{return new Operator("FullyConnected")
.SetParam("num_hidden", num_hidden)
.SetParam("no_bias", no_bias)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply matrix multiplication to input then add a bias.
/// </summary>
/// <param name="data">Input data to the FullyConnectedOp.</param>/// <param name="weight">Weight matrix.</param>/// <param name="bias">Bias parameter.</param>/// <param name="num_hidden">Number of hidden nodes of the output.</param>/// <param name="no_bias">Whether to disable bias parameter.</param> /// <returns>returns new symbol</returns>
public Symbol FullyConnected(Symbol data,
Symbol weight,
Symbol bias,
int num_hidden,
bool no_bias=false)
{return new Operator("FullyConnected")
.SetParam("num_hidden", num_hidden)
.SetParam("no_bias", no_bias)
.SetInput("data", data)
.SetInput("weight", weight)
.SetInput("bias", bias)
.CreateSymbol();
}
/// <summary>
/// Apply a sparse regularization to the output a sigmoid activation function.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data.</param>/// <param name="sparseness_target">The sparseness target</param>/// <param name="penalty">The tradeoff parameter for the sparseness penalty</param>/// <param name="momentum">The momentum for running average</param> /// <returns>returns new symbol</returns>
public Symbol IdentityAttachKLSparseReg(string symbol_name,
Symbol data,
float sparseness_target=0.1f,
float penalty=0.001f,
float momentum=0.9f)
{return new Operator("IdentityAttachKLSparseReg")
.SetParam("sparseness_target", sparseness_target)
.SetParam("penalty", penalty)
.SetParam("momentum", momentum)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply a sparse regularization to the output a sigmoid activation function.
/// </summary>
/// <param name="data">Input data.</param>/// <param name="sparseness_target">The sparseness target</param>/// <param name="penalty">The tradeoff parameter for the sparseness penalty</param>/// <param name="momentum">The momentum for running average</param> /// <returns>returns new symbol</returns>
public Symbol IdentityAttachKLSparseReg(Symbol data,
float sparseness_target=0.1f,
float penalty=0.001f,
float momentum=0.9f)
{return new Operator("IdentityAttachKLSparseReg")
.SetParam("sparseness_target", sparseness_target)
.SetParam("penalty", penalty)
.SetParam("momentum", momentum)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Set the l2 norm of each instance to a constant.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the L2NormalizationOp.</param>/// <param name="eps">Epsilon to prevent div 0</param> /// <returns>returns new symbol</returns>
public Symbol L2Normalization(string symbol_name,
Symbol data,
float eps=1e-010f)
{return new Operator("L2Normalization")
.SetParam("eps", eps)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Set the l2 norm of each instance to a constant.
/// </summary>
/// <param name="data">Input data to the L2NormalizationOp.</param>/// <param name="eps">Epsilon to prevent div 0</param> /// <returns>returns new symbol</returns>
public Symbol L2Normalization(Symbol data,
float eps=1e-010f)
{return new Operator("L2Normalization")
.SetParam("eps", eps)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Activation function to be applied.
/// </summary>
public enum LeakyReLUact_type
{elu,
leaky,
prelu,
rrelu
};/// <summary>
/// Apply activation function to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to activation function.</param>/// <param name="act_type">Activation function to be applied.</param>/// <param name="slope">Init slope for the activation. (For leaky and elu only)</param>/// <param name="lower_bound">Lower bound of random slope. (For rrelu only)</param>/// <param name="upper_bound">Upper bound of random slope. (For rrelu only)</param> /// <returns>returns new symbol</returns>
public Symbol LeakyReLU(string symbol_name,
Symbol data,
LeakyReLUact_type act_type=LeakyReLUact_type.leaky,
float slope=0.25f,
float lower_bound=0.125f,
float upper_bound=0.334f)
{return new Operator("LeakyReLU")
.SetParam("act_type", act_type)
.SetParam("slope", slope)
.SetParam("lower_bound", lower_bound)
.SetParam("upper_bound", upper_bound)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply activation function to input.
/// </summary>
/// <param name="data">Input data to activation function.</param>/// <param name="act_type">Activation function to be applied.</param>/// <param name="slope">Init slope for the activation. (For leaky and elu only)</param>/// <param name="lower_bound">Lower bound of random slope. (For rrelu only)</param>/// <param name="upper_bound">Upper bound of random slope. (For rrelu only)</param> /// <returns>returns new symbol</returns>
public Symbol LeakyReLU(Symbol data,
LeakyReLUact_type act_type=LeakyReLUact_type.leaky,
float slope=0.25f,
float lower_bound=0.125f,
float upper_bound=0.334f)
{return new Operator("LeakyReLU")
.SetParam("act_type", act_type)
.SetParam("slope", slope)
.SetParam("lower_bound", lower_bound)
.SetParam("upper_bound", upper_bound)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Calculate cross_entropy(lhs, one_hot(rhs))
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol softmax_cross_entropy(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("softmax_cross_entropy")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate cross_entropy(lhs, one_hot(rhs))
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol softmax_cross_entropy(Symbol lhs,
Symbol rhs)
{return new Operator("softmax_cross_entropy")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Apply convolution to input then add a bias.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the ConvolutionOp.</param>/// <param name="nsize">normalization window width in elements.</param>/// <param name="alpha">value of the alpha variance scaling parameter in the normalization formula</param>/// <param name="beta">value of the beta power parameter in the normalization formula</param>/// <param name="knorm">value of the k parameter in normalization formula</param> /// <returns>returns new symbol</returns>
public Symbol LRN(string symbol_name,
Symbol data,
int nsize,
float alpha=0.0001f,
float beta=0.75f,
float knorm=2f)
{return new Operator("LRN")
.SetParam("nsize", nsize)
.SetParam("alpha", alpha)
.SetParam("beta", beta)
.SetParam("knorm", knorm)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply convolution to input then add a bias.
/// </summary>
/// <param name="data">Input data to the ConvolutionOp.</param>/// <param name="nsize">normalization window width in elements.</param>/// <param name="alpha">value of the alpha variance scaling parameter in the normalization formula</param>/// <param name="beta">value of the beta power parameter in the normalization formula</param>/// <param name="knorm">value of the k parameter in normalization formula</param> /// <returns>returns new symbol</returns>
public Symbol LRN(Symbol data,
int nsize,
float alpha=0.0001f,
float beta=0.75f,
float knorm=2f)
{return new Operator("LRN")
.SetParam("nsize", nsize)
.SetParam("alpha", alpha)
.SetParam("beta", beta)
.SetParam("knorm", knorm)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Get output from a symbol and pass 1 gradient back. This is used as a terminal loss if unary and binary operator are used to composite a loss with no declaration of backward dependency
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data.</param>/// <param name="grad_scale">gradient scale as a supplement to unary and binary operators</param> /// <returns>returns new symbol</returns>
public Symbol MakeLoss(string symbol_name,
Symbol data,
float grad_scale=1f)
{return new Operator("MakeLoss")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Get output from a symbol and pass 1 gradient back. This is used as a terminal loss if unary and binary operator are used to composite a loss with no declaration of backward dependency
/// </summary>
/// <param name="data">Input data.</param>/// <param name="grad_scale">gradient scale as a supplement to unary and binary operators</param> /// <returns>returns new symbol</returns>
public Symbol MakeLoss(Symbol data,
float grad_scale=1f)
{return new Operator("MakeLoss")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Transpose the input matrix and return a new one
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axes">Target axis order. By default the axes will be inverted.</param> /// <returns>returns new symbol</returns>
public Symbol transpose(string symbol_name,
Symbol src,
Shape axes=null)
{axes= new Shape();return new Operator("transpose")
.SetParam("axes", axes)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Transpose the input matrix and return a new one
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axes">Target axis order. By default the axes will be inverted.</param> /// <returns>returns new symbol</returns>
public Symbol transpose(Symbol src,
Shape axes=null)
{axes= new Shape();return new Operator("transpose")
.SetParam("axes", axes)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Expand the shape of array by inserting a new axis.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Position (amongst axes) where new axis is to be inserted.</param> /// <returns>returns new symbol</returns>
public Symbol expand_dims(string symbol_name,
Symbol src,
int axis)
{return new Operator("expand_dims")
.SetParam("axis", axis)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Expand the shape of array by inserting a new axis.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">Position (amongst axes) where new axis is to be inserted.</param> /// <returns>returns new symbol</returns>
public Symbol expand_dims(Symbol src,
int axis)
{return new Operator("expand_dims")
.SetParam("axis", axis)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Slice the input along certain axis and return a sliced array.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">The axis to be sliced</param>/// <param name="begin">The beginning index to be sliced</param>/// <param name="end">The end index to be sliced</param> /// <returns>returns new symbol</returns>
public Symbol slice_axis(string symbol_name,
Symbol src,
int axis,
int begin,
int end)
{return new Operator("slice_axis")
.SetParam("axis", axis)
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Slice the input along certain axis and return a sliced array.
/// </summary>
/// <param name="src">Left symbolic input to the function</param>/// <param name="axis">The axis to be sliced</param>/// <param name="begin">The beginning index to be sliced</param>/// <param name="end">The end index to be sliced</param> /// <returns>returns new symbol</returns>
public Symbol slice_axis(Symbol src,
int axis,
int begin,
int end)
{return new Operator("slice_axis")
.SetParam("axis", axis)
.SetParam("begin", begin)
.SetParam("end", end)
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Calculate dot product of two matrices or two vectors
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol dot(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("dot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate dot product of two matrices or two vectors
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol dot(Symbol lhs,
Symbol rhs)
{return new Operator("dot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Calculate batched dot product of two matrices. (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol batch_dot(string symbol_name,
Symbol lhs,
Symbol rhs)
{return new Operator("batch_dot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate batched dot product of two matrices. (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)
/// </summary>
/// <param name="lhs">Left symbolic input to the function</param>/// <param name="rhs">Right symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol batch_dot(Symbol lhs,
Symbol rhs)
{return new Operator("batch_dot")
.SetInput("lhs", lhs)
.SetInput("rhs", rhs)
.CreateSymbol();
}
/// <summary>
/// Pooling type to be applied.
/// </summary>
public enum Poolingpool_type
{avg,
max,
sum
};/// <summary>
/// Perform spatial pooling on inputs.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the pooling operator.</param>/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>/// <param name="pool_type">Pooling type to be applied.</param>/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. This is useful for input with different shape</param>/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param> /// <returns>returns new symbol</returns>
public Symbol Pooling(string symbol_name,
Symbol data,
Shape kernel,
Poolingpool_type pool_type,
bool global_pool=false,
Shape stride=null,
Shape pad=null)
{stride= new Shape(1,1);pad= new Shape(0,0);return new Operator("Pooling")
.SetParam("kernel", kernel)
.SetParam("pool_type", pool_type)
.SetParam("global_pool", global_pool)
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Perform spatial pooling on inputs.
/// </summary>
/// <param name="data">Input data to the pooling operator.</param>/// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>/// <param name="pool_type">Pooling type to be applied.</param>/// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. This is useful for input with different shape</param>/// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>/// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param> /// <returns>returns new symbol</returns>
public Symbol Pooling(Symbol data,
Shape kernel,
Poolingpool_type pool_type,
bool global_pool=false,
Shape stride=null,
Shape pad=null)
{stride= new Shape(1,1);pad= new Shape(0,0);return new Operator("Pooling")
.SetParam("kernel", kernel)
.SetParam("pool_type", pool_type)
.SetParam("global_pool", global_pool)
.SetParam("stride", stride)
.SetParam("pad", pad)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Use linear regression for final output, this is used on final output of a net.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol LinearRegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("LinearRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Use linear regression for final output, this is used on final output of a net.
/// </summary>
/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol LinearRegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("LinearRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Use mean absolute error regression for final output, this is used on final output of a net.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol MAERegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("MAERegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Use mean absolute error regression for final output, this is used on final output of a net.
/// </summary>
/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol MAERegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("MAERegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Use Logistic regression for final output, this is used on final output of a net.Logistic regression is suitable for binary classification or probability prediction tasks.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol LogisticRegressionOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("LogisticRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Use Logistic regression for final output, this is used on final output of a net.Logistic regression is suitable for binary classification or probability prediction tasks.
/// </summary>
/// <param name="data">Input data to function.</param>/// <param name="label">Input label to function.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param> /// <returns>returns new symbol</returns>
public Symbol LogisticRegressionOutput(Symbol data,
Symbol label,
float grad_scale=1f)
{return new Operator("LogisticRegressionOutput")
.SetParam("grad_scale", grad_scale)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Reshape input to target shape
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to reshape.</param>/// <param name="target_shape">(Deprecated! Use shape instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims</param>/// <param name="keep_highest">(Deprecated! Use shape instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input</param>/// <param name="shape">Target new shape. If the dim is same, set it to 0. If the dim is set to be -1, it will be inferred from the rest of dims. One and only one dim can be -1</param> /// <returns>returns new symbol</returns>
public Symbol Reshape(string symbol_name,
Symbol data,
Shape target_shape=null,
bool keep_highest=false,
Shape shape=null)
{target_shape= new Shape(0,0);shape= new Shape();return new Operator("Reshape")
.SetParam("target_shape", target_shape)
.SetParam("keep_highest", keep_highest)
.SetParam("shape", shape)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Reshape input to target shape
/// </summary>
/// <param name="data">Input data to reshape.</param>/// <param name="target_shape">(Deprecated! Use shape instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims</param>/// <param name="keep_highest">(Deprecated! Use shape instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input</param>/// <param name="shape">Target new shape. If the dim is same, set it to 0. If the dim is set to be -1, it will be inferred from the rest of dims. One and only one dim can be -1</param> /// <returns>returns new symbol</returns>
public Symbol Reshape(Symbol data,
Shape target_shape=null,
bool keep_highest=false,
Shape shape=null)
{target_shape= new Shape(0,0);shape= new Shape();return new Operator("Reshape")
.SetParam("target_shape", target_shape)
.SetParam("keep_highest", keep_highest)
.SetParam("shape", shape)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// Flatten input
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to flatten.</param> /// <returns>returns new symbol</returns>
public Symbol Flatten(string symbol_name,
Symbol data)
{return new Operator("Flatten")
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Flatten input
/// </summary>
/// <param name="data">Input data to flatten.</param> /// <returns>returns new symbol</returns>
public Symbol Flatten(Symbol data)
{return new Operator("Flatten")
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// the type of RNN to compute
/// </summary>
public enum RNNmode
{gru,
lstm,
rnn_relu,
rnn_tanh
};/// <summary>
/// Apply a recurrent layer to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to RNN</param>/// <param name="parameters">Vector of all RNN trainable parameters</param>/// <param name="state">initial hidden state of the RNN</param>/// <param name="state_cell">initial cell state for LSTM networks (only for LSTM)</param>/// <param name="state_size">size of the state for each layer</param>/// <param name="num_layers">number of stacked layers</param>/// <param name="mode">the type of RNN to compute</param>/// <param name="bidirectional">whether to use bidirectional recurrent layers</param>/// <param name="p">Fraction of the input that gets dropped out at training time</param>/// <param name="state_outputs">Whether to have the states as symbol outputs.</param> /// <returns>returns new symbol</returns>
public Symbol RNN(string symbol_name,
Symbol data,
Symbol parameters,
Symbol state,
Symbol state_cell,
int state_size,
int num_layers,
RNNmode mode,
bool bidirectional=false,
float p=0f,
bool state_outputs=false)
{return new Operator("RNN")
.SetParam("state_size", state_size)
.SetParam("num_layers", num_layers)
.SetParam("mode", mode)
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
/// Apply a recurrent layer to input.
/// </summary>
/// <param name="data">Input data to RNN</param>/// <param name="parameters">Vector of all RNN trainable parameters</param>/// <param name="state">initial hidden state of the RNN</param>/// <param name="state_cell">initial cell state for LSTM networks (only for LSTM)</param>/// <param name="state_size">size of the state for each layer</param>/// <param name="num_layers">number of stacked layers</param>/// <param name="mode">the type of RNN to compute</param>/// <param name="bidirectional">whether to use bidirectional recurrent layers</param>/// <param name="p">Fraction of the input that gets dropped out at training time</param>/// <param name="state_outputs">Whether to have the states as symbol outputs.</param> /// <returns>returns new symbol</returns>
public Symbol RNN(Symbol data,
Symbol parameters,
Symbol state,
Symbol state_cell,
int state_size,
int num_layers,
RNNmode mode,
bool bidirectional=false,
float p=0f,
bool state_outputs=false)
{return new Operator("RNN")
.SetParam("state_size", state_size)
.SetParam("num_layers", num_layers)
.SetParam("mode", mode)
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
/// Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output indicated by pooled_size. batch_size will change to the number of region bounding boxes after ROIPooling
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the pooling operator, a 4D Feature maps</param>/// <param name="rois">Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data</param>/// <param name="pooled_size">fix pooled size: (h, w)</param>/// <param name="spatial_scale">Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers</param> /// <returns>returns new symbol</returns>
public Symbol ROIPooling(string symbol_name,
Symbol data,
Symbol rois,
Shape pooled_size,
float spatial_scale)
{return new Operator("ROIPooling")
.SetParam("pooled_size", pooled_size)
.SetParam("spatial_scale", spatial_scale)
.SetInput("data", data)
.SetInput("rois", rois)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output indicated by pooled_size. batch_size will change to the number of region bounding boxes after ROIPooling
/// </summary>
/// <param name="data">Input data to the pooling operator, a 4D Feature maps</param>/// <param name="rois">Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data</param>/// <param name="pooled_size">fix pooled size: (h, w)</param>/// <param name="spatial_scale">Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers</param> /// <returns>returns new symbol</returns>
public Symbol ROIPooling(Symbol data,
Symbol rois,
Shape pooled_size,
float spatial_scale)
{return new Operator("ROIPooling")
.SetParam("pooled_size", pooled_size)
.SetParam("spatial_scale", spatial_scale)
.SetInput("data", data)
.SetInput("rois", rois)
.CreateSymbol();
}
/// <summary>
/// Sample a uniform distribution
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="shape">The shape of the output</param>/// <param name="low">The lower bound of distribution</param>/// <param name="high">The upper bound of distribution</param> /// <returns>returns new symbol</returns>
public Symbol uniform(string symbol_name,
Shape shape,
float low=0f,
float high=1f)
{return new Operator("uniform")
.SetParam("shape", shape)
.SetParam("low", low)
.SetParam("high", high)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Sample a uniform distribution
/// </summary>
/// <param name="shape">The shape of the output</param>/// <param name="low">The lower bound of distribution</param>/// <param name="high">The upper bound of distribution</param> /// <returns>returns new symbol</returns>
public Symbol uniform(Shape shape,
float low=0f,
float high=1f)
{return new Operator("uniform")
.SetParam("shape", shape)
.SetParam("low", low)
.SetParam("high", high)
.CreateSymbol();
}
/// <summary>
/// Sample a normal distribution
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="shape">The shape of the output</param>/// <param name="loc">Mean of the distribution.</param>/// <param name="scale">Standard deviation of the distribution.</param> /// <returns>returns new symbol</returns>
public Symbol normal(string symbol_name,
Shape shape,
float loc=0f,
float scale=1f)
{return new Operator("normal")
.SetParam("shape", shape)
.SetParam("loc", loc)
.SetParam("scale", scale)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Sample a normal distribution
/// </summary>
/// <param name="shape">The shape of the output</param>/// <param name="loc">Mean of the distribution.</param>/// <param name="scale">Standard deviation of the distribution.</param> /// <returns>returns new symbol</returns>
public Symbol normal(Shape shape,
float loc=0f,
float scale=1f)
{return new Operator("normal")
.SetParam("shape", shape)
.SetParam("loc", loc)
.SetParam("scale", scale)
.CreateSymbol();
}
/// <summary>
/// Slice input equally along specified axis
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="num_outputs">Number of outputs to be sliced.</param>/// <param name="axis">Dimension along which to slice.</param>/// <param name="squeeze_axis">If true AND the sliced dimension becomes 1, squeeze that dimension.</param> /// <returns>returns new symbol</returns>
public Symbol SliceChannel(string symbol_name,
int num_outputs,
int axis=1,
bool squeeze_axis=false)
{return new Operator("SliceChannel")
.SetParam("num_outputs", num_outputs)
.SetParam("axis", axis)
.SetParam("squeeze_axis", squeeze_axis)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Slice input equally along specified axis
/// </summary>
/// <param name="num_outputs">Number of outputs to be sliced.</param>/// <param name="axis">Dimension along which to slice.</param>/// <param name="squeeze_axis">If true AND the sliced dimension becomes 1, squeeze that dimension.</param> /// <returns>returns new symbol</returns>
public Symbol SliceChannel(int num_outputs,
int axis=1,
bool squeeze_axis=false)
{return new Operator("SliceChannel")
.SetParam("num_outputs", num_outputs)
.SetParam("axis", axis)
.SetParam("squeeze_axis", squeeze_axis)
.CreateSymbol();
}
/// <summary>
/// Calculate Smooth L1 Loss(lhs, scalar)
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol smooth_l1(string symbol_name,
Symbol src)
{return new Operator("smooth_l1")
.SetInput("src", src)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Calculate Smooth L1 Loss(lhs, scalar)
/// </summary>
/// <param name="src">Left symbolic input to the function</param> /// <returns>returns new symbol</returns>
public Symbol smooth_l1(Symbol src)
{return new Operator("smooth_l1")
.SetInput("src", src)
.CreateSymbol();
}
/// <summary>
/// Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
/// </summary>
public enum SoftmaxActivationmode
{channel,
instance
};/// <summary>
/// Apply softmax activation to input. This is intended for internal layers. For output (loss layer) please use SoftmaxOutput. If mode=instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If mode=channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to activation function.</param>/// <param name="mode">Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.</param> /// <returns>returns new symbol</returns>
public Symbol SoftmaxActivation(string symbol_name,
Symbol data,
SoftmaxActivationmode mode=SoftmaxActivationmode.instance)
{return new Operator("SoftmaxActivation")
.SetParam("mode", mode)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply softmax activation to input. This is intended for internal layers. For output (loss layer) please use SoftmaxOutput. If mode=instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If mode=channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
/// </summary>
/// <param name="data">Input data to activation function.</param>/// <param name="mode">Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.</param> /// <returns>returns new symbol</returns>
public Symbol SoftmaxActivation(Symbol data,
SoftmaxActivationmode mode=SoftmaxActivationmode.instance)
{return new Operator("SoftmaxActivation")
.SetParam("mode", mode)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored
/// </summary>
public enum SoftmaxOutputnormalization
{batch,
_null,
valid
};/// <summary>
/// Perform a softmax transformation on input, backprop with logloss.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to softmax.</param>/// <param name="label">Label data, can also be probability value with same shape as data</param>/// <param name="grad_scale">Scale the gradient by a float factor</param>/// <param name="ignore_label">the label value will be ignored during backward (only works if use_ignore is set to be true).</param>/// <param name="multi_output">If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes</param>/// <param name="use_ignore">If set to true, the ignore_label value will not contribute to the backward gradient</param>/// <param name="normalization">If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored</param> /// <returns>returns new symbol</returns>
public Symbol SoftmaxOutput(string symbol_name,
Symbol data,
Symbol label,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
SoftmaxOutputnormalization normalization=SoftmaxOutputnormalization._null)
{return new Operator("SoftmaxOutput")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("normalization", normalization)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Perform a softmax transformation on input, backprop with logloss.
/// </summary>
/// <param name="data">Input data to softmax.</param>/// <param name="label">Label data, can also be probability value with same shape as data</param>/// <param name="grad_scale">Scale the gradient by a float factor</param>/// <param name="ignore_label">the label value will be ignored during backward (only works if use_ignore is set to be true).</param>/// <param name="multi_output">If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes</param>/// <param name="use_ignore">If set to true, the ignore_label value will not contribute to the backward gradient</param>/// <param name="normalization">If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored</param> /// <returns>returns new symbol</returns>
public Symbol SoftmaxOutput(Symbol data,
Symbol label,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
SoftmaxOutputnormalization normalization=SoftmaxOutputnormalization._null)
{return new Operator("SoftmaxOutput")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("normalization", normalization)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored
/// </summary>
public enum Softmaxnormalization
{batch,
_null,
valid
};/// <summary>
/// DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to softmax.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param>/// <param name="ignore_label">the label value will be ignored during backward (only works if use_ignore is set to be true).</param>/// <param name="multi_output">If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes</param>/// <param name="use_ignore">If set to true, the ignore_label value will not contribute to the backward gradient</param>/// <param name="normalization">If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored</param> /// <returns>returns new symbol</returns>
public Symbol Softmax(string symbol_name,
Symbol data,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
Softmaxnormalization normalization=Softmaxnormalization._null)
{return new Operator("Softmax")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("normalization", normalization)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
/// </summary>
/// <param name="data">Input data to softmax.</param>/// <param name="grad_scale">Scale the gradient by a float factor</param>/// <param name="ignore_label">the label value will be ignored during backward (only works if use_ignore is set to be true).</param>/// <param name="multi_output">If set to true, for a (n,k,x_1,..,x_n) dimensional input tensor, softmax will generate n*x_1*...*x_n output, each has k classes</param>/// <param name="use_ignore">If set to true, the ignore_label value will not contribute to the backward gradient</param>/// <param name="normalization">If set to null, op will do nothing on output gradient.If set to batch, op will normalize gradient by divide batch sizeIf set to valid, op will normalize gradient by divide sample not ignored</param> /// <returns>returns new symbol</returns>
public Symbol Softmax(Symbol data,
float grad_scale=1f,
float ignore_label=-1f,
bool multi_output=false,
bool use_ignore=false,
Softmaxnormalization normalization=Softmaxnormalization._null)
{return new Operator("Softmax")
.SetParam("grad_scale", grad_scale)
.SetParam("ignore_label", ignore_label)
.SetParam("multi_output", multi_output)
.SetParam("use_ignore", use_ignore)
.SetParam("normalization", normalization)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// transformation type
/// </summary>
public enum SpatialTransformertransform_type
{affine
};/// <summary>
/// sampling type
/// </summary>
public enum SpatialTransformersampler_type
{bilinear
};/// <summary>
/// Apply spatial transformer to input feature map.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the SpatialTransformerOp.</param>/// <param name="loc">localisation net, the output dim should be 6 when transform_type is affine, and the name of loc symbol should better starts with 'stn_loc', so that initialization it with iddentify tranform, or you shold initialize the weight and bias by yourself.</param>/// <param name="transform_type">transformation type</param>/// <param name="sampler_type">sampling type</param>/// <param name="target_shape">output shape(h, w) of spatial transformer: (y, x)</param> /// <returns>returns new symbol</returns>
public Symbol SpatialTransformer(string symbol_name,
Symbol data,
Symbol loc,
SpatialTransformertransform_type transform_type,
SpatialTransformersampler_type sampler_type,
Shape target_shape=null)
{target_shape= new Shape(0,0);return new Operator("SpatialTransformer")
.SetParam("transform_type", transform_type)
.SetParam("sampler_type", sampler_type)
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.SetInput("loc", loc)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply spatial transformer to input feature map.
/// </summary>
/// <param name="data">Input data to the SpatialTransformerOp.</param>/// <param name="loc">localisation net, the output dim should be 6 when transform_type is affine, and the name of loc symbol should better starts with 'stn_loc', so that initialization it with iddentify tranform, or you shold initialize the weight and bias by yourself.</param>/// <param name="transform_type">transformation type</param>/// <param name="sampler_type">sampling type</param>/// <param name="target_shape">output shape(h, w) of spatial transformer: (y, x)</param> /// <returns>returns new symbol</returns>
public Symbol SpatialTransformer(Symbol data,
Symbol loc,
SpatialTransformertransform_type transform_type,
SpatialTransformersampler_type sampler_type,
Shape target_shape=null)
{target_shape= new Shape(0,0);return new Operator("SpatialTransformer")
.SetParam("transform_type", transform_type)
.SetParam("sampler_type", sampler_type)
.SetParam("target_shape", target_shape)
.SetInput("data", data)
.SetInput("loc", loc)
.CreateSymbol();
}
/// <summary>
/// Support Vector Machine based transformation on input, backprop L2-SVM
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to svm.</param>/// <param name="label">Label data.</param>/// <param name="margin">Scale the DType(param_.margin) for activation size</param>/// <param name="regularization_coefficient">Scale the coefficient responsible for balacing coefficient size and error tradeoff</param>/// <param name="use_linear">If set true, uses L1-SVM objective function. Default uses L2-SVM objective</param> /// <returns>returns new symbol</returns>
public Symbol SVMOutput(string symbol_name,
Symbol data,
Symbol label,
float margin=1f,
float regularization_coefficient=1f,
bool use_linear=false)
{return new Operator("SVMOutput")
.SetParam("margin", margin)
.SetParam("regularization_coefficient", regularization_coefficient)
.SetParam("use_linear", use_linear)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Support Vector Machine based transformation on input, backprop L2-SVM
/// </summary>
/// <param name="data">Input data to svm.</param>/// <param name="label">Label data.</param>/// <param name="margin">Scale the DType(param_.margin) for activation size</param>/// <param name="regularization_coefficient">Scale the coefficient responsible for balacing coefficient size and error tradeoff</param>/// <param name="use_linear">If set true, uses L1-SVM objective function. Default uses L2-SVM objective</param> /// <returns>returns new symbol</returns>
public Symbol SVMOutput(Symbol data,
Symbol label,
float margin=1f,
float regularization_coefficient=1f,
bool use_linear=false)
{return new Operator("SVMOutput")
.SetParam("margin", margin)
.SetParam("regularization_coefficient", regularization_coefficient)
.SetParam("use_linear", use_linear)
.SetInput("data", data)
.SetInput("label", label)
.CreateSymbol();
}
/// <summary>
/// Apply swapaxis to input.
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Input data to the SwapAxisOp.</param>/// <param name="dim1">the first axis to be swapped.</param>/// <param name="dim2">the second axis to be swapped.</param> /// <returns>returns new symbol</returns>
public Symbol SwapAxis(string symbol_name,
Symbol data,
int dim1=0,
int dim2=0)
{return new Operator("SwapAxis")
.SetParam("dim1", dim1)
.SetParam("dim2", dim2)
.SetInput("data", data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Apply swapaxis to input.
/// </summary>
/// <param name="data">Input data to the SwapAxisOp.</param>/// <param name="dim1">the first axis to be swapped.</param>/// <param name="dim2">the second axis to be swapped.</param> /// <returns>returns new symbol</returns>
public Symbol SwapAxis(Symbol data,
int dim1=0,
int dim2=0)
{return new Operator("SwapAxis")
.SetParam("dim1", dim1)
.SetParam("dim2", dim2)
.SetInput("data", data)
.CreateSymbol();
}
/// <summary>
/// upsampling method
/// </summary>
public enum UpSamplingsample_type
{bilinear,
nearest
};/// <summary>
/// How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
/// </summary>
public enum UpSamplingmulti_input_mode
{concat,
sum
};/// <summary>
/// Perform nearest neighboor/bilinear up sampling to inputs
/// </summary>
/// <param name="symbol_name">name of the resulting symbol</param>/// <param name="data">Array of tensors to upsample</param>/// <param name="scale">Up sampling scale</param>/// <param name="sample_type">upsampling method</param>/// <param name="num_args">Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.</param>/// <param name="num_filter">Input filter. Only used by bilinear sample_type.</param>/// <param name="multi_input_mode">How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.</param>/// <param name="workspace">Tmp workspace for deconvolution (MB)</param> /// <returns>returns new symbol</returns>
public Symbol UpSampling(string symbol_name,
Symbol[] data,
int scale,
UpSamplingsample_type sample_type,
int num_args,
int num_filter=0,
UpSamplingmulti_input_mode multi_input_mode=UpSamplingmulti_input_mode.concat,
long workspace=512)
{return new Operator("UpSampling")
.SetParam("scale", scale)
.SetParam("sample_type", sample_type)
.SetParam("num_args", num_args)
.SetParam("num_filter", num_filter)
.SetParam("multi_input_mode", multi_input_mode)
.SetParam("workspace", workspace)
.AddInput(data)
.CreateSymbol(symbol_name);
}
/// <summary>
/// Perform nearest neighboor/bilinear up sampling to inputs
/// </summary>
/// <param name="data">Array of tensors to upsample</param>/// <param name="scale">Up sampling scale</param>/// <param name="sample_type">upsampling method</param>/// <param name="num_args">Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.</param>/// <param name="num_filter">Input filter. Only used by bilinear sample_type.</param>/// <param name="multi_input_mode">How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.</param>/// <param name="workspace">Tmp workspace for deconvolution (MB)</param> /// <returns>returns new symbol</returns>
public Symbol UpSampling(Symbol[] data,
int scale,
UpSamplingsample_type sample_type,
int num_args,
int num_filter=0,
UpSamplingmulti_input_mode multi_input_mode=UpSamplingmulti_input_mode.concat,
long workspace=512)
{return new Operator("UpSampling")
.SetParam("scale", scale)
.SetParam("sample_type", sample_type)
.SetParam("num_args", num_args)
.SetParam("num_filter", num_filter)
.SetParam("multi_input_mode", multi_input_mode)
.SetParam("workspace", workspace)
.AddInput(data)
.CreateSymbol();
}
}
}
