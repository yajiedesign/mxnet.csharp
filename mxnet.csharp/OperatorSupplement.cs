using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public partial class Operator
    {
        public static Symbol _Plus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Plus").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Mul(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mul").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Minus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minus").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Div(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Div").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Power(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Power").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Maximum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Maximum").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _Minimum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minimum").AddInput(lhs, rhs)
                     .CreateSymbol();
        }
        public static Symbol _PlusScalar(Symbol lhs, float scalar)
        {
            return new Operator("_PlusScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _MinusScalar(Symbol lhs, float scalar)
        {
            return new  Operator("_MinusScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _RMinusScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RMinusScalar").AddInput(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _MulScalar(Symbol lhs, float scalar)
        {
            return new Operator("_MulScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _DivScalar(Symbol lhs, float scalar)
        {
            return new Operator("_DivScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _RDivScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RDivScalar").AddInput(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _PowerScalar(Symbol lhs, float scalar)
        {
            return new Operator("_PowerScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _RPowerScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RPowerScalar").AddInput(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _MaximumScalar(Symbol lhs, float scalar)
        {
            return new Operator("_MaximumScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _MinimumScalar(Symbol lhs, float scalar)
        {
            return new Operator("_MinimumScalar").AddInput(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol Crop( string symbol_name,
    int num_args,
    Symbol data,
    Symbol crop_like,
    Shape offset = null,
    Shape h_w = null,
    bool center_crop = false) {
            if (offset == null)
            {
                offset = new Shape(0, 0);
            }
            if (h_w == null)
            {
                h_w = new Shape(0, 0);
            }

            return new Operator("Crop")
    .SetParam("num_args", num_args)
    .SetParam("offset", offset)
    .SetParam("h_w", h_w)
    .SetParam("center_crop", center_crop)
    .SetInput("arg0", data)
    .SetInput("arg1", crop_like)
    .CreateSymbol(symbol_name);
    }


    /*!
     * \breif Slice input equally along specified axis.
     * \param data input symbol. 
     * \param num_outputs Number of outputs to be sliced. 
     * \param axis Dimension along which to slice. 
     * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that dimension. 
     * \return new symbol
     */
     Symbol SliceChannel(Symbol data,
                               int num_outputs,
                               int axis = 1,
                               bool squeeze_axis = false)
    {
        return new Operator("SliceChannel")
                 .SetParam("num_outputs", num_outputs)
                 .SetParam("axis", axis)
                 .SetParam("squeeze_axis", squeeze_axis).
                 AddInput(data)
                 .CreateSymbol();
    }


    /*!
     * \breif Slice input equally along specified axis.
     * \param symbol_name name of the resulting symbol.
     * \param data input symbol. 
     * \param num_outputs Number of outputs to be sliced. 
     * \param axis Dimension along which to slice. 
     * \param squeeze_axis If true AND the sliced dimension becomes 1, squeeze that dimension. 
     * \return new symbol
     */
     Symbol SliceChannel(string symbol_name,
                               Symbol data,
                               int num_outputs,
                               int axis = 1,
                               bool squeeze_axis = false)
    {
        return new  Operator("SliceChannel")
                 .SetParam("num_outputs", num_outputs)
                 .SetParam("axis", axis)
                 .SetParam("squeeze_axis", squeeze_axis).AddInput(data)
                 .CreateSymbol(symbol_name);
    }

    /*!
     * \breif Apply activation function to input.
     *        Softmax Activation is only available with CUDNN on GPUand will be
     *        computed at each location across channel if input is 4D.
     * \param symbol_name name of the resulting symbol.
     * \param data Input data to activation function. 
     * \param act_type Activation function to be applied. 
     * \return new symbol
     */
     Symbol Activation(  string symbol_name,
                             Symbol data,
                              string act_type)
    {
        Debug.Assert(act_type == "relu" ||
               act_type == "sigmoid" ||
               act_type == "softrelu" ||
               act_type == "tanh");
        return new Operator("Activation")
                 .SetParam("act_type", act_type)
                 .SetInput("data", data)
                 .CreateSymbol(symbol_name);
    }
}
}
