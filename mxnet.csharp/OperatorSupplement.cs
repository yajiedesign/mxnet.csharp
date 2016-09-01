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
            return new Operator("_MinusScalar").AddInput(lhs)
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

    }
    public  partial class OperatorWarp
    {
        /// <summary>
        /// Slice input equally along specified axis
        /// </summary>
        /// <param name="data"></param>
        /// <param name="num_outputs">Number of outputs to be sliced.</param>
        /// <param name="axis">Dimension along which to slice.</param>
        /// <param name="squeeze_axis">If true AND the sliced dimension becomes 1, squeeze that dimension.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceChannel(Symbol data,
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


        /// <summary>
        /// Slice input equally along specified axis
        /// </summary>
        /// <param name="symbol_name">name of the resulting symbol</param>
        /// <param name="data"></param>
        /// <param name="num_outputs">Number of outputs to be sliced.</param>
        /// <param name="axis">Dimension along which to slice.</param>
        /// <param name="squeeze_axis">If true AND the sliced dimension becomes 1, squeeze that dimension.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceChannel(string symbol_name,
                               Symbol data,
                               int num_outputs,
                               int axis = 1,
                               bool squeeze_axis = false)
        {
            return new Operator("SliceChannel")
                     .SetParam("num_outputs", num_outputs)
                     .SetParam("axis", axis)
                     .SetParam("squeeze_axis", squeeze_axis).AddInput(data)
                     .CreateSymbol(symbol_name);
        }

        public static Symbol Max(Symbol lhs, float scalar)
        {
           return Operator._MaximumScalar(lhs, scalar);
        }
        public static Symbol Max(float scalar, Symbol rhs)
        {
            return Operator._MaximumScalar(rhs, scalar);
        }
        public static Symbol Max(Symbol lhs, Symbol rhs)
        {
            return Operator._Maximum(lhs, rhs);
        }

        public static Symbol Min(Symbol lhs, float scalar)
        {
            return Operator._MinimumScalar(lhs, scalar);
        }
        public static Symbol Min(float scalar, Symbol rhs)
        {
            return Operator._MinimumScalar(rhs, scalar);
        }
        public static Symbol Min(Symbol lhs, Symbol rhs)
        {
            return Operator._Minimum(lhs, rhs);
        }

    }
}
