using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    partial class Operator
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
    
}
}
