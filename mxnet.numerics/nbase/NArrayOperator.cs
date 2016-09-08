using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public partial class NArray<T, TC, TOut>
    {
        public static TOut operator-(NArray<T, TC, TOut> input)
        {
            TOut ret = new TOut();
            ret.Init(input.shape, Calculator.Minus(input.data));
            return ret;
        }

        public static TOut operator-(NArray<T, TC, TOut> inputl , NArray<T, TC, TOut> inputr)
        {
            if (inputl.shape != inputr.shape)
            {
                throw new ArgumentException("left and right shape not match");
            }

            TOut ret = new TOut();
            ret.Init(inputl.shape, Calculator.Minus(inputl.data, inputr.data));
            return ret;
        }

    }
}
