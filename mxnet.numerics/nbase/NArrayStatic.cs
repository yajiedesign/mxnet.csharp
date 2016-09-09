using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public partial class NArray<T, TC, TOut>
    {
        public static TOut Concatenate(int axis = 0 ,params TOut[] inputList)
        {
            var ndim = inputList.First().Shape.Ndim;
            if (inputList.Any(a => a.Shape.Ndim != ndim))
            {
                throw new ArgumentException("input_list dim not match");
            }
            // check shape
            for (int i = 0; i < ndim; i++)
            {
                if (i == axis)
                {
                    continue;
                }
                var size = inputList.First().Shape[i];
                if (inputList.Select(s => s.Shape[i]).Any(a => a != size))
                {
                    throw new ArgumentException("input_list shape not match");
                }
            }

            var tAxisSize = (uint) inputList.Select(s => (int)s.Shape[axis]).Sum();

            var dstDim = (uint[])inputList.First().Shape.Data.Clone();
            dstDim[axis] = tAxisSize;

            TOut ret = new TOut();
            ret.Init(new Shape(dstDim));

            var dstSclie = Enumerable.Range(0, inputList.Length).Select(s => (Slice) ":").ToArray();

            var dstIndex = 0;
            for (int inputIndex = 0; inputIndex < inputList.Length; inputIndex++)
            {
                var curr = inputList[inputIndex];
                var currAxisDim = (int) curr.Shape[axis];
                dstSclie[axis] =new  Slice(dstIndex, dstIndex +currAxisDim);

                ret[dstSclie] = curr;

                dstIndex += currAxisDim;
            }

            return ret;
        }

        public static TOut Pow(TOut x, T y)
        {
            TOut ret = new TOut();
            ret.Init(x.Shape, Calculator.Pow(x.Data,y));
            return ret;
        }

        public static TOut Abs(TOut x)
        {
            TOut ret = new TOut();
            ret.Init(x.Shape, Calculator.Abs(x.Data));
            return ret;
        }

    }
}
