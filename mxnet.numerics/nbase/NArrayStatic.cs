using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public partial class NArray<T, TC, TOut>
    {
        public static TOut Concatenate(int axis = 0 ,params TOut[] input_list)
        {
            var ndim = input_list.First().shape.ndim;
            if (input_list.Any(a => a.shape.ndim != ndim))
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
                var size = input_list.First().shape[i];
                if (input_list.Select(s => s.shape[i]).Any(a => a != size))
                {
                    throw new ArgumentException("input_list shape not match");
                }
            }

            var t_axis_size = (uint) input_list.Select(s => (int)s.shape[axis]).Sum();

            var dst_dim = (uint[])input_list.First().shape.data.Clone();
            dst_dim[axis] = t_axis_size;

            TOut ret = new TOut();
            ret.Init(new Shape(dst_dim));

            var dst_sclie = Enumerable.Range(0, input_list.Length).Select(s => (Slice) ":").ToArray();

            var dst_index = 0;
            for (int input_index = 0; input_index < input_list.Length; input_index++)
            {
                var curr = input_list[input_index];
                var curr_axis_dim = (int) curr.shape[axis];
                dst_sclie[axis] =new  Slice(dst_index, dst_index +curr_axis_dim);

                ret[dst_sclie] = curr;

                dst_index += curr_axis_dim;
            }

            return ret;
        }

        public static TOut Pow(TOut x, T y)
        {
            TOut ret = new TOut();
            ret.Init(x.shape, Calculator.Pow(x.data,y));
            return ret;
        }

        public static TOut Abs(TOut x)
        {
            TOut ret = new TOut();
            ret.Init(x.shape, Calculator.Abs(x.data));
            return ret;
        }

    }
}
