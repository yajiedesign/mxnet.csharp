using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;

namespace mxnet.numerics.int32
{
    public struct Int32Calculator : ICalculator<int>
    {
        public int Compare(int a, int b)
        {
            return (a - b) < float.Epsilon ? 1 : 0;
        }
    }

    public class Int32NArrayView : NArrayView<int, Int32Calculator, Int32NArrayView>
    {
        public Int32NArrayView()
        {

        }
        public Int32NArrayView(Shape shape, NArray<int, Int32Calculator, Int32NArrayView> src) : base(shape, src)
        {
        }
    }

    public class Int32NArray : NArrayStorage<int, Int32Calculator, Int32NArrayView>
    {
        public Int32NArray(Shape shape) : base(shape)
        {
        }

        public Int32NArray(Shape shape, int[] data) : base(shape, data)
        {
        }
    }
}
