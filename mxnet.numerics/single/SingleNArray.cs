using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;

namespace mxnet.numerics.single
{
    public struct SingleCalculator : ICalculator<float>
    {
        public float Compare(float a, float b)
        {
            return (a - b) < float.Epsilon ? 1 : 0;
        }
    }

    public class SingleCreateNArrayView : ICreateNArrayView<SingleNArrayView, SingleNArray>
    {
        public SingleNArrayView Create(Shape shape, SingleNArray src)
        {
            return new SingleNArrayView(shape, src);
        }
    }
    public class SingleNArrayView : NArrayView<float, SingleCalculator, SingleNArrayView>
    {
        public SingleNArrayView()
        {

        }
        public SingleNArrayView(Shape shape, NArray<float, SingleCalculator, SingleNArrayView> src) : base(shape, src)
        {
        }
    }

    public class SingleNArray : NArrayStorage<float, SingleCalculator, SingleNArrayView>
    {
        public SingleNArray(Shape shape) : base(shape)
        {
        }

        public SingleNArray(Shape shape, float[] data) : base(shape, data)
        {
        }

    }
}
