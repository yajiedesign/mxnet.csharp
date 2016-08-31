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
            return (Math.Abs (a - b)) < float.Epsilon ? 1 : 0;
        }

        public float Sum(IQueryable<float> data)
        {
            return data.Sum();
        }
    }

 

    public class SingleNArray : NArray<float, SingleCalculator, SingleNArray>
    {
        public SingleNArray()
        {

        }
        public SingleNArray(Shape shape) : base(shape)
        {
        }

        public SingleNArray(Shape shape, float[] data) : base(shape, data)
        {
        }

   
    }
}
