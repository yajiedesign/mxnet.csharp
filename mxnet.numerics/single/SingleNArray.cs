using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.int32;
using mxnet.numerics.nbase;

namespace mxnet.numerics.single
{
    public struct SingleCalculator : ICalculator<float>
    {
        public float Compare(float a, float b)
        {
            return (Math.Abs (a - b)) < float.Epsilon ? 1 : 0;
        }

        public float Sum(float[] data)
        {
            return data.Sum();
        }

        public int Argmax(float[] data)
        {
            return !data.Any()
                ? -1
                : data
                    .Select((value, index) => new {Value = value, Index = index})
                    .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                    .Index;
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


        public  Int32NArray ToInt32()
        {
            return new Int32NArray(shape, storage. Select(s => (int)s).ToArray());
        }
    }
}
