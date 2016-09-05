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
            return a == b ? 1 : 0;
        }

        public int Sum(int[] data)
        {
            return data.Sum();
        }

        public int Argmax(int[] data)
        {
          return  !data.Any()
                ? -1
                : data
                    .Select((value, index) => new { Value = value, Index = index })
                    .Aggregate((a, b) => (a.Value >= b.Value) ? a : b)
                    .Index;
        }
    }



    public class Int32NArray : NArray<int, Int32Calculator, Int32NArray>
    {
        public Int32NArray()
        {

        }
        public Int32NArray(Shape shape) : base(shape)
        {
        }

        public Int32NArray(Shape shape, int[] data) : base(shape, data)
        {
        }

     
    }
}
