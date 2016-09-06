using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;

namespace mxnet.numerics.int32
{
    public partial struct Int32Calculator : ICalculator<int>
    {
        public int Compare(int a, int b)
        {
            return a == b ? 1 : 0;
        }
    }

}
