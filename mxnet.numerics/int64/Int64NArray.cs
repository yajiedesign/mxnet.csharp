using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;

namespace mxnet.numerics.int64
{
    public partial struct Int64Calculator : ICalculator<long>
    {
        public long Compare(long a, long b)
        {
            return a == b ? 1 : 0;
        }
    }
}
