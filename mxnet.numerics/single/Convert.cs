using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.int32;
using mxnet.numerics.single;

namespace mxnet.numerics.single
{
    public static class SingleConvert
    {
      public static  Int32NArray ToInt32(this SingleNArray input)
      {
          return new Int32NArray(input.Shape, input.Data.Select(s => (int) s).ToArray());


      }

    }
}
