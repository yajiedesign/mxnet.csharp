using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FunctionHandle = System.IntPtr;

namespace mxnet.csharp.util
{
    class Random
    {
        public static void Uniform(float low, float high,NdArray @out)
        {
            NdArray.(@out, low, high);       
        }

        public static void Normal(float loc, float scale, NdArray @out)
        {
            NdArray.(@out, loc, scale);
        }
    }
}
