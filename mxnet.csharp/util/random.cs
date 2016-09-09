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

            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_sample_uniform", out funcHandle);

            var input = IntPtr.Zero;
            var output = @out.Handle;

            var paramKeys = new string[] {"low", "high", "shape"};
            var paramVals = new string[]
            {
                low.ToString(CultureInfo.InvariantCulture),
                high.ToString(CultureInfo.InvariantCulture),
                @out.GetShape().ToString()
            };

            Util.CallCheck(NativeMethods.MXFuncInvokeEx(
            funcHandle,
            ref input,
            new float[0],
            ref output,
            paramKeys.Length,
            paramKeys,
            paramVals
            ));
        }

        public static void Normal(float loc, float scale, NdArray @out)
        {

            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_sample_normal", out funcHandle);

            var input = IntPtr.Zero;
            var output = @out.Handle;

            var paramKeys = new string[] { "loc", "scale", "shape" };
            var paramVals = new string[]
            {
                loc.ToString(),
                scale.ToString(),
                @out.GetShape().ToString()
            };

            Util.CallCheck(NativeMethods.MXFuncInvokeEx(
            funcHandle,
            ref input,
            new float[0],
            ref output,
            paramKeys.Length,
            paramKeys,
            paramVals
            ));
        }
    }
}
