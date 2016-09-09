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
        public static void Uniform(float low, float high,NDArray @out)
        {

            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_sample_uniform", out func_handle);

            var input = IntPtr.Zero;
            var output = @out.get_handle();

            var param_keys = new string[] {"low", "high", "shape"};
            var param_vals = new string[]
            {
                low.ToString(CultureInfo.InvariantCulture),
                high.ToString(CultureInfo.InvariantCulture),
                @out.get_shape().ToString()
            };

            Util.CallCheck(NativeMethods.MXFuncInvokeEx(
            func_handle,
            ref input,
            new float[0],
            ref output,
            param_keys.Length,
            param_keys,
            param_vals
            ));
        }

        public static void Normal(float loc, float scale, NDArray @out)
        {

            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_sample_normal", out func_handle);

            var input = IntPtr.Zero;
            var output = @out.get_handle();

            var param_keys = new string[] { "loc", "scale", "shape" };
            var param_vals = new string[]
            {
                loc.ToString(),
                scale.ToString(),
                @out.get_shape().ToString()
            };

            Util.CallCheck(NativeMethods.MXFuncInvokeEx(
            func_handle,
            ref input,
            new float[0],
            ref output,
            param_keys.Length,
            param_keys,
            param_vals
            ));
        }
    }
}
