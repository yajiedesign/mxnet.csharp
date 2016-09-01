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
        public static void uniform(float low, float high,NDArray @out)
        {

            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_sample_uniform", out func_handle);

            var input = IntPtr.Zero;
            var output = @out.Get_handle();

            var param_keys = new string[] {"low", "high", "shape"};
            var param_vals = new string[]
            {
                low.ToString(CultureInfo.InvariantCulture),
                high.ToString(CultureInfo.InvariantCulture),
                @out.Get_shape().ToString()
            };

            Util.call_check(NativeMethods.MXFuncInvokeEx(
            func_handle,
            ref input,
            new float[0],
            ref output,
            param_keys.Length,
            param_keys,
            param_vals
            ));
        }

        public static void normal(float loc, float scale, NDArray @out)
        {

            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_sample_normal", out func_handle);

            var input = IntPtr.Zero;
            var output = @out.Get_handle();

            var param_keys = new string[] { "loc", "scale", "shape" };
            var param_vals = new string[]
            {
                loc.ToString(),
                scale.ToString(),
                @out.Get_shape().ToString()
            };

            Util.call_check(NativeMethods.MXFuncInvokeEx(
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
