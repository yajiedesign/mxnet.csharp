using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public static class Util
    {

        public static readonly Dictionary<Type, int> _DTYPE_NP_TO_MX = new Dictionary<Type, int>
        {
            {typeof(float), 0},
            {typeof(double), 1},
          //  {typeof(np.float16), 2},
            {typeof(byte), 3},
            {typeof(int), 4}
        };


        public static readonly Dictionary<int, Type> _DTYPE_MX_TO_NP = new Dictionary<int, Type>
        {
            {0, typeof(float)},
            {1, typeof(double)},
            // {  2 , typeof(np.float16)},
            {3, typeof(byte)},
            {4, typeof(int)}
        };

        public static void _check_arguments(Symbol symbol)
        {
            var arg_names = symbol.ListArguments();
            var arg_names_duplicate = arg_names.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));
            foreach (var name in arg_names_duplicate)
            {
                throw new Exception($"Find duplicated argument name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", arg_names)}");
            }
            var aux_names = symbol.ListAuxiliaryStates();
            var aux_names_duplicate = aux_names.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));

            foreach (var name in aux_names_duplicate)
            {
                throw new Exception($"Find duplicated auxiliary name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", arg_names)}");
            }
        }


        public static long prod(uint[] shape)
        {
            return shape.Aggregate((long)1, (a, b) => (a * b));
        }


        public static void call_check(int ret)
        {
            if (ret != 0)
            {
                throw new Exception(NativeMethods.MXGetLastError());
            }
        }

        public static void assert(bool ret,string str=null)
        {
            if (!ret)
            {
                if (String.IsNullOrWhiteSpace(str))
                {
                    throw new Exception("Assert faild");
                }
                else
                {
                    throw new Exception(str);
                }
            }
         
        }
    }
}
