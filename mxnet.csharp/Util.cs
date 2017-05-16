using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{

    public static class LineHelp
    {
        public static TValue GetValueOrDefault<TKey, TValue>
            (this IDictionary<TKey, TValue> dictionary,
                TKey key,
                TValue defaultValue)
        {
            TValue value;
            return dictionary.TryGetValue(key, out value) ? value : defaultValue;
        }

        public static TValue GetValueOrDefault<TKey, TValue>
            (this IDictionary<TKey, TValue> dictionary,
                TKey key,
                Func<TValue> defaultValueProvider)
        {
            TValue value;
            return dictionary.TryGetValue(key, out value)
                ? value
                : defaultValueProvider();
        }

        public static IEnumerable<TResult> Zip<T1, T2, T3, TResult>(
            this IEnumerable<T1> source,
            IEnumerable<T2> second,
            IEnumerable<T3> third,
            Func<T1, T2, T3, TResult> func)
        {
            using (var e1 = source.GetEnumerator())
            using (var e2 = second.GetEnumerator())
            using (var e3 = third.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext())
                    yield return func(e1.Current, e2.Current, e3.Current);
            }
        }
    }

    public static class Util
    {

        public static readonly Dictionary<Type, int> DtypeNpToMx = new Dictionary<Type, int>
        {
            {typeof(float), 0},
            {typeof(double), 1},
          //  {typeof(np.float16), 2},
            {typeof(byte), 3},
            {typeof(int), 4}
        };


        public static readonly Dictionary<int, Type> DtypeMxToNp = new Dictionary<int, Type>
        {
            {0, typeof(float)},
            {1, typeof(double)},
            // {  2 , typeof(np.float16)},
            {3, typeof(byte)},
            {4, typeof(int)}
        };


        public static long Prod(uint[] shape)
        {
            return shape.Aggregate((long)1, (a, b) => (a * b));
        }


        public static void CallCheck(int ret)
        {
            if (ret != 0)
            {
                throw new Exception(NativeMethods.MxGetLastError());
            }
        }
        public static void NnCallCheck(int ret)
        {
            if (ret != 0)
            {
                throw new Exception(NativeMethods.NnGetLastError());
            }
        }

        public static void Assert(bool ret,string str=null)
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



        public static string EnumToString<TEnum>(TEnum? _enum, List<string> convert) where TEnum : struct, IConvertible
        {
            if (_enum.HasValue)
            {
                var v = _enum.Value as object;
                return convert[(int)v];
            }

            return null;

        }
    }
}
