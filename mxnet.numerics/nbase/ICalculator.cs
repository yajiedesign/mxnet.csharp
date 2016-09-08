using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public interface ICalculator<T>
    {
        T Compare(T a, T b);
        T Sum(T[] data);
        int Argmax(T[] storage);
        T[] Log(T[] data);
        T[] Abs(T[] data);
        T Mean(T[] data);
        T[] Minus(T[] data);
        T[] Minus(T[] l, T[] r);
        T[] Pow(T[] data, T y);
    }
}
