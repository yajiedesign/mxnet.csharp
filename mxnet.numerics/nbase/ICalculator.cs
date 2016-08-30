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
        T Sum(IQueryable<T> data);
    }
}
