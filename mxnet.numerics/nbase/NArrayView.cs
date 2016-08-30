using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public class NArrayView<T, TCalculator, TView> : NArray<T, TCalculator, TView>
        where T : new()
        where TCalculator : ICalculator<T>, new()
        where TView : NArrayView<T, TCalculator, TView>, new()
    {
        protected NArrayView()
        {

        }

        public NArrayView(Shape shape, NArray<T, TCalculator, TView> src)
        {
            Shape = shape;
            Storage  = src.Data.ToArray();
        }
    }
}
